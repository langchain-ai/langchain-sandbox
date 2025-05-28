"""Python wrapper that calls pyodide & deno for code execution."""

import asyncio
import base64
import dataclasses
import json
import logging
import subprocess
import time
from typing import Annotated, Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, InjectedToolCallId
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


Status = Literal["success", "error"]


@dataclasses.dataclass(kw_only=True)
class CodeExecutionResult:
    """Container for code execution results."""

    result: Any = None
    stdout: str | None = None
    stderr: str | None = None
    status: Status
    execution_time: float
    session_metadata: dict | None = None
    session_bytes: bytes | None = None
    filesystem_info: dict | None = None
    filesystem_operations: list[dict] | None = None


@dataclasses.dataclass(kw_only=True)
class FileSystemOperation:
    """Container for filesystem operations."""
    
    operation: Literal["read", "write", "list", "mkdir", "exists", "remove", "copy"]
    path: str
    content: str | None = None
    encoding: str | None = None
    destination: str | None = None
    
    def to_dict(self) -> dict[str, str]:
        """Convert to dict for JSON serialization."""
        result = {
            "operation": self.operation,
            "path": self.path,
        }
        
        if self.content is not None:
            result["content"] = self.content
        if self.encoding is not None:
            result["encoding"] = self.encoding
        if self.destination is not None:
            result["destination"] = self.destination
            
        return result


# Published package name
PKG_NAME = "../pyodide-sandbox-js/main.ts"


def build_permission_flag(
    flag: str,
    *,
    value: bool | list[str],
) -> str | None:
    """Build a permission flag string based on the provided setting."""
    if value is True:
        return flag
    if isinstance(value, list) and value:
        return f"{flag}={','.join(value)}"
    return None


class BasePyodideSandbox:
    """Base class for PyodideSandbox implementations."""

    def __init__(
        self,
        *,
        stateful: bool = False,
        allow_env: list[str] | bool = False,
        allow_read: list[str] | bool = False,
        allow_write: list[str] | bool = False,
        allow_net: list[str] | bool = False,
        allow_run: list[str] | bool = False,
        allow_ffi: list[str] | bool = False,
        node_modules_dir: str = "auto",
        skip_deno_check: bool = False,
        enable_filesystem: bool = False,
    ) -> None:
        """Initialize the sandbox with specific Deno permissions."""
        self.stateful = stateful
        self.enable_filesystem = enable_filesystem
        self._filesystem_operations: list[FileSystemOperation] = []
        
        if not skip_deno_check:
            # Check if Deno is installed
            try:
                subprocess.run(["deno", "--version"], check=True, capture_output=True)  # noqa: S607, S603
            except subprocess.CalledProcessError as e:
                msg = "Deno is installed, but running it failed."
                raise RuntimeError(msg) from e
            except FileNotFoundError as e:
                msg = "Deno is not installed or not in PATH."
                raise RuntimeError(msg) from e

        # Define permission configurations
        perm_defs = [
            ("--allow-env", allow_env, None),
            ("--allow-read", allow_read, ["node_modules"]),
            ("--allow-write", allow_write, ["node_modules"]),
            ("--allow-net", allow_net, None),
            ("--allow-run", allow_run, None),
            ("--allow-ffi", allow_ffi, None),
        ]

        self.permissions = []
        for flag, value, defaults in perm_defs:
            perm = build_permission_flag(flag, value=value)
            if perm is None and defaults is not None:
                default_value = ",".join(defaults)
                perm = f"{flag}={default_value}"
            if perm:
                self.permissions.append(perm)

        self.permissions.append(f"--node-modules-dir={node_modules_dir}")

    def attach_file(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Attach a file to the sandbox filesystem."""
        self.enable_filesystem = True
        
        if not isinstance(content, str):
            raise ValueError("Content must be a string for text files")
        
        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=content,
            encoding=encoding,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Attached file: {path} ({len(content)} chars, encoding: {encoding})")

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox filesystem."""
        self.enable_filesystem = True
        
        if not isinstance(content, bytes):
            raise ValueError("Content must be bytes for binary files")
        
        b64_content = base64.b64encode(content).decode("ascii")
        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=b64_content,
            encoding="binary",
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Attached binary file: {path} ({len(content)} bytes -> {len(b64_content)} b64 chars)")

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox filesystem."""
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="mkdir",
            path=path,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Created directory: {path}")

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths."""
        files = []
        for op in self._filesystem_operations:
            if op.operation in ["write"]:
                files.append(op.path)
        return files

    def clear_filesystem_operations(self) -> None:
        """Clear all queued filesystem operations."""
        self._filesystem_operations.clear()
        logger.debug("Cleared filesystem operations")

    def _build_command(self, code: str, **kwargs) -> list[str]:
        cmd = ["deno", "run"]
        
        # Apply permissions
        cmd.extend(self.permissions)
        
        # Memory limit
        if kwargs.get('memory_limit_mb'):
            cmd.append(f"--v8-flags=--max-old-space-size={kwargs['memory_limit_mb']}")

        cmd.append(PKG_NAME)
        cmd.extend(["-c", code])

        # Stateful
        if self.stateful:
            cmd.extend(["-s"])

        # Session data
        if kwargs.get('session_bytes'):
            bytes_array = list(kwargs['session_bytes'])
            cmd.extend(["-b", json.dumps(bytes_array)])

        if kwargs.get('session_metadata'):
            cmd.extend(["-m", json.dumps(kwargs['session_metadata'])])

        # FILESYSTEM
        if self._filesystem_operations or self.enable_filesystem:
            if self._filesystem_operations:
                fs_ops = [op.to_dict() for op in self._filesystem_operations]
                fs_json = json.dumps(fs_ops, ensure_ascii=True, separators=(',', ':'))
                cmd.extend(["-x", fs_json])
                
                logger.debug(f"Filesystem enabled with {len(fs_ops)} operations")
            else:
                cmd.extend(["-x", "[]"])
                logger.debug("Filesystem enabled with no initial operations")

        logger.debug(f"Full command: {' '.join(cmd)}")
        return cmd


class PyodideSandbox(BasePyodideSandbox):
    """Asynchronous implementation of PyodideSandbox."""

    async def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously in a sandboxed Deno subprocess."""
        start_time = time.time()
        stdout = ""
        stderr = ""
        result = None
        status: Literal["success", "error"] = "success"

        cmd = self._build_command(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
        )

        # Create and run the subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                filesystem_info = full_result.get("fileSystemInfo", None)
                filesystem_operations = full_result.get("fileSystemOperations", None)
                
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                status = "error"
                filesystem_info = None
                filesystem_operations = None
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            status = "error"
            stderr = f"Execution timed out after {timeout_seconds} seconds"
            filesystem_info = None
            filesystem_operations = None
        except json.JSONDecodeError as e:
            status = "error"
            stderr = f"Failed to parse output as JSON: {e}\nRaw output: {stdout}"
            filesystem_info = None
            filesystem_operations = None
        except asyncio.CancelledError:
            pass
        
        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
            filesystem_info=filesystem_info,
            filesystem_operations=filesystem_operations,
        )


class SyncPyodideSandbox(BasePyodideSandbox):
    """Synchronous version of PyodideSandbox."""

    def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code synchronously in a sandboxed Deno subprocess."""
        start_time = time.time()
        stdout = ""
        result = None
        stderr: str
        status: Literal["success", "error"]

        cmd = self._build_command(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            memory_limit_mb=memory_limit_mb,
        )

        try:
            process = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                check=False,
            )

            stdout_bytes = process.stdout
            stderr_bytes = process.stderr

            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                filesystem_info = full_result.get("fileSystemInfo", None)
                filesystem_operations = full_result.get("fileSystemOperations", None)
                
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                status = "error"
                filesystem_info = None
                filesystem_operations = None

        except subprocess.TimeoutExpired:
            status = "error"
            stderr = f"Execution timed out after {timeout_seconds} seconds"
            filesystem_info = None
            filesystem_operations = None
        except json.JSONDecodeError as e:
            status = "error"
            stderr = f"Failed to parse output as JSON: {e}\nRaw output: {stdout}"
            filesystem_info = None
            filesystem_operations = None

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
            filesystem_info=filesystem_info,
            filesystem_operations=filesystem_operations,
        )


# Input schema para ferramentas
class PyodideSandboxInput(BaseModel):
    """Input schema for PyodideSandbox tool."""
    code: str = Field(description="Python code to execute.")


# =============================================================================
# CLASSE PRINCIPAL - Herda de BaseTool mas oferece acesso ao StructuredTool
# =============================================================================

class PyodideSandboxTool(BaseTool):
    """
    Flexible PyodideSandbox tool that can be used as BaseTool or StructuredTool.
    
    Usage examples:
    
    # As BaseTool (herança direta):
    tool = PyodideSandboxTool(enable_filesystem=True)
    result = tool.invoke({"code": "print('Hello')"})
    
    # As StructuredTool (via propriedade):
    tool = PyodideSandboxTool(enable_filesystem=True)
    result = tool.as_structured_tool().invoke({"code": "print('Hello')"})
    
    # Para agents que precisam de StructuredTool:
    agent = create_react_agent(llm, [tool.as_structured_tool()])
    
    # Para agents que aceitam BaseTool:
    agent = create_react_agent(llm, [tool])
    """

    name: str = "python_code_sandbox"
    
    # Mirror the PyodideSandbox constructor arguments
    stateful: bool = False
    allow_env: list[str] | bool = False
    allow_read: list[str] | bool = False
    allow_write: list[str] | bool = False
    allow_net: list[str] | bool = False
    allow_run: list[str] | bool = False
    allow_ffi: list[str] | bool = False
    timeout_seconds: float | None
    node_modules_dir: str = "auto"
    enable_filesystem: bool = False

    # CORREÇÃO: Usar PrivateAttr para atributos privados no Pydantic
    _sandbox: PyodideSandbox = PrivateAttr()
    _sync_sandbox: SyncPyodideSandbox = PrivateAttr() 
    _structured_tool: StructuredTool | None = PrivateAttr(default=None)
    _stateful: bool = PrivateAttr()
    _input_schema: type[BaseModel] = PrivateAttr()

    def _build_description(self) -> str:
        """Build the complete description string with attached files."""
        base = (
            "A secure Python code sandbox with filesystem support. "
            "Use this to execute python commands.\n"
            "- Input should be a valid python command.\n"
            "- To return output, you should print it out with `print(...)`.\n"
            "- Don't use f-strings when printing outputs.\n"
            "- If you need to make web requests, use `httpx.AsyncClient`.\n"
            "- Files can be read/written using standard Python file operations.\n"
            "- All file operations work within a sandboxed memory filesystem.\n"
            "- Check for attached files using: import os; print(os.listdir('.'))"
        )

        files = self._sandbox.get_attached_files()
        if files:
            base += "\n\n🗂️ ATTACHED FILES AVAILABLE:\n"
            base += "\n".join(f"  • {p}" for p in files)
            base += (
                "\nThese files are already loaded and ready to use with pandas, "
                "open(), etc."
            )
        return base

    def __init__(
        self,
        *,
        stateful: bool = False,
        timeout_seconds: float | None = 60,
        allow_net: list[str] | bool = False,
        enable_filesystem: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the tool."""
        
        if stateful:
            try:
                from langgraph.prebuilt import InjectedState
            except ImportError as e:
                error_msg = (
                    "The 'langgraph' package is required when using a stateful sandbox."
                    " Please install it with 'pip install langgraph'."
                )
                raise ImportError(error_msg) from e

            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""

                code: str = Field(description="Code to execute.")
                state: Annotated[dict[str, Any] | BaseModel, InjectedState]
                tool_call_id: Annotated[str, InjectedToolCallId]

        else:

            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""

                code: str = Field(description="Code to execute.")

        # Criar os sandboxes
        sandbox = PyodideSandbox(
            stateful=stateful,
            allow_env=kwargs.get('allow_env', False),
            allow_read=kwargs.get('allow_read', False),
            allow_write=kwargs.get('allow_write', False),
            allow_net=allow_net,
            allow_run=kwargs.get('allow_run', False),
            allow_ffi=kwargs.get('allow_ffi', False),
            node_modules_dir=kwargs.get('node_modules_dir', 'auto'),
            enable_filesystem=enable_filesystem,
        )
        sync_sandbox = SyncPyodideSandbox(
            stateful=stateful,
            allow_env=kwargs.get('allow_env', False),
            allow_read=kwargs.get('allow_read', False),
            allow_write=kwargs.get('allow_write', False),
            allow_net=allow_net,
            allow_run=kwargs.get('allow_run', False),
            allow_ffi=kwargs.get('allow_ffi', False),
            node_modules_dir=kwargs.get('node_modules_dir', 'auto'),
            enable_filesystem=enable_filesystem,
            skip_deno_check=True,
        )

        # Definir a descrição inicial
        initial_description = (
            "A secure Python code sandbox with filesystem support. "
            "Use this to execute python commands.\n"
            "- Input should be a valid python command.\n"
            "- To return output, you should print it out with `print(...)`.\n"
            "- Don't use f-strings when printing outputs.\n"
            "- If you need to make web requests, use `httpx.AsyncClient`.\n"
            "- Files can be read/written using standard Python file operations.\n"
       )
        
        # Chamar super().__init__() com a descrição calculada
        super().__init__(
            stateful=stateful,
            timeout_seconds=timeout_seconds,
            allow_net=allow_net,
            enable_filesystem=enable_filesystem,
            description=initial_description,
            args_schema=PyodideSandboxToolInput,
            **kwargs,
        )

        # IMPORTANTE: Definir atributos privados APÓS super().__init__()
        self._sandbox = sandbox
        self._sync_sandbox = sync_sandbox
        self._stateful = stateful
        self._input_schema = PyodideSandboxToolInput
        self._structured_tool = None

    def attach_file(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Attach a file to the sandbox environment."""
        self._sandbox.attach_file(path, content, encoding=encoding)
        self._sync_sandbox.attach_file(path, content, encoding=encoding)
        # Atualizar descrição em ambas as versões
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox environment."""
        self._sandbox.attach_binary_file(path, content)
        self._sync_sandbox.attach_binary_file(path, content)
        # Atualizar descrição em ambas as versões
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox environment."""
        self._sandbox.create_directory(path)
        self._sync_sandbox.create_directory(path)
        # Atualizar descrição em ambas as versões
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths."""
        return self._sandbox.get_attached_files()

    def clear_filesystem_operations(self) -> None:
        """Clear all filesystem operations and update description."""
        self._sandbox.clear_filesystem_operations()
        self._sync_sandbox.clear_filesystem_operations()
        # Atualizar descrição em ambas as versões
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def as_structured_tool(self) -> StructuredTool:
        """
        Return a StructuredTool version of this tool.
        
        This allows users to access the tool as a StructuredTool when needed,
        while maintaining the BaseTool interface as the primary one.
        """
        if self._structured_tool is None:
            self._structured_tool = StructuredTool.from_function(
                name=self.name,
                description=self.description,
                func=self._run_sync if not self._stateful else self._run_stateful_sync,
                args_schema=self._input_schema,
            )
        return self._structured_tool

    @property
    def tool(self) -> StructuredTool:
        """
        Legacy property for backwards compatibility.
        
        DEPRECATED: Use as_structured_tool() instead.
        """
        return self.as_structured_tool()

    def _run_sync(self, code: str) -> str:
        """Synchronous execution function for non-stateful mode."""
        result = self._sync_sandbox.execute(
            code, timeout_seconds=self.timeout_seconds
        )

        if result.status == "error":
            error_msg = result.stderr if result.stderr else "Execution failed with unknown error"
            return f"Error during execution: {error_msg}"
        
        if result.stdout:
            return result.stdout
            
        if result.result is not None:
            return str(result.result)
            
        return ""
    
    def _run_stateful_sync(
        self,
        code: str,
        state: dict[str, Any] | BaseModel,
        tool_call_id: str,
    ) -> Any:
        """Synchronous execution function for stateful mode."""
        required_keys = {"session_bytes", "session_metadata", "messages"}
        actual_keys = set(state) if isinstance(state, dict) else set(state.__dict__)
        if missing_keys := required_keys - actual_keys:
            error_msg = (
                "Input state is missing "
                f"the following required keys: {missing_keys}"
            )
            raise ValueError(error_msg)

        if isinstance(state, dict):
            session_bytes = state["session_bytes"]
            session_metadata = state["session_metadata"]
        else:
            session_bytes = state.session_bytes
            session_metadata = state.session_metadata

        result = self._sync_sandbox.execute(
            code,
            session_bytes=session_bytes,
            session_metadata=session_metadata,
            timeout_seconds=self.timeout_seconds,
        )
        
        if result.stderr:
            tool_result = f"Error during execution: {result.stderr}"
        else:
            tool_result = result.stdout

        from langgraph.types import Command

        return Command(
            update={
                "session_bytes": result.session_bytes,
                "session_metadata": result.session_metadata,
                "messages": [
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    def _run(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Use the tool synchronously (BaseTool interface)."""
        if self.stateful:
            return self._run_stateful_sync(code, state, tool_call_id)
        else:
            return self._run_sync(code)

    async def _arun(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        """Use the tool asynchronously (BaseTool interface)."""
        if self.stateful:
            required_keys = {"session_bytes", "session_metadata", "messages"}
            actual_keys = set(state) if isinstance(state, dict) else set(state.__dict__)
            if missing_keys := required_keys - actual_keys:
                error_msg = (
                    "Input state is missing "
                    f"the following required keys: {missing_keys}"
                )
                raise ValueError(error_msg)

            if isinstance(state, dict):
                session_bytes = state["session_bytes"]
                session_metadata = state["session_metadata"]
            else:
                session_bytes = state.session_bytes
                session_metadata = state.session_metadata

            result = await self._sandbox.execute(
                code,
                session_bytes=session_bytes,
                session_metadata=session_metadata,
                timeout_seconds=self.timeout_seconds,
            )
            
            if result.stderr:
                tool_result = f"Error during execution: {result.stderr}"
            else:
                tool_result = result.stdout

            from langgraph.types import Command

            return Command(
                update={
                    "session_bytes": result.session_bytes,
                    "session_metadata": result.session_metadata,
                    "messages": [
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        else:
            result = await self._sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

            if result.status == "error":
                error_msg = result.stderr if result.stderr else "Execution failed with unknown error"
                return f"Error during execution: {error_msg}"
            
            if result.stdout:
                return result.stdout
                
            if result.result is not None:
                return str(result.result)
                
            return ""


# =============================================================================
# WRAPPER ALTERNATIVO - Para manter compatibilidade com código existente
# =============================================================================

class PyodideSandboxDynamicTool:
    """
    Pure StructuredTool wrapper for PyodideSandbox (legacy compatibility).
    
    DEPRECATED: Use PyodideSandboxTool instead.
    """
    
    def __init__(self, **kwargs):
        """Initialize the wrapper - prefer PyodideSandboxTool instead."""
        logger.warning(
            "PyodideSandboxDynamicTool is deprecated. "
            "Use PyodideSandboxTool instead."
        )
        self._base_tool = PyodideSandboxTool(**kwargs)
        self.tool = self._base_tool.as_structured_tool()

    def attach_file(self, path: str, content: str, *, encoding: str = "utf-8") -> None:
        """Attach a file to the sandbox environment."""
        self._base_tool.attach_file(path, content, encoding=encoding)

    def attach_binary_file(self, path: str, content: bytes) -> None:
        """Attach a binary file to the sandbox environment."""
        self._base_tool.attach_binary_file(path, content)

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox environment."""
        self._base_tool.create_directory(path)

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths."""
        return self._base_tool.get_attached_files()

    def clear_filesystem_operations(self) -> None:
        """Clear all filesystem operations and update description."""
        self._base_tool.clear_filesystem_operations()

    def invoke(self, input_data: dict[str, Any]) -> str:
        """Direct invoke method for easier usage."""
        return self.tool.invoke(input_data)

    async def ainvoke(self, input_data: dict[str, Any]) -> str:
        """Async direct invoke method for easier usage."""
        return await self.tool.ainvoke(input_data)