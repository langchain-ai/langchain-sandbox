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
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, Field

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
        enable_filesystem: bool = False,  # Novo: controle explícito do filesystem
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
        """Attach a file to the sandbox filesystem.
        
        Args:
            path: Path where the file should be created (relative to /sandbox)
            content: Content of the file as a string
            encoding: Text encoding for the file (default: utf-8)
        """
        # Auto-enable filesystem when files are attached
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=content,
            encoding=encoding,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Attached file: {path} ({len(content)} chars)")

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox filesystem.
        
        Args:
            path: Path where the file should be created (relative to /sandbox)
            content: Binary content of the file
        """
        # Auto-enable filesystem when files are attached
        self.enable_filesystem = True
        
        b64_content = base64.b64encode(content).decode("ascii")
        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=b64_content,
            encoding="binary",
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Attached binary file: {path} ({len(content)} bytes)")

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox filesystem.
        
        Args:
            path: Directory path to create (relative to /sandbox)
        """
        # Auto-enable filesystem when directories are created
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="mkdir",
            path=path,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Created directory: {path}")

    def read_file(self, path: str, *, encoding: str = "utf-8") -> None:
        """Queue a file read operation.
        
        Args:
            path: Path to read from (relative to /sandbox)
            encoding: Text encoding for the file (default: utf-8)
        
        Note: This queues the operation but doesn't return content immediately.
        Use this when you need to read files during code execution.
        """
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="read",
            path=path,
            encoding=encoding,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Queued read operation: {path}")

    def list_directory(self, path: str = ".") -> None:
        """Queue a directory listing operation.
        
        Args:
            path: Directory path to list (relative to /sandbox, default: current)
        """
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="list",
            path=path,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Queued list operation: {path}")

    def remove_path(self, path: str) -> None:
        """Queue a file or directory removal operation.
        
        Args:
            path: Path to remove (relative to /sandbox)
        """
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="remove",
            path=path,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Queued remove operation: {path}")

    def copy_path(self, source: str, destination: str) -> None:
        """Queue a file or directory copy operation.
        
        Args:
            source: Source path (relative to /sandbox)
            destination: Destination path (relative to /sandbox)
        """
        self.enable_filesystem = True
        
        operation = FileSystemOperation(
            operation="copy",
            path=source,
            destination=destination,
        )
        self._filesystem_operations.append(operation)
        logger.debug(f"Queued copy operation: {source} -> {destination}")

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

        # FILESYSTEM: Ativado se há operações ou foi explicitamente habilitado
        if self._filesystem_operations or self.enable_filesystem:
            # Construir operações filesystem se existem
            if self._filesystem_operations:
                fs_ops = []
                for op in self._filesystem_operations:
                    op_dict = {
                        "operation": op.operation,
                        "path": op.path,
                    }
                    if op.content is not None:
                        op_dict["content"] = op.content
                    if op.encoding is not None:
                        op_dict["encoding"] = op.encoding
                    if op.destination is not None:
                        op_dict["destination"] = op.destination
                    fs_ops.append(op_dict)
                
                cmd.extend(["-fs", json.dumps(fs_ops)])
                
                logger.debug(f"Filesystem enabled with {len(fs_ops)} operations")
                if len(fs_ops) <= 5:  # Log detalhes se poucas operações
                    for i, op in enumerate(fs_ops):
                        logger.debug(f"  Op {i+1}: {op['operation']} {op['path']}")
            else:
                # Filesystem habilitado mas sem operações iniciais
                cmd.extend(["-fs", "[]"])
                logger.debug("Filesystem enabled with no initial operations")

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

        # Debug logging
        logger.debug(f"Executing command: {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")

        # Create and run the subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Wait for process with a timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                # Extract JSON from output that may contain loading messages
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                filesystem_info = full_result.get("fileSystemInfo", None)
                filesystem_operations = full_result.get("fileSystemOperations", None)
                
                # Convert the Uint8Array to Python bytes
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
                
                # Log filesystem info if available
                if filesystem_info:
                    logger.debug(f"Filesystem: {filesystem_info['type']} at {filesystem_info['mountPoint']}")
                if filesystem_operations:
                    logger.debug(f"Filesystem operations completed: {len(filesystem_operations)}")
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
            # Optionally: log cancellation if needed
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

        # Debug logging
        logger.debug(f"Executing command: {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")

        try:
            # Run the subprocess with timeout
            process = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=False,  # Keep as bytes for proper decoding
                timeout=timeout_seconds,
                check=False,  # Don't raise on non-zero exit
            )

            stdout_bytes = process.stdout
            stderr_bytes = process.stderr

            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                # Extract JSON from output that may contain loading messages
                full_result = json.loads(stdout)
                stdout = full_result.get("stdout", None)
                stderr = full_result.get("stderr", None)
                result = full_result.get("result", None)
                status = "success" if full_result.get("success", False) else "error"
                session_metadata = full_result.get("sessionMetadata", None)
                filesystem_info = full_result.get("fileSystemInfo", None)
                filesystem_operations = full_result.get("fileSystemOperations", None)
                
                # Convert the Uint8Array to Python bytes
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
                
                # Log filesystem info if available
                if filesystem_info:
                    logger.debug(f"Filesystem: {filesystem_info['type']} at {filesystem_info['mountPoint']}")
                if filesystem_operations:
                    logger.debug(f"Filesystem operations completed: {len(filesystem_operations)}")
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


class PyodideSandboxTool(BaseTool):
    """Tool for running python code in a PyodideSandbox."""

    name: str = "python_code_sandbox"
    description: str = (
        "A secure Python code sandbox with filesystem support. Use this to execute python commands.\n"
        "- Input should be a valid python command.\n"
        "- To return output, you should print it out with `print(...)`.\n"
        "- Don't use f-strings when printing outputs.\n"
        "- If you need to make web requests, use `httpx.AsyncClient`.\n"
        "- Files can be read/written using standard Python file operations.\n"
        "- All file operations work within a sandboxed memory filesystem."
    )

    # Mirror the PyodideSandbox constructor arguments
    stateful: bool = False
    allow_env: list[str] | bool = False
    allow_read: list[str] | bool = False
    allow_write: list[str] | bool = False
    allow_net: list[str] | bool = False
    allow_run: list[str] | bool = False
    allow_ffi: list[str] | bool = False
    timeout_seconds: float | None
    """Timeout for code execution in seconds. By default set to 60 seconds."""
    node_modules_dir: str = "auto"
    enable_filesystem: bool = False  # NOVO: controle do filesystem

    _sandbox: PyodideSandbox
    _sync_sandbox: SyncPyodideSandbox

    def __init__(
        self,
        *,
        stateful: bool = False,
        timeout_seconds: float | None = 60,
        allow_net: list[str] | bool = False,
        enable_filesystem: bool = False,  # NOVO: habilitar filesystem explicitamente
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the tool.

        Args:
            stateful: Whether to use a stateful sandbox.
            timeout_seconds: Timeout for code execution in seconds.
            allow_net: configure network access.
            enable_filesystem: Enable filesystem operations in the sandbox.
                              This is automatically enabled when files are attached.
            **kwargs: Other attributes will be passed to the PyodideSandbox
        """
        super().__init__(
            stateful=stateful,
            timeout_seconds=timeout_seconds,
            allow_net=allow_net,
            enable_filesystem=enable_filesystem,
            **kwargs,
        )

        if self.stateful:
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
                # these fields will be ignored by the LLM
                # and automatically injected by LangGraph's ToolNode
                state: Annotated[dict[str, Any] | BaseModel, InjectedState]
                tool_call_id: Annotated[str, InjectedToolCallId]

        else:

            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""

                code: str = Field(description="Code to execute.")

        self.args_schema: type[BaseModel] = PyodideSandboxToolInput
        self._sandbox = PyodideSandbox(
            stateful=self.stateful,
            allow_env=self.allow_env,
            allow_read=self.allow_read,
            allow_write=self.allow_write,
            allow_net=self.allow_net,
            allow_run=self.allow_run,
            allow_ffi=self.allow_ffi,
            node_modules_dir=self.node_modules_dir,
            enable_filesystem=self.enable_filesystem,  # NOVO
        )
        # Initialize sync sandbox with deno check skipped since async sandbox already
        # checked
        self._sync_sandbox = SyncPyodideSandbox(
            stateful=self.stateful,
            allow_env=self.allow_env,
            allow_read=self.allow_read,
            allow_write=self.allow_write,
            allow_net=self.allow_net,
            allow_run=self.allow_run,
            allow_ffi=self.allow_ffi,
            node_modules_dir=self.node_modules_dir,
            enable_filesystem=self.enable_filesystem,  # NOVO
            skip_deno_check=True,  # Skip deno check since async sandbox already checked
        )

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

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox environment."""
        self._sandbox.attach_binary_file(path, content)
        self._sync_sandbox.attach_binary_file(path, content)

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox environment."""
        self._sandbox.create_directory(path)
        self._sync_sandbox.create_directory(path)

    def read_file(self, path: str, *, encoding: str = "utf-8") -> None:
        """Queue a file read operation for the next execution."""
        self._sandbox.read_file(path, encoding=encoding)
        self._sync_sandbox.read_file(path, encoding=encoding)

    def list_directory(self, path: str = ".") -> None:
        """Queue a directory listing operation for the next execution."""
        self._sandbox.list_directory(path)
        self._sync_sandbox.list_directory(path)

    def remove_path(self, path: str) -> None:
        """Queue a file/directory removal operation for the next execution."""
        self._sandbox.remove_path(path)
        self._sync_sandbox.remove_path(path)

    def copy_path(self, source: str, destination: str) -> None:
        """Queue a file/directory copy operation for the next execution."""
        self._sandbox.copy_path(source, destination)
        self._sync_sandbox.copy_path(source, destination)

    def clear_filesystem_operations(self) -> None:
        """Clear all queued filesystem operations."""
        self._sandbox.clear_filesystem_operations()
        self._sync_sandbox.clear_filesystem_operations()

    def _run(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:  # noqa: ANN401
        """Use the tool synchronously."""
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

            # if the tool is used with a stateful sandbox,
            # we need to update the graph state with the new session bytes and metadata
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
            # Para sandbox não stateful
            result = self._sync_sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

            # Tratamento mais robusto de erros
            if result.status == "error":
                error_msg = result.stderr if result.stderr else "Execution failed with unknown error"
                return f"Error during execution: {error_msg}"
            
            # Se foi sucesso, retornar stdout ou result
            if result.stdout:
                return result.stdout
                
            if result.result is not None:
                return str(result.result)
                
            return ""

    async def _arun(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:  # noqa: ANN401
        """Use the tool asynchronously."""
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

            # if the tool is used with a stateful sandbox,
            # we need to update the graph state with the new session bytes and metadata
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
            # Para sandbox não stateful
            result = await self._sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

            # Tratamento mais robusto de erros
            if result.status == "error":
                error_msg = result.stderr if result.stderr else "Execution failed with unknown error"
                return f"Error during execution: {error_msg}"
            
            # Se foi sucesso, retornar stdout ou result
            if result.stdout:
                return result.stdout
                
            if result.result is not None:
                return str(result.result)
                
            return ""