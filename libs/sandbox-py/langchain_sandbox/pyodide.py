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
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
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
    """Container for filesystem operations.

    This class encapsulates a single filesystem operation that can be performed
    within the sandboxed environment. Operations are serialized to JSON and
    passed to the Deno subprocess for execution.

    Supported operations:
    - write: Create or write a file
    - read: Read file contents
    - mkdir: Create a directory
    - list: List directory contents
    - exists: Check if file/directory exists
    - remove: Delete file/directory
    - copy: Copy file/directory
    """

    operation: Literal["read", "write", "list", "mkdir", "exists", "remove", "copy"]
    path: str
    content: str | None = None
    encoding: str | None = None
    destination: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
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
PKG_NAME = "jsr:@langchain/pyodide-sandbox@0.0.4"
#PKG_NAME = "../pyodide-sandbox-js/main.ts"  # noqa: ERA001

def build_permission_flag(
    flag: str,
    *,
    value: bool | list[str],
) -> str | None:
    """Build a permission flag string based on the provided setting.

    Args:
        flag: The base permission flag (e.g., "--allow-read").
        value: Either a boolean (True for unrestricted access, False for no access)
                 or a list of allowed items.

    Returns:
        A string with the permission flag and items, or None if no permission should
        be added.
    """
    if value is True:
        return flag
    if isinstance(value, list) and value:
        return f"{flag}={','.join(value)}"
    return None


class BasePyodideSandbox:
    """Base class for PyodideSandbox implementations.

    This class provides the common initialization and configuration logic for both
    synchronous and asynchronous PyodideSandbox implementations.

    The sandbox leverages Deno's security model to create a secure runtime for
    executing untrusted Python code. It works by spawning a Deno subprocess that loads
    Pyodide (Python compiled to WebAssembly) and executes the provided code in an
    isolated environment.

    Security features:
    - Configurable permissions for file system, network, and environment access
    - Support for execution timeouts to prevent infinite loops
    - Memory usage monitoring
    - Process isolation via Deno's security sandbox
    - In-memory filesystem with file attachment capabilities

    The sandbox supports fine-grained permission control through its initializer:
    - Restrict network access to specific domains
    - Limit file system access to specific directories
    - Control environment variable access
    - Prevent subprocess execution and FFI
    - Attach files to in-memory filesystem before execution
    """

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
        """Initialize the sandbox with specific Deno permissions.

        This method configures the security permissions for the Deno subprocess that
        will execute Python code via Pyodide. By default, all permissions are
        disabled (False) for maximum security. Permissions can be enabled selectively
        based on the needs of the code being executed.

        Args:
            stateful: Whether to use a stateful session. If True, `sandbox.execute`
                will include session metadata and the session bytes containing the
                session state (variables, imports, etc.) in the execution result.
                This allows saving and reusing the session state between executions.

            allow_env: Environment variable access configuration:
                - False: No environment access (default, most secure)
                - True: Unrestricted access to all environment variables
                - List[str]: Access restricted to specific environment variables, e.g.
                  ["PATH", "PYTHONPATH"]

            allow_read: File system read access configuration:
                - False: No file system read access (default, most secure)
                - True: Unrestricted read access to the file system
                - List[str]: Read access restricted to specific paths, e.g.
                  ["/tmp/sandbox", "./data"]

                  By default allows read from node_modules

            allow_write: File system write access configuration:
                - False: No file system write access (default, most secure)
                - True: Unrestricted write access to the file system
                - List[str]: Write access restricted to specific paths, e.g.
                  ["/tmp/sandbox/output"]

                  By default allows write to node_modules

            allow_net: Network access configuration:
                - False: No network access (default, most secure)
                - True: Unrestricted network access
                - List[str]: Network access restricted to specific domains/IPs, e.g.
                  ["api.example.com", "data.example.org:8080"]

            allow_run: Subprocess execution configuration:
                - False: No subprocess execution allowed (default, most secure)
                - True: Unrestricted subprocess execution
                - List[str]: Subprocess execution restricted to specific commands, e.g.
                  ["python", "git"]

            allow_ffi: Foreign Function Interface access configuration:
                - False: No FFI access (default, most secure)
                - True: Unrestricted FFI access
                - List[str]: FFI access restricted to specific libraries, e.g.
                  ["/usr/lib/libm.so"]

            node_modules_dir: Directory for Node.js modules. Set to "auto" to use
                the default directory for Deno modules.
            skip_deno_check: If True, skip the check for Deno installation.
            enable_filesystem: If True, enable in-memory filesystem support for
                attaching files and directories to the sandbox environment.
        """
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

        # Define permission configurations:
        # each tuple contains (flag, setting, defaults)
        perm_defs = [
            ("--allow-env", allow_env, None),
            # For file system permissions, if no permission is specified,
            # force node_modules
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
        """Attach a text file to the sandbox filesystem.

        This method queues a file to be created in the sandbox's in-memory
        filesystem when code is executed. The file will be available for
        reading and manipulation within the Python environment.

        Args:
            path: File path within the sandbox filesystem
            content: Text content of the file
            encoding: Text encoding (default: utf-8)

        Raises:
            TypeError: If content is not a string
        """
        self.enable_filesystem = True

        if not isinstance(content, str):
            msg = "Content must be a string for text files"
            raise TypeError(msg)

        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=content,
            encoding=encoding,
        )
        self._filesystem_operations.append(operation)
        logger.debug(
            "Attached file: %s (%d chars, encoding: %s)",
            path,
            len(content),
            encoding,
        )

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox filesystem.

        This method queues a binary file to be created in the sandbox's in-memory
        filesystem when code is executed. The content is base64-encoded for
        transport to the sandbox environment.

        Args:
            path: File path within the sandbox filesystem
            content: Binary content of the file

        Raises:
            TypeError: If content is not bytes
        """
        self.enable_filesystem = True

        if not isinstance(content, bytes):
            msg = "Content must be bytes for binary files"
            raise TypeError(msg)

        b64_content = base64.b64encode(content).decode("ascii")
        operation = FileSystemOperation(
            operation="write",
            path=path,
            content=b64_content,
            encoding="binary",
        )
        self._filesystem_operations.append(operation)
        logger.debug(
            "Attached binary file: %s (%d bytes -> %d b64 chars)",
            path,
            len(content),
            len(b64_content),
        )

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox filesystem.

        This method queues a directory to be created in the sandbox's in-memory
        filesystem when code is executed.

        Args:
            path: Directory path within the sandbox filesystem
        """
        self.enable_filesystem = True

        operation = FileSystemOperation(
            operation="mkdir",
            path=path,
        )
        self._filesystem_operations.append(operation)
        logger.debug("Created directory: %s", path)

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths.

        Returns:
            List of file paths that will be available in the sandbox filesystem
        """
        return [
            op.path
            for op in self._filesystem_operations
            if op.operation == "write"
        ]

    def clear_filesystem_operations(self) -> None:
        """Clear all queued filesystem operations.

        This removes all files and directories that were queued to be created
        in the sandbox filesystem.
        """
        self._filesystem_operations.clear()
        logger.debug("Cleared filesystem operations")

    def _build_command(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        memory_limit_mb: int | None = None,
    ) -> list[str]:
        """Build the Deno command with all necessary arguments.

        Args:
            code: The Python code to execute
            session_bytes: Optional session state bytes
            session_metadata: Optional session metadata
            memory_limit_mb: Optional memory limit in MB

        Returns:
            List of command arguments for subprocess execution
        """
        cmd = [
            "deno",
            "run",
        ]

        # Apply permissions
        cmd.extend(self.permissions)

        # Deno uses the V8 flag --max-old-space-size to limit memory usage in MB
        if memory_limit_mb is not None and memory_limit_mb > 0:
            cmd.append(f"--v8-flags=--max-old-space-size={memory_limit_mb}")

        # Add the path to the JavaScript wrapper script
        cmd.append(PKG_NAME)

        # Add script path and code
        cmd.extend(["-c", code])

        if self.stateful:
            cmd.extend(["-s"])

        if session_bytes:
            # Convert bytes to list of integers and then to JSON string
            bytes_array = list(session_bytes)
            cmd.extend(["-b", json.dumps(bytes_array)])

        if session_metadata:
            cmd.extend(["-m", json.dumps(session_metadata)])

        # Add filesystem operations if any are queued
        if self._filesystem_operations or self.enable_filesystem:
            if self._filesystem_operations:
                fs_ops = [op.to_dict() for op in self._filesystem_operations]
                fs_json = json.dumps(
                    fs_ops, ensure_ascii=True, separators=(",", ":")
                )
                cmd.extend(["-x", fs_json])
                logger.debug("Filesystem enabled with %d operations", len(fs_ops))
            else:
                cmd.extend(["-x", "[]"])
                logger.debug("Filesystem enabled with no initial operations")

        return cmd


class PyodideSandbox(BasePyodideSandbox):
    """Asynchronous implementation of PyodideSandbox.

    This class provides an asynchronous interface for executing Python code in a
    sandboxed Deno environment using Pyodide. It supports file attachment and
    in-memory filesystem operations.
    """

    async def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously in a sandboxed Deno subprocess.

        This method spawns a Deno subprocess that loads Pyodide (Python compiled
        to WebAssembly) and executes the provided code within that sandboxed
        environment. The execution is subject to the permissions configured in the
        sandbox's initialization and the resource constraints provided as arguments.

        Any attached files will be made available in the sandbox's in-memory
        filesystem before code execution begins.

        Args:
            code: The Python code to execute in the sandbox
            session_bytes: Optional bytes containing session state
            session_metadata: Optional metadata for session state
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB

        Returns:
            CodeExecutionResult containing execution results and metadata
        """
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
            # Wait for process with a timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")

            if stdout:
                # stdout encodes the full result from the sandbox.
                # including stdout, stderr, and the json result.
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
    """Synchronous version of PyodideSandbox.

    This class provides a synchronous interface to the PyodideSandbox functionality,
    including file attachment and in-memory filesystem operations.
    """

    def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code synchronously in a sandboxed Deno subprocess.

        This method provides the same functionality as PyodideSandbox.execute() but
        in a synchronous/blocking manner. Any attached files will be made available
        in the sandbox's in-memory filesystem before code execution begins.

        Args:
            code: The Python code to execute in the sandbox
            session_bytes: Optional bytes containing session state
            session_metadata: Optional metadata for session state
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB

        Returns:
            CodeExecutionResult containing execution results and metadata
        """
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
            # Run the subprocess with timeout
            # Ignoring S603 for subprocess.run as the cmd is built safely.
            # Untrusted input comes from `code` parameter, which should be
            # escaped properly as we are **not** using shell=True.
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
                # stdout encodes the full result from the sandbox
                # including stdout, stderr, and the json result
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
    r"""Tool for running python code in a PyodideSandbox.

    This tool extends the base PyodideSandbox functionality with support for
    attaching files and creating an in-memory filesystem. Files attached to
    the tool will be available within the Python execution environment.

    If you use a stateful sandbox (PyodideSandboxTool(stateful=True)),
    the state between code executions (to variables, imports,
    and definitions, etc.), will be persisted using LangGraph checkpointer.

    !!! important
        When you use a stateful sandbox, this tool can only be used
        inside a LangGraph graph with a checkpointer, and
        has to be used with the prebuilt `create_react_agent` or `ToolNode`.

    Example: stateless sandbox usage

        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_sandbox import PyodideSandboxTool

        tool = PyodideSandboxTool(enable_filesystem=True, allow_net=True)

        # Attach data files
        tool.attach_file("data.csv", "name,age\\nJohn,25\\nMary,30")

        agent = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[tool],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "analyze the data.csv file"}]},
        )
        ```

    Example: stateful sandbox usage

        ```python
        from langgraph.prebuilt import create_react_agent
        from langgraph.prebuilt.chat_agent_executor import AgentState
        from langgraph.checkpoint.memory import InMemorySaver
        from langchain_sandbox import PyodideSandboxTool, PyodideSandbox

        class State(AgentState):
            session_bytes: bytes
            session_metadata: dict

        tool = PyodideSandboxTool(stateful=True, enable_filesystem=True, allow_net=True)
        agent = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[tool],
            checkpointer=InMemorySaver(),
            state_schema=State
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "what's 5 + 7? save result as 'a'"}
                ],
                "session_bytes": None,
                "session_metadata": None
            },
            config={"configurable": {"thread_id": "123"}},
        )
        second_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's the sine of 'a'?"}]},
            config={"configurable": {"thread_id": "123"}},
        )
        ```
    """

    name: str = "python_code_sandbox"
    description: str = (
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
    enable_filesystem: bool = False

    _sandbox: PyodideSandbox
    _sync_sandbox: SyncPyodideSandbox
    _structured_tool: StructuredTool | None

    def __init__(
        self,
        *,
        stateful: bool = False,
        timeout_seconds: float | None = 60,
        allow_net: list[str] | bool = False,
        enable_filesystem: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the tool.

        Args:
            stateful: Whether to use a stateful sandbox. If True, `sandbox.execute`
                will include session metadata and the session bytes containing the
                session state (variables, imports, etc.) in the execution result.
                This allows saving and reusing the session state between executions.
            timeout_seconds: Timeout for code execution in seconds.
            enable_filesystem: Enable in-memory filesystem support for attaching files.
            allow_net: configure network access. If setting to True, any network access
                is allowed, including potentially internal network addresses that you
                may not want to expose to a malicious actor.
                Depending on your use case, you can restrict the network access to
                only the URLs you need (e.g., required to set up micropip / pyodide).
                Please refer to pyodide documentation for more details.
            **kwargs: Other attributes will be passed to the PyodideSandbox
        """
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
                # these fields will be ignored by the LLM
                # and automatically injected by LangGraph's ToolNode
                state: Annotated[dict[str, Any] | BaseModel, InjectedState]
                tool_call_id: Annotated[str, InjectedToolCallId]

        else:

            class PyodideSandboxToolInput(BaseModel):
                """Python code to execute in the sandbox."""

                code: str = Field(description="Code to execute.")

        super().__init__(
            stateful=stateful,
            timeout_seconds=timeout_seconds,
            allow_net=allow_net,
            enable_filesystem=enable_filesystem,
            **kwargs,
        )

        self.args_schema: type[BaseModel] = PyodideSandboxToolInput
        self._structured_tool = None  # Initialize as None
        self._sandbox = PyodideSandbox(
            stateful=self.stateful,
            allow_env=self.allow_env,
            allow_read=self.allow_read,
            allow_write=self.allow_write,
            allow_net=self.allow_net,
            allow_run=self.allow_run,
            allow_ffi=self.allow_ffi,
            node_modules_dir=self.node_modules_dir,
            enable_filesystem=self.enable_filesystem,
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
            enable_filesystem=self.enable_filesystem,
            skip_deno_check=True,  # Skip deno check since async sandbox already checked
        )

    def _build_description(self) -> str:
        """Build the complete description string with attached files.

        Returns:
            Tool description including information about attached files
        """
        base = (
            "A secure Python code sandbox with filesystem support. "
            "Use this to execute python commands.\n"
            "- Input should be a valid python command.\n"
            "- To return output, you should print it out with `print(...)`.\n"
            "- Don't use f-strings when printing outputs.\n"
            "- If you need to make web requests, use `httpx.AsyncClient`.\n"
            "- Files can be read/written using standard Python file operations."
        )

        files = self._sandbox.get_attached_files()
        if files:
            base += "\n\nATTACHED FILES AVAILABLE:\n"
            base += "\n".join(f"  • {p}" for p in files)
            base += (
                "\nThese files are already loaded and ready to use with pandas, "
                "open(), etc."
            )
        return base

    def as_structured_tool(self) -> StructuredTool:
        """Return a StructuredTool version of this tool.

        This method provides access to a StructuredTool interface while maintaining
        the BaseTool as the primary interface. The StructuredTool's description
        is kept in sync with attached files.

        Returns:
            StructuredTool instance with dynamic description updates
        """
        if self._structured_tool is None:
            self._structured_tool = StructuredTool.from_function(
                name=self.name,
                description=self._build_description(),
                func=(
                    self._run_sync
                    if not self.stateful
                    else self._run_stateful_sync
                ),
                args_schema=self.args_schema,
            )
        return self._structured_tool

    @property
    def tool(self) -> StructuredTool:
        """Legacy property for backwards compatibility.

        DEPRECATED: Use as_structured_tool() instead.

        Returns:
            StructuredTool instance
        """
        return self.as_structured_tool()

    def _run_sync(self, code: str) -> str:
        """Synchronous execution function for non-stateful mode."""
        result = self._sync_sandbox.execute(
            code, timeout_seconds=self.timeout_seconds
        )

        if result.status == "error":
            error_msg = (
                result.stderr
                if result.stderr
                else "Execution failed with unknown error"
            )
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
    ) -> Any:  # noqa: ANN401
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

    def attach_file(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Attach a text file to the sandbox environment.

        This file will be available in the sandbox's in-memory filesystem
        when code is executed. The tool's description will be automatically
        updated to reflect the attached files.

        Args:
            path: File path within the sandbox filesystem
            content: Text content of the file
            encoding: Text encoding (default: utf-8)
        """
        self._sandbox.attach_file(path, content, encoding=encoding)
        self._sync_sandbox.attach_file(path, content, encoding=encoding)
        # Update both BaseTool and StructuredTool descriptions
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox environment.

        This file will be available in the sandbox's in-memory filesystem
        when code is executed. The tool's description will be automatically
        updated to reflect the attached files.

        Args:
            path: File path within the sandbox filesystem
            content: Binary content of the file
        """
        self._sandbox.attach_binary_file(path, content)
        self._sync_sandbox.attach_binary_file(path, content)
        # Update both BaseTool and StructuredTool descriptions
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox environment.

        This directory will be available in the sandbox's in-memory filesystem
        when code is executed.

        Args:
            path: Directory path within the sandbox filesystem
        """
        self._sandbox.create_directory(path)
        self._sync_sandbox.create_directory(path)
        # Update both BaseTool and StructuredTool descriptions
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths.

        Returns:
            List of file paths that will be available in the sandbox filesystem
        """
        return self._sandbox.get_attached_files()

    def clear_filesystem_operations(self) -> None:
        """Clear all attached files and directories.

        This removes all files and directories that were queued to be created
        in the sandbox filesystem and updates the tool description.
        """
        self._sandbox.clear_filesystem_operations()
        self._sync_sandbox.clear_filesystem_operations()
        # Update both BaseTool and StructuredTool descriptions
        new_description = self._build_description()
        self.description = new_description
        if self._structured_tool:
            self._structured_tool.description = new_description

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
        else:
            result = self._sync_sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

        if result.stderr:
            tool_result = f"Error during execution: {result.stderr}"
        else:
            tool_result = result.stdout

        if self.stateful:
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

        return tool_result

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
        else:
            result = await self._sandbox.execute(
                code, timeout_seconds=self.timeout_seconds
            )

        if result.stderr:
            tool_result = f"Error during execution: {result.stderr}"
        else:
            tool_result = result.stdout

        if self.stateful:
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

        return tool_result


class PyodideSandboxStructuredTool:
    r"""Pure StructuredTool wrapper for PyodideSandbox with dynamic description updates.

    This class provides a standalone StructuredTool interface for users who prefer
    to work exclusively with StructuredTool rather than BaseTool. It maintains all
    the filesystem functionality and dynamic description updates.

    Example usage:
        ```python
        from langchain_sandbox import PyodideSandboxStructuredTool
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI

        # Create tool
        sandbox_tool = PyodideSandboxStructuredTool(
            enable_filesystem=True,
            allow_net=True,
        )

        # Attach files
        sandbox_tool.attach_file("data.csv", "name,age\\nJohn,25")

        # Use in agent
        agent = create_react_agent(llm, [sandbox_tool.tool])
        ```
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the StructuredTool wrapper.

        Args:
            **kwargs: All arguments are passed to PyodideSandboxTool
        """
        self._base_tool = PyodideSandboxTool(**kwargs)
        # Force creation of the StructuredTool
        self._tool = self._base_tool.as_structured_tool()

    @property
    def tool(self) -> StructuredTool:
        """Access to the underlying StructuredTool.

        Returns:
            StructuredTool instance with current description
        """
        return self._tool

    def attach_file(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        """Attach a text file to the sandbox environment.

        Args:
            path: File path within the sandbox filesystem
            content: Text content of the file
            encoding: Text encoding (default: utf-8)
        """
        self._base_tool.attach_file(path, content, encoding=encoding)

    def attach_binary_file(
        self,
        path: str,
        content: bytes,
    ) -> None:
        """Attach a binary file to the sandbox environment.

        Args:
            path: File path within the sandbox filesystem
            content: Binary content of the file
        """
        self._base_tool.attach_binary_file(path, content)

    def create_directory(self, path: str) -> None:
        """Create a directory in the sandbox environment.

        Args:
            path: Directory path within the sandbox filesystem
        """
        self._base_tool.create_directory(path)

    def get_attached_files(self) -> list[str]:
        """Get list of attached file paths.

        Returns:
            List of file paths that will be available in the sandbox filesystem
        """
        return self._base_tool.get_attached_files()

    def clear_filesystem_operations(self) -> None:
        """Clear all attached files and directories."""
        self._base_tool.clear_filesystem_operations()

    def invoke(self, input_data: dict[str, Any]) -> str:
        """Direct invoke method for easier usage.

        Args:
            input_data: Input data containing 'code' key

        Returns:
            Execution result as string
        """
        return self.tool.invoke(input_data)

    async def ainvoke(self, input_data: dict[str, Any]) -> str:
        """Async direct invoke method for easier usage.

        Args:
            input_data: Input data containing 'code' key

        Returns:
            Execution result as string
        """
        return await self.tool.ainvoke(input_data)
