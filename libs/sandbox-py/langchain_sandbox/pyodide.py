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


# Published package name
PKG_NAME = "jsr:@langchain/pyodide-sandbox@0.0.4"


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
        default_values: Optional default items that should always be included.

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

    The sandbox supports fine-grained permission control through its initializer:
    - Restrict network access to specific domains
    - Limit file system access to specific directories
    - Control environment variable access
    - Prevent subprocess execution and FFI
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
        """
        self.stateful = stateful
        # Configure permissions
        self.permissions = []

        self.file_operations = []

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

    def attach_file(self, path: str, content: str | bytes) -> None:
        """Attach a file to the sandbox filesystem.

        The file will be created in the sandbox's memfs filesystem and will be
        available to the Python code when executed. Binary content is automatically
        detected based on content type.

        Args:
            path: Path in the sandbox filesystem where the file should be created.
                 If not starting with '/sandbox/', it will be prefixed automatically.
            content: The content of the file, either as a string or bytes.
                    If bytes are provided, it will be treated as binary data.
        """
        binary = isinstance(content, bytes)

        if not path.startswith("/sandbox/"):
            path = f"/sandbox/{path}"

        encoding = "binary" if binary else "utf-8"

        if binary:
            content = base64.b64encode(content).decode("ascii")

        self.file_operations.append({
            "operation": "write",
            "path": path,
            "content": content,
            "encoding": encoding
        })

    def attach_files(
        self, files: dict[str, str | bytes | dict[str, str | bool]]
    ) -> None:
        """Attach multiple files to the sandbox filesystem.

        Args:
            files: Dictionary mapping paths to file contents.
                  Each value can be:
                  - a string (treated as text content)
                  - bytes (treated as binary content)
                  - a dictionary with 'content' key (and optional 'binary' key
                    if explicit format control is needed)
        """
        for path, content_info in files.items():
            if isinstance(content_info, (str, bytes)):
                self.attach_file(path, content_info)
            elif isinstance(content_info, dict):
                content = content_info.get("content", "")

                if "binary" in content_info:
                    binary_flag = content_info.get("binary", False)
                    if isinstance(content, str) and binary_flag:
                        # Convert string to bytes when binary flag is True
                        content = content.encode("utf-8")

                self.attach_file(path, content)

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

        # Add filesystem operations if there are any
        if self.file_operations:
            cmd.extend(["--fs-operations", json.dumps(self.file_operations)])

        return cmd


class PyodideSandbox(BasePyodideSandbox):
    """Asynchronous implementation of PyodideSandbox.

    This class provides an asynchronous interface for executing Python code in a
    sandboxed Deno environment using Pyodide.
    """

    async def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
        clear_files: bool = False,
    ) -> CodeExecutionResult:
        """Execute Python code asynchronously in a sandboxed Deno subprocess.

        This method spawns a Deno subprocess that loads Pyodide (Python compiled
        to WebAssembly) and executes the provided code within that sandboxed
        environment. The execution is subject to the permissions configured in the
        sandbox's initialization and the resource constraints provided as arguments.

        Args:
            code: The Python code to execute in the sandbox
            session_bytes: Optional bytes containing session state
            session_metadata: Optional metadata for session state
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            clear_files: If True, clear the attached files after execution

        Returns:
            CodeExecutionResult containing execution results and metadata
        """
        start_time = time.time()
        stdout = ""
        stderr = ""
        result = None
        status: Literal["success", "error"] = "success"

        try:
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
                    # Convert the Uint8Array to Python bytes
                    session_bytes_array = full_result.get("sessionBytes", None)
                    session_bytes = (
                        bytes(session_bytes_array) if session_bytes_array else None
                    )
                else:
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                    status = "error"
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                status = "error"
                stderr = f"Execution timed out after {timeout_seconds} seconds"
            except asyncio.CancelledError:
                # Optionally: log cancellation if needed
                pass
        finally:
            if clear_files:
                self.file_operations = []

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
        )


class SyncPyodideSandbox(BasePyodideSandbox):
    """Synchronous version of PyodideSandbox.

    This class provides a synchronous interface to the PyodideSandbox functionality.
    """

    def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
        clear_files: bool = False,
    ) -> CodeExecutionResult:
        """Execute Python code synchronously in a sandboxed Deno subprocess.

        This method provides the same functionality as PyodideSandbox.execute() but
        in a synchronous/blocking manner.

        Args:
            code: The Python code to execute in the sandbox
            session_bytes: Optional bytes containing session state
            session_metadata: Optional metadata for session state
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            clear_files: If True, clear the attached files after execution

        Returns:
            CodeExecutionResult containing execution results and metadata
        """
        start_time = time.time()
        stdout = ""
        result = None
        stderr: str
        status: Literal["success", "error"]

        try:
            cmd = self._build_command(
                code,
                session_bytes=session_bytes,
                session_metadata=session_metadata,
                memory_limit_mb=memory_limit_mb,
            )

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
                # Convert the Uint8Array to Python bytes
                session_bytes_array = full_result.get("sessionBytes", None)
                session_bytes = (
                    bytes(session_bytes_array) if session_bytes_array else None
                )
            else:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                status = "error"

        except subprocess.TimeoutExpired:
            status = "error"
            stderr = f"Execution timed out after {timeout_seconds} seconds"
        finally:
            if clear_files:
                self.file_operations = []

        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            session_metadata=session_metadata,
            session_bytes=session_bytes,
        )


class PyodideSandboxTool(BaseTool):
    """Tool for running python code in a PyodideSandbox."""

    name: str = "python_code_sandbox"
    description: str = (
        "A secure Python code sandbox. Use this to execute python commands.\n"
        "- Input should be a valid python command.\n"
        "- To return output, you should print it out with `print(...)`.\n"
        "- Don't use f-strings when printing outputs.\n"
        "- If you need to make web requests, use `httpx.AsyncClient`."
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

    _sandbox: PyodideSandbox
    _sync_sandbox: SyncPyodideSandbox

    def __init__(
        self,
        *,
        stateful: bool = False,
        timeout_seconds: float | None = 60,
        allow_net: list[str] | bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the tool.

        Args:
            stateful: Whether to use a stateful sandbox. If True, `sandbox.execute`
                will include session metadata and the session bytes containing the
                session state (variables, imports, etc.) in the execution result.
                This allows saving and reusing the session state between executions.
            timeout_seconds: Timeout for code execution in seconds.
            allow_net: configure network access. If setting to True, any network access
                is allowed, including potentially internal network addresses that you
                may not want to expose to a malicious actor.
                Depending on your use case, you can restrict the network access to
                only the URLs you need (e.g., required to set up micropip / pyodide).
                Please refer to pyodide documentation for more details.
            **kwargs: Other attributes will be passed to the PyodideSandbox
        """
        super().__init__(
            stateful=stateful,
            timeout_seconds=timeout_seconds,
            allow_net=allow_net,
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
            skip_deno_check=True,  # Skip deno check since async sandbox already checked
        )

    def attach_file(self, path: str, content: str | bytes) -> None:
        """Attach a file to the sandbox filesystem.

        This method delegates to both the async and sync sandboxes to ensure consistency
        Binary content is automatically detected based on the content type.

        Args:
            path: Path in the sandbox filesystem where the file should be created.
                 If not starting with '/sandbox/', it will be prefixed automatically.
            content: The content of the file, either as a string or bytes.
                    If bytes are provided, it will be treated as binary data.
        """
        self._sandbox.attach_file(path, content)
        self._sync_sandbox.attach_file(path, content)

    def attach_files(
            self, files: dict[str, str | bytes | dict[str, str | bool]]
        ) -> None:
        """Attach multiple files to the sandbox filesystem.

        This method delegates to both the async and sync sandboxes to ensure consistency

        Args:
            files: Dictionary mapping paths to file contents.
                  Each value can be:
                  - a string (treated as text content)
                  - bytes (treated as binary content)
                  - a dictionary with 'content' key (and optional 'binary' key
                    if explicit format control is needed)
        """
        self._sandbox.attach_files(files)
        self._sync_sandbox.attach_files(files)

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
