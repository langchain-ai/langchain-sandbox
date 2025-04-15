import asyncio
import dataclasses
import logging
import re
import subprocess
import json
import time
from typing import Any, Literal

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


PKG_NAME = "jsr:@eyurtsev/test-sandbox@0.0.7"


def build_permission_flag(
    flag: str,
    *,
    value: bool | list[str],
    default_values: list[str] | None = None,
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
        finalized_values = value if default_values is None else value + default_values
        return f"{flag}={','.join(finalized_values)}"
    return None


class PyodideSandbox:
    """Run Python code in a sandboxed environment using Deno and Pyodide.

    This executor leverages Deno's security model to create a secure runtime for executing
    untrusted Python code. It works by spawning a Deno subprocess that loads Pyodide (Python
    compiled to WebAssembly) and executes the provided code in an isolated environment.

    Security features:
    - Configurable permissions for file system, network, and environment access
    - Support for execution timeouts to prevent infinite loops
    - Memory usage monitoring
    - Process isolation via Deno's security sandbox

    The executor supports fine-grained permission control through its initializer:
    - Restrict network access to specific domains
    - Limit file system access to specific directories
    - Control environment variable access
    - Prevent subprocess execution and FFI

    Performance characteristics:
    - Each execution creates a new Deno process
    - Session support for maintaining state between executions
    - Streaming stdout/stderr capture
    """

    def __init__(
        self,
        sessions_dir: str,
        *,
        allow_env: list[str] | bool = False,
        allow_read: list[str] | bool = False,
        allow_write: list[str] | bool = False,
        allow_net: list[str] | bool = False,
        allow_run: list[str] | bool = False,
        allow_ffi: list[str] | bool = False,
        node_modules_dir: str = "auto",
    ) -> None:
        """Initialize the executor with specific Deno permissions.

        This method configures the security permissions for the Deno subprocess that
        will execute Python code via Pyodide. By default, all permissions are
        disabled (False) for maximum security. Permissions can be enabled selectively
        based on the needs of the code being executed.

        Args:
            sessions_dir: Directory for storing session data. This directory must
                be writable by the Deno subprocess. It is used to persist session
                state between executions.
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

            allow_write: File system write access configuration:
                - False: No file system write access (default, most secure)
                - True: Unrestricted write access to the file system
                - List[str]: Write access restricted to specific paths, e.g.
                  ["/tmp/sandbox/output"]

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

        """
        # Store configuration
        self.sessions_dir = sessions_dir

        # Configure permissions
        self.permissions = []

        # Check if Deno is installed
        try:
            subprocess.run(["deno", "--version"], check=True)
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
            ("--allow-read", allow_read, [sessions_dir]),
            ("--allow-write", allow_write, [sessions_dir]),
            ("--allow-net", allow_net, None),
            ("--allow-run", allow_run, None),
            ("--allow-ffi", allow_ffi, None),
        ]

        self.permissions = []
        for flag, value, defaults in perm_defs:
            perm = build_permission_flag(flag, value=value, default_values=defaults)
            # For file system permissions, if no permission is specified,
            # force at least sessions_dir.
            if perm is None and flag in ("--allow-read", "--allow-write"):
                perm = f"{flag}={sessions_dir}"
            if perm:
                self.permissions.append(perm)

        self.permissions.append(f"--node-modules-dir={node_modules_dir}")

        # Regular expression for validating session IDs
        self.session_id_pattern = re.compile(r"^[a-zA-Z0-9\-_]+$")

    def _validate_session_id(self, session_id: str | None) -> str | None:
        """Validate the session ID against the allowed pattern.

        Args:
            session_id: The session ID to validate

        Returns:
            The session ID if valid, None otherwise

        Raises:
            ValueError: If the session ID contains invalid characters

        """
        if session_id is None:
            return None

        if not self.session_id_pattern.match(session_id):
            msg = (
                f"Invalid session ID: {session_id}. "
                "Session IDs must contain only alphanumeric characters, "
                "hyphens, and underscores."
            )
            raise ValueError(
                msg,
            )

        return session_id

    async def execute(
        self,
        code: str,
        *,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> CodeExecutionResult:
        """Execute Python code in a sandboxed Deno subprocess with resource constraints.

        This method spawns a Deno subprocess that loads Pyodide (Python compiled
        to WebAssembly) and executes the provided code within that sandboxed
        environment. The execution is subject to the permissions configured in the
        executor's initialization and the resource constraints provided as arguments.

        The code execution flow:
        1. A Deno subprocess is created with the configured permissions
        2. The JavaScript wrapper loads Pyodide in the Deno context
        3. The Python code is passed to Pyodide for execution
        4. Results and output are captured and returned

        Security features:
        - Process isolation through Deno's security sandbox
        - Configurable timeout to prevent infinite loops or long-running code
        - Memory limit to prevent excessive resource consumption
        - Permission restrictions based on executor configuration
        - Controlled access to file system, network, and environment

        Args:
            code: The Python code to execute in the sandbox
            session_id: Optional session identifier for maintaining state between executions.
                        Can be used to persist variables, imports, and definitions across
                        multiple execute() calls. If None, a new session is created.
            timeout_seconds: Maximum execution time in seconds before the process is terminated.
                            If None, execution may run indefinitely (not recommended for
                            untrusted code).
            memory_limit_mb: Maximum memory usage in MB. Pass this to Deno to enforce memory
                            limits in the WebAssembly VM.

        Returns:
            CodeExecutionResult containing:
            - result: The value returned by the executed code (if any)
            - stdout: Standard output captured during execution
            - stderr: Standard error captured during execution
            - status: Execution status (success, error, timeout, etc.)
            - session_id: The session identifier (if provided)
            - execution_time: Time taken for execution in seconds
            - execution_info: Additional metadata about the execution

        Raises:
            No exceptions are raised directly; execution errors are captured in the
            CodeExecutionResult object with the appropriate status.

        """
        start_time = time.time()
        stdout = ""
        stderr = ""
        result = None
        status: Literal["success", "error"] = "success"

        # Create base command with the configured permissions
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
        # Developer version
        cmd.append(PKG_NAME)

        # Add script path and code
        cmd.extend(["-c", code])

        # Add session ID if provided
        if session_id:
            cmd.extend(["-s", session_id])

        # Ensure the sessions directory exists
        cmd.extend(["-d", self.sessions_dir])

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
        end_time = time.time()

        return CodeExecutionResult(
            status=status,
            execution_time=end_time - start_time,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
        )
