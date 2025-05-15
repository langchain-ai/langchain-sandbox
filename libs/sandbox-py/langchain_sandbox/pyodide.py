"""Python wrapper that calls pyodide & deno for code execution."""

import asyncio
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
PKG_NAME = "jsr:@eyurtsev/test-sandbox@0.0.7"


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


class PyodideSandbox:
    """Run Python code in a sandboxed environment using Deno and Pyodide.

    This executor leverages Deno's security model to create a secure runtime for
    executing untrusted Python code. It works by spawning a Deno subprocess that loads
    Pyodide (Python compiled to WebAssembly) and executes the provided code in an
    isolated environment.

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
        *,
        stateful: bool = False,
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

        """
        self.stateful = stateful
        # Configure permissions
        self.permissions = []

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

    async def execute(
        self,
        code: str,
        *,
        session_bytes: bytes | None = None,
        session_metadata: dict | None = None,
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
            session_bytes: Optional bytes to be used as the initial session state.
                        This is used to resume code execution from the same session.
                        You can use this instead of `session_id`.
                        Note: when using this, you need to provide `session_metadata`.
            session_metadata: Optional metadata to be used as the initial session state.
                        This is used to resume code execution from the same session.
                        Note: when using this, you need to provide `session_bytes`.
            timeout_seconds: Maximum execution time in seconds before the process
                        is terminated. If None, execution may run indefinitely
                        (not recommended for untrusted code).
            memory_limit_mb: Maximum memory usage in MB. Pass this to Deno to
                        enforce memory limits in the WebAssembly VM.

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
    """Tool for running python code in a PyodideSandbox.

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

        tool = PyodideSandboxTool(allow_net=True)
        agent = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[tool],
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
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

        tool = PyodideSandboxTool(stateful=True, allow_net=True)
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
    node_modules_dir: str = "auto"

    _sandbox: PyodideSandbox

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the tool."""
        super().__init__(**kwargs)

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

    def _run(
        self,
        code: str,
        state: dict[str, Any] | BaseModel | None = None,
        tool_call_id: str | None = None,
        config: RunnableConfig | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:  # noqa: ANN401
        """Use the tool."""
        error_msg = (
            "Sync invocation of PyodideSandboxTool is not supported - "
            "please invoke the tool asynchronously using `await tool.ainvoke()`"
        )
        raise NotImplementedError(error_msg)

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
                code, session_bytes=session_bytes, session_metadata=session_metadata
            )
        else:
            result = await self._sandbox.execute(code)

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
