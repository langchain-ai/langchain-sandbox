"""LangChain tool for running python code in a PyodideSandbox."""

from typing import Annotated, Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_sandbox import PyodideSandbox


def _get_default_pyodide_sandbox() -> PyodideSandbox:
    """Get default sandbox for the tool."""
    return PyodideSandbox(
        "./sessions",  # Directory to store session files
        # Allow Pyodide to install python packages that
        # might be required.
        allow_net=True,
    )


def _get_last_tool_artifact(
    state: dict[str, Any],
) -> dict[str, bytes | dict | None] | None:
    if "messages" not in state:
        error_msg = "Expected state to have 'messages' key."
        raise ValueError(error_msg)

    last_tool_message = next(
        (
            msg
            for msg in reversed(state["messages"])
            if isinstance(msg, ToolMessage) and msg.name == "python_code_sandbox"
        ),
        None,
    )
    if last_tool_message is None:
        return None

    if (
        "session_bytes" not in last_tool_message.artifact
        or "session_metadata" not in last_tool_message.artifact
    ):
        error_msg = (
            "Expected tool artifact to have 'session_bytes' and "
            "'session_metadata' keys."
        )
        raise ValueError(error_msg)

    return last_tool_message.artifact


class PyodideSandboxTool(BaseTool):
    """Tool for running python code in a PyodideSandbox.

    If you want to persist state between code executions (to persist variables, imports,
    and definitions, etc.), you need to invoke the tool with `thread_id` in the config:

    ```python
    from langchain_sandbox import PyodideSandboxTool

    tool = PyodideSandboxTool()
    result = await tool.ainvoke(
        "print('Hello, world!')",
        config={"configurable": {"thread_id": "123"}},
    )
    ```

    If you are using this tool inside an agent, like LangGraph `create_react_agent`, you
    can invoke the agent with a config, and it will automatically be passed to the tool:

    ```python
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain_sandbox import PyodideSandboxTool

    tool = PyodideSandboxTool()
    agent = create_react_agent(
        "anthropic:claude-3-7-sonnet-latest",
        tools=[tool],
        checkpointer=InMemorySaver()
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
        config={"configurable": {"thread_id": "123"}},
    )
    second_result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's the sine of that?"}]},
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
    sandbox: PyodideSandbox = Field(default_factory=_get_default_pyodide_sandbox)
    use_checkpointer: bool = False
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the tool with correct args schema."""
        super().__init__(**kwargs)
        if self.use_checkpointer:
            try:
                from langgraph.prebuilt import InjectedState
            except ImportError as e:
                error_msg = (
                    "The 'langgraph' package is required when use_checkpointer=True. "
                    "Please install it with 'pip install langgraph'."
                )
                raise ImportError(error_msg) from e

            class PythonInputs(BaseModel):
                """Python code to execute in the sandbox."""

                state: Annotated[dict, InjectedState]
                code: str = Field(description="Code to execute.")

        else:

            class PythonInputs(BaseModel):
                """Python code to execute in the sandbox."""

                code: str = Field(description="Code to execute.")

        self.args_schema: type[BaseModel] = PythonInputs

    def _run(
        self,
        code: str,
        state: dict,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, bytes | dict | None]]:
        """Use the tool."""
        error_msg = (
            "Sync invocation of PyodideSandboxTool is not supported - "
            "please invoke the tool asynchronously using `await tool.ainvoke()`"
        )
        raise NotImplementedError(error_msg)

    async def _arun(
        self,
        code: str,
        state: dict,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, bytes | dict | None]]:
        """Use the tool asynchronously."""
        if self.use_checkpointer:
            # If we're using a checkpointer, load the most recent available
            # tool artifact for this tool from the checkpointed messages.
            # The tool artifact will contain the session bytes and session metadata,
            # which allows us to resume code execution from the same sandbox session.
            last_artifact = _get_last_tool_artifact(state)
            sandbox_kwargs = {}
            if last_artifact is not None:
                sandbox_kwargs = {
                    "session_bytes": last_artifact["session_bytes"],
                    "session_metadata": last_artifact["session_metadata"],
                }

            result = await self.sandbox.execute(code, **sandbox_kwargs)
        else:
            # If we're not using a checkpointer, we rely
            # on passing the thread_id as a session_id
            # and storing session bytes and metadata on disk.
            session_id = config.get("configurable", {}).get("thread_id")
            result = await self.sandbox.execute(code, session_id=session_id)

        artifact = {
            "session_bytes": result.session_bytes,
            "session_metadata": result.session_metadata,
        }
        if result.stderr:
            return f"Error during execution: {result.stderr}", artifact
        return result.stdout, artifact
