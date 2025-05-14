"""LangChain tool for running python code in a PyodideSandbox."""

from typing import Annotated, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_sandbox import PyodideSandbox


def _get_default_pyodide_sandbox() -> PyodideSandbox:
    """Get default sandbox for the tool."""
    return PyodideSandbox(
        # Allow Pyodide to install python packages that
        # might be required.
        allow_net=True,
    )


class PythonInputs(BaseModel):
    """Python code to execute in the sandbox."""

    state: Annotated[dict[str, Any] | BaseModel, InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]
    code: str = Field(description="Code to execute.")


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

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the tool with correct args schema."""
        super().__init__(**kwargs)
        self.args_schema: type[BaseModel] = PythonInputs

    def _run(
        self,
        code: str,
        state: dict[str, Any] | BaseModel,
        tool_call_id: str,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> None:
        """Use the tool."""
        error_msg = (
            "Sync invocation of PyodideSandboxTool is not supported - "
            "please invoke the tool asynchronously using `await tool.ainvoke()`"
        )
        raise NotImplementedError(error_msg)

    async def _arun(
        self,
        code: str,
        state: dict[str, Any] | BaseModel,
        tool_call_id: str,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str | Command:
        """Use the tool asynchronously."""
        if isinstance(state, dict):
            session_bytes = state.get("session_bytes")
            session_metadata = state.get("session_metadata")
        else:
            session_bytes = getattr(state, "session_bytes", None)
            session_metadata = getattr(state, "session_metadata", None)

        result = await self.sandbox.execute(
            code, session_bytes=session_bytes, session_metadata=session_metadata
        )

        if result.stderr:
            return f"Error during execution: {result.stderr}"

        if self.sandbox.stateful:
            return Command(
                update={
                    "session_bytes": result.session_bytes,
                    "session_metadata": result.session_metadata,
                    "messages": [
                        ToolMessage(
                            content=result.stdout,
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        return result.stdout
