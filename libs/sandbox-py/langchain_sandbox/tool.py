
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
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


class PythonInputs(BaseModel):
    """Python code to execute in the sandbox."""

    code: str = Field(description="Code to execute.")


class PyodideSandboxTool(BaseTool):
    """Tool for running python code in a PyodideSandbox."""

    name: str = "python_code_sandbox"
    description: str = (
        "A secure Python code sandbox. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`. Don't use f-strings when printing outputs."
    )
    sandbox: PyodideSandbox = Field(default_factory=_get_default_pyodide_sandbox)
    args_schema: type[BaseModel] = PythonInputs

    def _run(
        self,
        code: str,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool."""
        error_msg = (
            "Sync invocation of PyodideSandboxTool is not supported - "
            "please invoke the tool asynchronously using `await tool.ainvoke()`"
        )
        raise NotImplementedError(error_msg)

    async def _arun(
        self,
        code: str,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        session_id = config.get("configurable", {}).get("thread_id")
        result = await self.sandbox.execute(code, session_id=session_id)
        if result.stderr:
            return f"Error during execution: {result.stderr}"
        return result.stdout
