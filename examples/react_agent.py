# pip install langgraph "langchain[anthropic]"
import asyncio

from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent

# Define the sandbox tool
sandbox_tool = PyodideSandboxTool(
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True,
)

# Create an agent with the sandbox tool
agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest", [sandbox_tool]
)

query = """A batter hits a baseball at 45.847 m/s at an angle of 23.474° above the horizontal. The outfielder, who starts facing the batter, picks up the baseball as it lands, then throws it back towards the batter at 24.12 m/s at an angle of 39.12 degrees. How far is the baseball from where the batter originally hit it? Assume zero air resistance."""


async def run_agent(query: str):
    # Stream agent outputs
    async for chunk in agent.astream({"messages": query}):
        print(chunk)
        print("\n")


if __name__ == "__main__":
    # Run the agent
    asyncio.run(run_agent(query))
