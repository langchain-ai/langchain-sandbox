# pip install langgraph-codeact "langchain[anthropic]"
import asyncio

from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent


# Define the sandbox tool with filesystem support
sandbox_tool = PyodideSandboxTool(
    enable_filesystem=True,
    allow_net=True,
)

sales_data = """date,product,category,quantity,price,region
2024-01-15,Laptop,Electronics,2,1299.99,North
2024-01-16,Chair,Furniture,1,249.50,South
2024-01-16,T-shirt,Clothing,5,29.99,East
2024-01-17,Laptop,Electronics,1,1299.99,West
2024-01-18,Phone,Electronics,3,799.99,North
2024-01-19,Desk,Furniture,2,399.99,South
2024-01-20,Jeans,Clothing,4,79.99,East
2024-01-21,Tablet,Electronics,2,499.99,West
2024-01-22,Sofa,Furniture,1,899.99,North
2024-01-23,Shoes,Clothing,3,129.99,South"""

sandbox_tool.attach_file("sales.csv", sales_data)

# Create an agent with the sandbox tool
agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest", [sandbox_tool]
)

query = """Please analyze the sales data and tell me:
1. What is the total revenue by category?
2. Which region has the highest sales?
3. What are the top 3 best-selling products by revenue?

Use pandas to read the CSV file and perform the analysis."""

async def run_agent(query: str):
    # Stream agent outputs
    async for chunk in agent.astream({"messages": query}):
        print(chunk)
        print("\n")

if __name__ == "__main__":
    # Run the agent
    asyncio.run(run_agent(query))
