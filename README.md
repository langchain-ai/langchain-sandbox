> Under development. Feel free to test out and provide feedback.

# 🛡️ LangChain Sandbox

> A secure environment for running Python code using Pyodide (WebAssembly) and Deno

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Deno](https://img.shields.io/badge/Deno-Required-green.svg)](https://deno.land/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

LangChain Sandbox provides a secure environment for executing untrusted Python code. It leverages Pyodide (Python compiled to WebAssembly) to run Python code in a sandboxed environment.

## ✨ Key Features

- **🔒 Security** - Isolated execution environment with configurable permissions
- **💻 Local Execution** - No remote execution or Docker containers needed
- **🔄 Session Support** - Maintain state across multiple code executions

## Limitations

- **Latency**: There is a few seconds of latency when starting the sandbox per run
- **File access**: Currently not supported. You will not be able to access the files written by the sandbox.
- **Network requests**: If you need to make network requests please use `httpx.AsyncClient` instead of `requests`.

## 🚀 Quick Install

1. Install Deno (required): https://docs.deno.com/runtime/getting_started/installation/

2. Install `langchain-sandbox`:
    
    ```bash
    pip install langchain-sandbox
    ```

## 💡 Example Usage

```python
from langchain_sandbox import PyodideSandbox

# Create a sandbox instance
sandbox = PyodideSandbox(
   "./sessions", # Directory to store session files
   # Allow Pyodide to install python packages that
   # might be required.
   allow_net=True,
)
code = """\
import numpy as np
x = np.array([1, 2, 3])
print(x)
"""

# Execute Python code
print(await sandbox.execute(code, session_id="123"))

# CodeExecutionResult(
#   result=None, 
#   stdout='[1 2 3]', 
#   stderr=None, 
#   status='success', 
#   execution_time=2.8578367233276367
# )

# Can still access a previous result!
print(await sandbox.execute("float(x[0])", session_id="123"))

#  CodeExecutionResult(
#     result=1, 
#     stdout=None, 
#     stderr=None, 
#     status='success', 
#     execution_time=2.7027177810668945
# )
```

### Using as a tool

You can use `PyodideSandbox` as a LangChain tool inside an agent.

```python
from langchain_sandbox import PyodideSandboxTool

tool = PyodideSandboxTool()
result = await tool.ainvoke("print('Hello, world!')")
```

If you want to persist state between code executions (to persist variables, imports,
and definitions, etc.), you need to invoke the tool with `thread_id` in the config:

```python
code = """\
import numpy as np
x = np.array([1, 2, 3])
print(x)
"""
result = await tool.ainvoke(
    code,
    config={"configurable": {"thread_id": "123"}},
)

second_result = await tool.ainvoke(
    "print(float(x[0]))",  # tool is aware of the previous result
    config={"configurable": {"thread_id": "123"}},
)
```

### Using with an agent

You can use `PyodideSandboxTool` inside a LangGraph agent. If you are using this tool inside an agent, you can invoke the agent with a config, and it will automatically be passed to the tool:

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
    {"messages": [{"role": "user", "content": "what's 5 + 7? save result "}]},
    config={"configurable": {"thread_id": "123"}},
)
second_result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's the sine of that?"}]},
    config={"configurable": {"thread_id": "123"}},
)
```

## 🧩 Components

The sandbox consists of two main components:

- **`pyodide-sandbox-js`**: JavaScript/TypeScript module using Deno to provide the core sandboxing functionality.
- **`sandbox-py`**: Contains `PyodideSandbox` which just wraps the JavaScript/TypeScript module and executes it as a subprocess.
