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


> [!warning]
> Use `allow_net` to limit the network requests that can be made by the sandboxed code to avoid SSRF attacks
> https://docs.deno.com/runtime/fundamentals/security/#network-access

```python
from langchain_sandbox import PyodideSandbox

# Create a sandbox instance
sandbox = PyodideSandbox(
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
print(await sandbox.execute(code))

# CodeExecutionResult(
#   result=None, 
#   stdout='[1 2 3]', 
#   stderr=None, 
#   status='success', 
#   execution_time=2.8578367233276367,
#   session_metadata={'created': '2025-05-15T21:26:37.204Z', 'lastModified': '2025-05-15T21:26:37.831Z', 'packages': ['numpy']},
#   session_bytes=None
# )
```

### Stateful Sandbox

If you want to persist state between code executions (to persist variables, imports,
and definitions, etc.), you can set `stateful=True` in the sandbox. This will return
`session_bytes` and `session_metadata` that you can pass to `.execute()`.

> [!warning]
> `session_bytes` contains pickled session state. It should not be unpickled
> and is only meant to be used by the sandbox itself

```python
sandbox = PyodideSandbox(
    # Create stateful sandbox
    stateful=True,
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True,
)
code = """\
import numpy as np
x = np.array([1, 2, 3])
print(x)
"""

result = await sandbox.execute(code)

# Pass previous result
print(await sandbox.execute("float(x[0])", session_bytes=result.session_bytes, session_metadata=result.session_metadata))

#  CodeExecutionResult(
#     result=1, 
#     stdout=None, 
#     stderr=None, 
#     status='success', 
#     execution_time=2.7027177810668945
#     session_metadata={'created': '2025-05-15T21:27:57.120Z', 'lastModified': '2025-05-15T21:28:00.061Z', 'packages': ['numpy', 'dill']},
#     session_bytes=b'\x80\x04\x95d\x01\x00..."
# )
```

### Using as a tool

You can use `PyodideSandbox` as a LangChain tool:

```python
from langchain_sandbox import PyodideSandboxTool

tool = PyodideSandboxTool()
result = await tool.ainvoke("print('Hello, world!')")
```

### Using with an agent

You can use sandbox tools inside a LangGraph agent:

```python
from langgraph.prebuilt import create_react_agent
from langchain_sandbox import PyodideSandboxTool

tool = PyodideSandboxTool(
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True
)
agent = create_react_agent(
    "anthropic:claude-3-7-sonnet-latest",
    tools=[tool],
)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
)
```

#### Stateful Tool

> [!important]
> **Stateful** `PyodideSandboxTool` works only in LangGraph agents that use the prebuilt [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.chat_agent_executor.create_react_agent) or [`ToolNode`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode).

If you want to persist state between code executions (to persist variables, imports,
and definitions, etc.), you need to set `stateful=True`:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_sandbox import PyodideSandboxTool, PyodideSandbox

class State(AgentState):
    # important: add session_bytes & session_metadata keys to your graph state schema - 
    # these keys are required to store the session data between tool invocations.
    # `session_bytes` contains pickled session state. It should not be unpickled
    # and is only meant to be used by the sandbox itself
    session_bytes: bytes
    session_metadata: dict

tool = PyodideSandboxTool(
    # Create stateful sandbox
    stateful=True,
    # Allow Pyodide to install python packages that
    # might be required.
    allow_net=True
)
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



See full examples here:

* [ReAct agent](examples/react_agent.py)
* [CodeAct agent](examples/codeact_agent.py)

## 🧩 Components

The sandbox consists of two main components:

- **`pyodide-sandbox-js`**: JavaScript/TypeScript module using Deno to provide the core sandboxing functionality.
- **`sandbox-py`**: Contains `PyodideSandbox` which just wraps the JavaScript/TypeScript module and executes it as a subprocess.
