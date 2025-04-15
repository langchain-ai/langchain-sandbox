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
- **Network requests/file access**: Currently network requests and ability to write read/files are not supported

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

## 🧩 Components

The sandbox consists of two main components:

- **`pyodide-sandbox-js`**: JavaScript/TypeScript module using Deno to provide the core sandboxing functionality.
- **`sandbox-py`**: Contains `PyodideSandbox` which just wraps the JavaScript/TypeScript module and executes it as a subprocess.
