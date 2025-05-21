# Pyodide Sandbox

A simple CLI tool for running Python code in a sandboxed environment using Pyodide.

## Installation

No installation needed! Run directly using Deno with JSR.

## Usage

### Basic Examples

Run a simple Python statement:

```shell
deno run -A jsr:@langchain/pyodide-sandbox -c "print('Hello, World!')"
```

Run calculations and get results:

```shell
deno run -A jsr:@langchain/pyodide-sandbox -c "result = 2 + 2; print(f'2 + 2 = {result}')"
```

### Using External Packages

Imports will be automatically detected and installed using micropip if possible:

```shell
deno run -A jsr:@langchain/pyodide-sandbox -c "import numpy as np; x = np.ones((3, 3)); print(x)"
```

### Stateful Sessions

Create a stateful session that remembers variables:

```shell
deno run -A jsr:@langchain/pyodide-sandbox -c "import numpy as np; x = np.ones((3, 3))" -s
```

Use the previous session to access variables:

```shell
deno run -A jsr:@langchain/pyodide-sandbox -c "print(x)" -s -b <session_bytes>
```

## Notes on Package Compatibility

- Most pure Python packages work seamlessly
- Packages with C extensions or system dependencies may encounter issues
- The sandbox uses micropip to install packages automatically
- Some packages may require manual installation or configuration

## CLI Options

```
OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
  -s, --stateful <bool>        Use a stateful session
  -b, --session-bytes <bytes>  Session bytes
  -m, --session-metadata       Session metadata
  -h, --help                   Display help
  -V, --version                Display version
```

## Local Development

### Testing

Run the tests with:

```shell
deno test --allow-read --allow-write
```

