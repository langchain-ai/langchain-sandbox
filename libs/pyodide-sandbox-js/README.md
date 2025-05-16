# Pyodide Sandbox

A simple CLI tool for running Python code in a sandboxed environment using Pyodide.

## Installation

```shell
# Install using Deno
deno install -A pyodide-sandbox.ts
```

## Testing

Run the tests with:

```shell
deno test --allow-read --allow-write
```

## Usage

### Basic Examples

Run a simple Python statement:

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto ./main.ts -c "print('Hello, World!')"
```

Run calculations and get results:

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto ./main.ts -c "result = 2 + 2; print(f'2 + 2 = {result}')"
```

### Using External Packages

Imports will be automatically detected and installed using micropip if possible:

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto ./main.ts -c "import numpy as np; x = np.ones((3, 3)); print(x)"
```

### Stateful Sessions

Create a stateful session that remembers variables:

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto ./main.ts -c "import numpy as np; x = np.ones((3, 3))" -s -d sessions
```

Use the previous session to access variables:

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto ./main.ts -c "print(x)" -s -d sessions
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