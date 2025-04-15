# Pyodide Sandbox

A simple CLI tool for running Python code in a sandboxed environment using Pyodide.

```shell
deno run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto  ./main.ts -c "import numpy as np; x = np.ones((3, 3))" -s 123 -d sessions
```

```shell
deno  run -N -R=node_modules,sessions -W=node_modules,sessions --node-modules-dir=auto  ./main.ts -c "print(x)" -s 123 -d sessions 
```
