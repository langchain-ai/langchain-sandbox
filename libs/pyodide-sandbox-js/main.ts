// Parts of this code were adapted from
// https://github.com/alexmojaki/pyodide-worker-runner/blob/master/lib/pyodide_worker_runner.py
// and
// https://github.com/pydantic/pydantic-ai/blob/main/mcp-run-python/src/runCode.ts
import { loadPyodide } from "pyodide";
import { join } from "@std/path";
import { parseArgs } from "@std/cli/parse-args";

const pkgVersion = "0.0.7";

const prepareEnvCode = `
import datetime
import importlib
import json
import sys
from typing import Union, TypedDict, List, Any, Callable, Literal

try:
    from pyodide.code import find_imports  # noqa
except ImportError:
    from pyodide import find_imports  # noqa

import pyodide_js  # noqa

sys.setrecursionlimit(400)


class InstallEntry(TypedDict):
    module: str
    package: str


def find_imports_to_install(imports: list[str]) -> list[InstallEntry]:
    """
    Given a list of module names being imported, return a list of dicts
    representing the packages that need to be installed to import those modules.
    The returned list will only contain modules that aren't already installed.
    Each returned dict has the following keys:
      - module: the name of the module being imported
      - package: the name of the package that needs to be installed
    """
    try:
        to_package_name = pyodide_js._module._import_name_to_package_name.to_py()
    except AttributeError:
        to_package_name = pyodide_js._api._import_name_to_package_name.to_py()

    to_install: list[InstallEntry] = []
    for module in imports:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            to_install.append(
                dict(
                    module=module,
                    package=to_package_name.get(module, module),
                )
            )
    return to_install


async def install_imports(
    source_code_or_imports: Union[str, list[str]],
    additional_packages: list[str] = [],
    message_callback: Callable[
          [
              Literal[
                "failed",
              ],
              Union[InstallEntry, list[InstallEntry]],
          ],
          None,
      ] = lambda event_type, data: None,
) -> List[InstallEntry]:
    if isinstance(source_code_or_imports, str):
        try:
            imports: list[str] = find_imports(source_code_or_imports)
        except SyntaxError:
            return
    else:
        imports: list[str] = source_code_or_imports

    to_install = find_imports_to_install(imports)
    # Merge with additional packages
    for package in additional_packages:
        if package not in to_install:
            to_install.append(dict(module=package, package=package))

    if to_install:
        try:
            import micropip  # noqa
        except ModuleNotFoundError:
            await pyodide_js.loadPackage("micropip")
            import micropip  # noqa

        for entry in to_install:
            try:
                await micropip.install(entry["package"], verbose=False)
            except Exception as e:
                message_callback("failed", entry["package"])
                break # Fail fast
    return to_install


def load_session_bytes(session_bytes: bytes) -> list[str]:
    """Load the session module."""
    import dill
    import io

    buffer = io.BytesIO(session_bytes.to_py())
    dill.session.load_session(filename=buffer)


def dump_session_bytes() -> bytes:
    """Dump the session module."""
    import dill
    import io

    buffer = io.BytesIO()
    dill.session.dump_session(filename=buffer)
    return buffer.getvalue()


def robust_serialize(obj):
    """Recursively converts an arbitrary Python object into a JSON-serializable structure.

    The function handles:
      - Primitives: str, int, float, bool, None are returned as is.
      - Lists and tuples: Each element is recursively processed.
      - Dictionaries: Keys are converted to strings (if needed) and values are recursively processed.
      - Sets: Converted to lists.
      - Date and datetime objects: Converted to their ISO format strings.
      - For unsupported/unknown objects, a dictionary containing a 'type'
        indicator and the object's repr is returned.
    """
    # Base case: primitives that are already JSON-serializable
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Process lists or tuples recursively.
    if isinstance(obj, (list, tuple)):
        return [robust_serialize(item) for item in obj]

    # Process dictionaries.
    if isinstance(obj, dict):
        # Convert keys to strings if necessary and process values recursively.
        return {str(key): robust_serialize(value) for key, value in obj.items()}

    # Process sets by converting them to lists.
    if isinstance(obj, (set, frozenset)):
        return [robust_serialize(item) for item in obj]

    # Process known datetime objects.
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    # Fallback: for objects that are not directly serializable,
    # return a dictionary with type indicator and repr.
    return {"type": "not_serializable", "repr": repr(obj)}


def dumps(result: Any) -> str:
    """Get the result of the session."""
    result = robust_serialize(result)
    return json.dumps(result)
`;

interface SessionMetadata {
  created: string;
  lastModified: string;
  packages: string[];
}

interface PyodideResult {
  success: boolean;
  result?: any;
  stdout?: string[];
  stderr?: string[];
  error?: string;
  jsonResult?: string;
  sessionBytes?: Uint8Array;
  sessionMetadata?: SessionMetadata;
}

async function initPyodide(pyodide: any): Promise<void> {
  const sys = pyodide.pyimport("sys");
  const pathlib = pyodide.pyimport("pathlib");

  const dirPath = "/tmp/pyodide_worker_runner/";
  sys.path.append(dirPath);
  pathlib.Path(dirPath).mkdir();
  pathlib.Path(dirPath + "prepare_env.py").write_text(prepareEnvCode);
}

async function runPython(
  pythonCode: string,
  options: {
    stateful?: boolean;
    sessionBytes?: string;
    sessionMetadata?: string;
  }
): Promise<PyodideResult> {
  const output: string[] = [];
  const err_output: string[] = [];
  const originalLog = console.log;
  console.log = (...args: any[]) => {}

  try {
    const pyodide = await loadPyodide({
      stdout: (msg) => output.push(msg),
      stderr: (msg) => err_output.push(msg),
    })
    await pyodide.loadPackage(["micropip"], {
      messageCallback: () => {},
      errorCallback: (msg: string) => {
        output.push(`install error: ${msg}`)
      },
    });
    await initPyodide(pyodide);


    // Determine session directory
    let sessionMetadata: SessionMetadata;
    if (options.sessionMetadata) {
      sessionMetadata = JSON.parse(options.sessionMetadata);
    } else {
      sessionMetadata = {
        created: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        packages: [],
      };
    };
    let sessionData: Uint8Array | null = null;

    if (options.sessionBytes && !options.sessionMetadata) {
      console.error("sessionMetadata is required when providing sessionBytes");
      return { success: false, error: "sessionMetadata is required when providing sessionBytes" };
    }
      
    // Import our prepared environment module
    const prepare_env = pyodide.pyimport("prepare_env");
    // Prepare additional packages to install (include dill)
    const defaultPackages = options.stateful ? ["dill"] : [];
    const additionalPackagesToInstall = options.sessionBytes
      ? [...new Set([...defaultPackages, ...sessionMetadata.packages])]
      : defaultPackages;

    let installErrors: string[] = []

    const installedPackages = await prepare_env.install_imports(
      pythonCode,
      additionalPackagesToInstall,
      (event_type: string, data: string) => {
        if (event_type === "failed") {
          installErrors.push(data)
        }
      }
    );

    output.length = 0; // Clear output after installation

    if (installErrors.length > 0) {
      // Restore the original console.log function
      console.log = originalLog;
      return {
        success: false,
        error: `Failed to install required Python packages: ${installErrors.join(", ")}. ` +
          `This is likely because these packages are not available in the Pyodide environment. ` +
          `Pyodide is a Python runtime that runs in the browser and has a limited set of ` +
          `pre-built packages. You may need to use alternative packages that are compatible ` +
          `with Pyodide.`
      };
    }

    if (options.sessionBytes) {
      sessionData = Uint8Array.from(JSON.parse(options.sessionBytes));
      // Run session preamble
      await prepare_env.load_session_bytes(sessionData);
    }

    const packages = installedPackages.map((pkg: any) => pkg.get("package"));

    // Restore the original console.log function
    console.log = originalLog;
    // Run the Python code
    const rawValue = await pyodide.runPythonAsync(pythonCode);
    // Dump result to string
    const jsonValue = await prepare_env.dumps(rawValue);

    // Update session metadata with installed packages
    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      // Save session state to sessionBytes
      sessionData = await prepare_env.dump_session_bytes() as Uint8Array;
    };
    // Return the result with stdout and stderr output
    const result: PyodideResult = {
      success: true, 
      result: rawValue,
      jsonResult: jsonValue,
      stdout: output,
      stderr: err_output,
      sessionMetadata: sessionMetadata,
    };
    if (options.stateful && sessionData) {
      result["sessionBytes"] = sessionData;
    }
    return result;
  } catch (error: any) {
    return { 
      success: false, 
      error: error.message,
      stdout: output,
      stderr: err_output
    };
  }
}

async function main(): Promise<void> {
  const flags = parseArgs(Deno.args, {
    string: ["code", "file", "session-bytes", "session-metadata"],
    alias: {
      c: "code",
      f: "file",
      h: "help",
      V: "version",
      s: "stateful",
      b: "session-bytes",
      m: "session-metadata",
    },
    boolean: ["help", "version", "stateful"],
    default: { help: false, version: false, stateful: false },
  });

  if (flags.help) {
    console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
  -s, --stateful <bool>        Use a stateful session
  -b, --session-bytes <bytes>  Session bytes
  -m, --session-metadata       Session metadata
  -h, --help                   Display help
  -V, --version                Display version
`);
    return;
  }

  if (flags.version) {
    console.log(pkgVersion)
    return
  }

  const options = {
    code: flags.code,
    file: flags.file,
    stateful: flags.stateful,
    sessionBytes: flags["session-bytes"],
    sessionMetadata: flags["session-metadata"],
  };

  if (!options.code && !options.file) {
    console.error(
      "Error: You must provide Python code using either -c/--code or -f/--file option.\nUse --help for usage information."
    );
    Deno.exit(1);
  }

  // Get Python code from file or command line argument
  let pythonCode = "";

  if (options.file) {
    try {
      // Resolve relative or absolute file path
      const filePath = options.file.startsWith("/")
        ? options.file
        : join(Deno.cwd(), options.file);
      pythonCode = await Deno.readTextFile(filePath);
    } catch (error: any) {
      console.error(`Error reading file ${options.file}:`, error.message);
      Deno.exit(1);
    }
  } else {
    // Process code from command line (replacing escaped newlines)
    pythonCode = options.code?.replace(/\\n/g, "\n") ?? "";
  }

  const result = await runPython(pythonCode, {
    stateful: options.stateful,
    sessionBytes: options.sessionBytes,
    sessionMetadata: options.sessionMetadata,
  });

  // Exit with error code if Python execution failed
  // Create output JSON with stdout, stderr, and result
  const outputJson = {
    stdout: result.stdout?.join('') || null,
    stderr: result.success ? (result.stderr?.join('') || null) : result.error || null,
    result: result.success ? JSON.parse(result.jsonResult || 'null') : null,
    success: result.success,
    sessionBytes: result.sessionBytes,
    sessionMetadata: result.sessionMetadata,
  };

  // Output as JSON to stdout
  console.log(JSON.stringify(outputJson));

  // Exit with error code if Python execution failed
  if (!result.success) {
    Deno.exit(1);
  }
}

// If this module is run directly
if (import.meta.main) {
  // Override the global environment variables that Deno's permission prompts look for
  // to suppress color-related permission prompts
  main().catch((err) => {
    console.error("Unhandled error:", err);
    Deno.exit(1);
  });
}

export { runPython };
