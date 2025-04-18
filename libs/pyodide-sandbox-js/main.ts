import { loadPyodide } from "pyodide";
import { join } from "@std/path";
import { parseArgs } from "@std/cli/parse-args";


const pkgVersion = "0.0.7";

// Python environment preparation code
// This code was adapted from
// https://github.com/alexmojaki/pyodide-worker-runner/blob/master/lib/pyodide_worker_runner.py
const prepareEnvCode = `
import datetime
import importlib
import json
import sys
from typing import Union, TypedDict, List, Any

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
            await micropip.install(entry["package"])
    return to_install


def load_session(path: str) -> List[str]:
    """Load the session module."""
    import dill

    dill.session.load_session(filename=path)


def dump_session(path: str) -> None:
    """Dump the session module."""
    import dill

    dill.session.dump_session(filename=path)


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
    session?: string;
    sessionsDir?: string;
  }
): Promise<PyodideResult> {
  const output: string[] = [];
  const err_output: string[] = [];
  const originalLog = console.log;
  console.log = (...args: any[]) => {}

  const pyodide = await loadPyodide({
    stdout: (msg) => output.push(msg),
    stderr: (msg) => err_output.push(msg),
  });
  await pyodide.loadPackage(["micropip"], {
    messageCallback: () => {},
    errorCallback: (msg: string) => {
      output.push(`install error: ${msg}`);
    },
  });
  await initPyodide(pyodide);

  // Determine session directory
  const sessionsDir = options.sessionsDir || Deno.cwd();
  let sessionMetadata: SessionMetadata = {
    created: new Date().toISOString(),
    lastModified: new Date().toISOString(),
    packages: [],
  };
  let sessionJsonPath: string;
  let isExistingSession = false;

  // Handle session if provided
  if (options.session) {
    // Create session directory path
    const sessionDirPath = join(sessionsDir, options.session);
    const sessionPklPath = join(sessionDirPath, `session.pkl`);
    sessionJsonPath = join(sessionDirPath, `session.json`);

    // Ensure session directory exists
    try {
      const dirInfo = await Deno.stat(sessionDirPath);
      if (!dirInfo.isDirectory) {
        console.error(`Path exists but is not a directory: ${sessionDirPath}`);
        return {
          success: false,
          error: `Path exists but is not a directory: ${sessionDirPath}`,
        };
      }
    } catch (error) {
      // Directory doesn't exist, create it
      if (error instanceof Deno.errors.NotFound) {
        try {
          await Deno.mkdir(sessionDirPath, { recursive: true });
        } catch (mkdirError: any) {
          console.error(
            `Error creating session directory: ${mkdirError.message}`
          );
          return { success: false, error: mkdirError.message };
        }
      } else {
        console.error(
          `Error accessing session directory: ${(error as Error).message}`
        );
        return { success: false, error: (error as Error).message };
      }
    }

    try {
      // Check if both session pickle and metadata files exist
      const [pklStat, jsonStat] = await Promise.all([
        Deno.stat(sessionPklPath),
        Deno.stat(sessionJsonPath),
      ]).catch(() => [null, null]);

      isExistingSession = (pklStat?.isFile && jsonStat?.isFile) || false;
    } catch {
      // Error checking files, assume they don't exist
      isExistingSession = false;
    }

    // Create or load session metadata
    if (!isExistingSession) {
      // Create new session metadata file
      await Deno.writeTextFile(
        sessionJsonPath,
        JSON.stringify(sessionMetadata, null, 2)
      );
    } else {
      // Load existing session metadata
      const jsonContent = await Deno.readTextFile(sessionJsonPath);
      sessionMetadata = JSON.parse(jsonContent);
    }

    // Load PKL file into pyodide if it exists
    try {
      const sessionData = await Deno.readFile(sessionPklPath);
      pyodide.FS.writeFile(`/${options.session}.pkl`, sessionData);
    } catch (error) {
      // File doesn't exist or can't be read, skip loading
    }
  }

  // Import our prepared environment module
  const prepare_env = pyodide.pyimport("prepare_env");

  // Prepare additional packages to install (include dill if using sessions)
  const additionalPackagesToInstall = options.session
    ? [...new Set([...sessionMetadata.packages, "dill"])]
    : [];

  const installedPackages = await prepare_env.install_imports(
    pythonCode,
    additionalPackagesToInstall
  );

  if (options.session && isExistingSession) {
    // Run session preamble
    await prepare_env.load_session(`/${options.session}.pkl`);
  }

  const packages = installedPackages.map((pkg: any) => pkg.get("package"));

  // Restore the original console.log function
  console.log = originalLog;
  // Run the Python code
  const rawValue = await pyodide.runPythonAsync(pythonCode);
  // Dump result to string
  const jsonValue = await prepare_env.dumps(rawValue);

  if (options.session) {
    // Save session state
    await prepare_env.dump_session(`/${options.session}.pkl`);

    // Update session metadata with installed packages
    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();
    await Deno.writeTextFile(
      sessionJsonPath as string,
      JSON.stringify(sessionMetadata, null, 2)
    );

    // Save session file back to host machine
    const sessionData = pyodide.FS.readFile(`/${options.session}.pkl`);
    const sessionDirPath = join(sessionsDir, options.session);
    const sessionPklPath = join(sessionDirPath, `session.pkl`);
    await Deno.writeFile(sessionPklPath, sessionData);
  }
  // Return the result with stdout and stderr output
  return {
    success: true,
    result: rawValue,
    jsonResult: jsonValue,
    stdout: output,
    stderr: err_output,
  };
}

async function main(): Promise<void> {
  const flags = parseArgs(Deno.args, {
    string: ["code", "file", "session", "sessions-dir"],
    alias: {
      c: "code",
      f: "file",
      s: "session",
      d: "sessions-dir",
      h: "help",
      V: "version",
    },
    boolean: ["help", "version"],
    default: { help: false, version: false },
  });

  if (flags.help) {
    console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
  -s, --session <string>       Session name
  -d, --sessions-dir <path>    Directory to store session files
  -h, --help                   Display help
  -V, --version                Display version
`);
    return;
  }

  if (flags.version) {
    console.log(pkgVersion);
    return;
  }

  const options = {
    code: flags.code,
    file: flags.file,
    session: flags.session,
    sessionsDir: flags["sessions-dir"],
  };

  if (!options.code && !options.file) {
    console.error(
      "Error: You must provide Python code using either -c/--code or -f/--file option.\nUse --help for usage information."
    );
    Deno.exit(1);
  }

  // Validate session ID if provided
  if (options.session) {
    const validSessionIdRegex = /^[a-zA-Z0-9_-]+$/;
    if (!validSessionIdRegex.test(options.session)) {
      console.error(
        "Error: Session ID must only contain letters, numbers, underscores, and hyphens."
      );
      Deno.exit(1);
    }
  }

  // Ensure sessions directory exists if specified
  if (options.sessionsDir) {
    try {
      try {
        const dirInfo = await Deno.stat(options.sessionsDir);
        if (!dirInfo.isDirectory) {
          throw new Error(
            `Path exists but is not a directory: ${options.sessionsDir}`
          );
        }
      } catch (error) {
        // Directory doesn't exist, create it
        if (error instanceof Deno.errors.NotFound) {
          await Deno.mkdir(options.sessionsDir, { recursive: true });
        } else {
          throw error;
        }
      }
    } catch (error: any) {
      console.error(`Error creating sessions directory: ${error.message}`);
      Deno.exit(1);
    }
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
    pythonCode = options.code.replace(/\\n/g, "\n");
  }

  const result = await runPython(pythonCode, {
    session: options.session,
    sessionsDir: options.sessionsDir,
  });

  let artifacts: string[] | null = null;
  if (options.session && options.sessionsDir) {
    artifacts = await listArtifacts(options.session, options.sessionsDir);
  }

  // Exit with error code if Python execution failed
  // Create output JSON with stdout, stderr, and result
  const outputJson = {
    stdout: result.success ? result.stdout.join("") || null : null,
    stderr: result.success
      ? result.stderr.join("") || null
      : result.error || null,
    result: result.success ? JSON.parse(result.jsonResult || "null") : null,
    success: result.success,
    files: artifacts,
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
