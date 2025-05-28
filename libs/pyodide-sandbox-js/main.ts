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
import os
import base64
from pathlib import Path
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

def perform_fs_operation(op) -> dict:
    """Filesystem operation function for file operations."""
    try:
        if hasattr(op, 'to_py'):
            op = op.to_py()
        
        operation = op.get("operation")
        path = op.get("path")
        content = op.get("content")
        encoding = op.get("encoding", "utf-8")
        destination = op.get("destination")
        
        if operation == "read":
            if os.path.exists(path):
                if encoding == "binary":
                    with open(path, "rb") as f:
                        content = base64.b64encode(f.read()).decode('ascii')
                    return {"success": True, "content": content, "is_binary": True}
                else:
                    with open(path, "r", encoding=encoding) as f:
                        content = f.read()
                    return {"success": True, "content": content, "is_binary": False}
            else:
                return {"success": False, "error": f"File not found: {path}"}
                
        elif operation == "write":
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            if encoding == "binary":
                content = base64.b64decode(content)
                with open(path, "wb") as f:
                    f.write(content)
            else:
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)
                    
            exists = os.path.exists(path)
            if exists:
                return {"success": True}
            else:
                return {"success": False, "error": f"Failed to create file at {path}"}
            
        elif operation == "list":
            if os.path.exists(path):
                items = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    stat_info = os.stat(item_path)
                    items.append({
                        "name": item,
                        "is_dir": os.path.isdir(item_path),
                        "is_file": os.path.isfile(item_path),
                        "size": stat_info.st_size,
                        "modified": stat_info.st_mtime
                    })
                return {"success": True, "items": items}
            else:
                return {"success": False, "error": f"Directory not found: {path}"}
                
        elif operation == "mkdir":
            try:
                os.makedirs(path, exist_ok=True)
                exists = os.path.exists(path)
                return {"success": exists, "error": None if exists else "Failed to create directory"}
            except Exception as e:
                return {"success": False, "error": f"Error creating directory: {str(e)}"}
            
        elif operation == "exists":
            exists = os.path.exists(path)
            return {"success": True, "exists": exists}
            
        elif operation == "remove":
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                return {"success": True}
            else:
                return {"success": False, "error": f"Path not found for removal: {path}"}
                
        elif operation == "copy":
            if not destination:
                return {"success": False, "error": "Destination path required for copy operation"}
            if os.path.exists(path):
                import shutil
                if os.path.isfile(path):
                    shutil.copy2(path, destination)
                elif os.path.isdir(path):
                    shutil.copytree(path, destination, dirs_exist_ok=True)
                return {"success": True}
            else:
                return {"success": False, "error": f"Source path not found for copy: {path}"}
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

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
            return []
    else:
        imports: list[str] = source_code_or_imports

    to_install = find_imports_to_install(imports)
    # Merge with additional packages
    for package in additional_packages:
        if package not in [entry["package"] for entry in to_install]:
            to_install.append(dict(module=package, package=package))

    if to_install:
        try:
            import micropip  # noqa
        except ModuleNotFoundError:
            await pyodide_js.loadPackage("micropip")
            import micropip  # noqa

        for entry in to_install:
            try:
                await micropip.install(entry["package"])
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

interface FileSystemOptions {
  enableFileSystem?: boolean;
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
  fileSystemOperations?: any[];
  fileSystemInfo?: {
    type: "memfs";
    mountPoint: string;
    workingDirectory: string;
    mounted: boolean;
  };
}

interface FileSystemOperation {
  operation: "read" | "write" | "list" | "mkdir" | "exists" | "remove" | "copy";
  path: string;
  content?: string | Uint8Array;
  encoding?: string;
  destination?: string;
}


function resolvePathInSandbox(
  inputPath: string,
  mountPoint: string = "/sandbox"
): string {
  // Se já é absoluto, retorna como está
  if (inputPath.startsWith("/")) {
    return inputPath;
  }
  
  // Resolve direto no mount point
  if (inputPath.startsWith("./")) {
    const cleanPath = inputPath.substring(2);
    return `${mountPoint}/${cleanPath}`.replace(/\/+/g, "/");
  } else if (inputPath.startsWith("../")) {
    return `${mountPoint}/${inputPath}`.replace(/\/+/g, "/");
  } else {
    return `${mountPoint}/${inputPath}`.replace(/\/+/g, "/");
  }
}

/**
 * Setup memory filesystem environment in Python
 */
function setupFileSystem(pyodide: any): void {
  const mountPoint = "/sandbox";
  
  pyodide.runPython(`
import os
import sys

# Setup memory filesystem environment
MOUNT_POINT = "${mountPoint}"

# Ensure directory exists
os.makedirs(MOUNT_POINT, exist_ok=True)

# Change to mount point
os.chdir(MOUNT_POINT)

# Make variables available globally
sys.modules['__main__'].MOUNT_POINT = MOUNT_POINT

# Add helper function for path resolution
def resolve_path(path):
    """Resolve a path relative to the sandbox"""
    if path.startswith("/"):
        return path
    return os.path.join(MOUNT_POINT, path)

sys.modules['__main__'].resolve_path = resolve_path
  `);
}

async function initPyodide(pyodide: any, options: FileSystemOptions = {}): Promise<void> {
  const sys = pyodide.pyimport("sys");
  const pathlib = pyodide.pyimport("pathlib");

  const dirPath = "/tmp/pyodide_worker_runner/";
  sys.path.append(dirPath);
  pathlib.Path(dirPath).mkdir();
  pathlib.Path(dirPath + "prepare_env.py").write_text(prepareEnvCode);

  // Initialize filesystem if enabled
  if (options.enableFileSystem) {
    // Ensure sandbox mount point exists
    try {
      pyodide.FS.mkdirTree("/sandbox");
    } catch (e) {
      // Directory might already exist, which is fine
    }
    
    setupFileSystem(pyodide);
  }
}

async function performFileSystemOperations(
  pyodide: any,
  operations: FileSystemOperation[],
  options: FileSystemOptions = {}
): Promise<any[]> {
  const results: any[] = [];

  // Ensure sandbox mount point exists
  try {
    pyodide.FS.mkdirTree("/sandbox");
  } catch (e) {
    // Directory might already exist, which is fine
  }

  const prepare_env = pyodide.pyimport("prepare_env");

  for (const op of operations) {
    try {
      // Resolve paths using sandbox resolution
      const resolvedPath = resolvePathInSandbox(op.path, "/sandbox");
      let resolvedDestination: string | undefined;
      
      if (op.operation === "copy" && op.destination) {
        resolvedDestination = resolvePathInSandbox(op.destination, "/sandbox");
      }

      // Create resolved operation
      const resolvedOp = { 
        ...op, 
        path: resolvedPath,
        ...(resolvedDestination && { destination: resolvedDestination })
      };

      // Handle binary write operations
      if (op.operation === "write" && typeof op.content === "string") {
        if (op.encoding === "binary") {
          const result = await prepare_env.perform_fs_operation(resolvedOp);
          results.push(result.toJs());
          continue;
        }

        // Use pyodide.FS for text writes (better performance)
        try {
          const parentDir = resolvedPath.substring(0, resolvedPath.lastIndexOf("/"));
          if (parentDir) {
            pyodide.FS.mkdirTree(parentDir);
          }
          pyodide.FS.writeFile(resolvedPath, op.content, { encoding: op.encoding || "utf8" });
          results.push({ success: true, operation: op.operation, path: resolvedPath });
          continue;
        } catch {
          // Fallback to Python method if pyodide.FS fails
        }
      }

      // Use Python method for other operations
      const result = await prepare_env.perform_fs_operation(resolvedOp);
      results.push(result.toJs());

    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      results.push({
        success: false,
        error: errorMessage,
        operation: op.operation,
        path: op.path,
      });
    }
  }

  return results;
}

async function runPython(
  pythonCode: string,
  options: {
    stateful?: boolean;
    sessionBytes?: string;
    sessionMetadata?: string;
    fileSystemOptions?: FileSystemOptions;
    fileSystemOperations?: FileSystemOperation[];
  } = {}
): Promise<PyodideResult> {
  const output: string[] = [];
  const err_output: string[] = [];
  const originalLog = console.log;
  console.log = (...args: any[]) => {}

  try {
    const pyodide = await loadPyodide({
      stdout: (msg) => output.push(msg),
      stderr: (msg) => err_output.push(msg),
    });

    await pyodide.loadPackage(["micropip"], {
      messageCallback: () => {},
      errorCallback: (msg: string) => {
        output.push(`install error: ${msg}`)
      },
    });

    // Auto-enable filesystem if operations are provided or explicitly enabled
    const shouldEnableFileSystem = 
      options.fileSystemOperations?.length > 0 || 
      options.fileSystemOptions?.enableFileSystem ||
      // Detect file operations in Python code
      (pythonCode.includes("open(") || 
      pythonCode.includes("with open") ||
      pythonCode.includes("os.") ||
      pythonCode.includes("pathlib") ||
      pythonCode.includes("Path("));

    const fsOptions: FileSystemOptions = {
      enableFileSystem: shouldEnableFileSystem,
      ...options.fileSystemOptions
    };

    await initPyodide(pyodide, fsOptions);

    // Determine session metadata
    let sessionMetadata: SessionMetadata;
    if (options.sessionMetadata) {
      sessionMetadata = JSON.parse(options.sessionMetadata);
    } else {
      sessionMetadata = {
        created: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        packages: [],
      };
    }
    let sessionData: Uint8Array | null = null;

    // Import prepared environment module
    const prepare_env = pyodide.pyimport("prepare_env");

    // Execute filesystem operations before Python code
    let fileSystemResults: any[] = [];
    if (options.fileSystemOperations && options.fileSystemOperations.length > 0) {
      fileSystemResults = await performFileSystemOperations(pyodide, options.fileSystemOperations, fsOptions);
    }

    // Prepare packages to install
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

    if (installErrors.length > 0) {
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
      await prepare_env.load_session_bytes(sessionData);
    }

    const packages = installedPackages.map((pkg: any) => pkg.get("package"));

    console.log = originalLog;
    
    // Execute Python code
    const rawValue = await pyodide.runPythonAsync(pythonCode);
    const jsonValue = await prepare_env.dumps(rawValue);

    // Update session metadata
    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      sessionData = await prepare_env.dump_session_bytes() as Uint8Array;
    }

    // Build result
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

    // Add filesystem info if enabled
    if (fsOptions.enableFileSystem) {
      result["fileSystemOperations"] = fileSystemResults;
      result["fileSystemInfo"] = { 
        type: "memfs",
        mountPoint: "/sandbox",
        workingDirectory: "", 
        mounted: true
      };
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
    string: ["code", "file", "session-bytes", "session-metadata", "fs-operations"],
    alias: {
      c: "code",
      f: "file",
      h: "help",
      V: "version",
      s: "stateful",
      b: "session-bytes",
      m: "session-metadata",
      x: "fs-operations",
    },
    boolean: ["help", "version", "stateful"],
    default: { 
      help: false, 
      version: false, 
      stateful: false
    },
  });

  if (flags.help) {
    console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>              Python code to execute
  -f, --file <path>              Path to Python file to execute
  -s, --stateful <bool>          Use a stateful session
  -b, --session-bytes <bytes>    Session bytes
  -m, --session-metadata         Session metadata
  -x, --fs-operations <json>     JSON array of filesystem operations
  -h, --help                     Display help
  -V, --version                  Display version
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
    fsOperations: flags["fs-operations"],
  };

  if (!options.code && !options.file) {
    console.error("Error: You must provide Python code using either -c/--code or -f/--file option.");
    Deno.exit(1);
  }

  let pythonCode = "";

  if (options.file) {
    try {
      const filePath = options.file.startsWith("/")
        ? options.file
        : join(Deno.cwd(), options.file);
      pythonCode = await Deno.readTextFile(filePath);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`Error reading file ${options.file}:`, errorMessage);
      Deno.exit(1);
    }
  } else {
    pythonCode = options.code?.replace(/\\n/g, "\n") ?? "";
  }

  let fileSystemOperations: FileSystemOperation[] = [];
  if (options.fsOperations) {
    try {
      fileSystemOperations = JSON.parse(options.fsOperations);
    } catch (error: unknown) {
      console.error("Error parsing filesystem operations:", error instanceof Error ? error.message : String(error));
      Deno.exit(1);
    }
  }

  const runOptions: any = {
    stateful: options.stateful,
    sessionBytes: options.sessionBytes,
    sessionMetadata: options.sessionMetadata,
  };

  // Enable filesystem if operations are provided
  if (fileSystemOperations.length > 0) {
    runOptions.fileSystemOptions = {
      enableFileSystem: true,
    };
    runOptions.fileSystemOperations = fileSystemOperations;
  }

  const result = await runPython(pythonCode, runOptions);

  // Output result
  const outputJson: any = {
    stdout: result.stdout?.join('\n') || null,  // <-- ADICIONAR '\n'
    stderr: result.success ? (result.stderr?.join('\n') || null) : result.error || null,
    result: result.success ? JSON.parse(result.jsonResult || 'null') : null,
    success: result.success,
    sessionBytes: result.sessionBytes,
    sessionMetadata: result.sessionMetadata,
  };

  // Include filesystem info if used
  if (result.fileSystemInfo) {
    outputJson.fileSystemInfo = result.fileSystemInfo;
  }
  if (result.fileSystemOperations) {
    outputJson.fileSystemOperations = result.fileSystemOperations;
  }

  console.log(JSON.stringify(outputJson));

  if (!result.success) {
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main().catch((err) => {
    console.error("Unhandled error:", err);
    Deno.exit(1);
  });
}

export { runPython, resolvePathInSandbox, type FileSystemOperation, type FileSystemOptions };