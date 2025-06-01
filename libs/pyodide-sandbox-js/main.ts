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
  result?: unknown;
  stdout?: string[];
  stderr?: string[];
  error?: string;
  jsonResult?: string;
  sessionBytes?: Uint8Array;
  sessionMetadata?: SessionMetadata;
  fileSystemOperations?: Record<string, unknown>[];
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

/**
 * Resolves a relative path within the sandbox environment.
 */
function resolvePathInSandbox(
  inputPath: string,
  mountPoint: string = "/sandbox"
): string {
  // If already absolute, return as is
  if (inputPath.startsWith("/")) {
    return inputPath;
  }
  
  // Resolve directly in mount point
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
 * Setup memory filesystem environment in Python.
 */
function setupFileSystem(pyodide: unknown): void {
  const mountPoint = "/sandbox";
  
  (pyodide as { runPython: (code: string) => void }).runPython(`
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
    if isinstance(path, str) and path.startswith("/"):
        return path
    return os.path.join(MOUNT_POINT, path)

sys.modules['__main__'].resolve_path = resolve_path
  `);
}

function initPyodide(pyodide: unknown): void {
  const sys = (pyodide as { pyimport: (name: string) => unknown }).pyimport("sys");
  const pathlib = (pyodide as { pyimport: (name: string) => unknown }).pyimport("pathlib");

  const dirPath = "/tmp/pyodide_worker_runner/";
  (sys as { path: { append: (path: string) => void } }).path.append(dirPath);
  (pathlib as { Path: (path: string) => { mkdir: () => void; write_text: (text: string) => void } }).Path(dirPath).mkdir();
  (pathlib as { Path: (path: string) => { mkdir: () => void; write_text: (text: string) => void } }).Path(dirPath + "prepare_env.py").write_text(prepareEnvCode);

  // Ensure sandbox mount point exists
  try {
    (pyodide as { FS: { mkdirTree: (path: string) => void } }).FS.mkdirTree("/sandbox");
  } catch (_e) {
    // Directory might already exist, which is fine
  }
  
  setupFileSystem(pyodide);
}

/**
 * Process stdin using ReadableStream for large files
 */
async function processStreamedFiles(pyodide: unknown): Promise<Record<string, unknown>[]> {
  const results: Record<string, unknown>[] = [];
  
  // Read binary protocol header
  const headerBuffer = new Uint8Array(8);
  const headerRead = await Deno.stdin.read(headerBuffer);
  
  if (!headerRead || headerRead < 8) {
    // No stdin data or insufficient data
    return results;
  }

  // Check magic header
  const magic = new TextDecoder().decode(headerBuffer.slice(0, 3));
  const version = headerBuffer[3];
  if (magic !== "PSB" || version !== 1) {
      throw new Error(`Invalid PSB header: ${magic} v${version}`);
  }

  // Get metadata length
  const metadataLength = new DataView(headerBuffer.buffer).getUint32(4, false);
  
  // Read metadata
  const metadataBuffer = new Uint8Array(metadataLength);
  const metadataRead = await Deno.stdin.read(metadataBuffer);
  
  if (!metadataRead || metadataRead < metadataLength) {
    throw new Error("Failed to read metadata");
  }

  // Parse metadata
  const metadata = JSON.parse(new TextDecoder().decode(metadataBuffer)) as {
    directories?: string[];
    files?: Array<{ path: string; size: number; binary: boolean }>;
  };
  
  // Process directories first
  if (metadata.directories) {
    for (const dir of metadata.directories) {
      const resolvedPath = resolvePathInSandbox(dir, "/sandbox");
      try {
        (pyodide as { FS: { mkdirTree: (path: string) => void } }).FS.mkdirTree(resolvedPath);
        results.push({
          success: true,
          operation: "mkdir",
          path: dir
        });
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        results.push({
          success: false,
          error: errorMsg,
          operation: "mkdir",
          path: dir
        });
      }
    }
  }
  
  // Process files
  if (metadata.files && metadata.files.length > 0) {
    for (const fileInfo of metadata.files) {
      const resolvedPath = resolvePathInSandbox(fileInfo.path, "/sandbox");
      
      // Create parent directories if needed
      const parentDir = resolvedPath.substring(0, resolvedPath.lastIndexOf("/"));
      if (parentDir) {
        try {
          (pyodide as { FS: { mkdirTree: (path: string) => void } }).FS.mkdirTree(parentDir);
        } catch (_e) {
          // Directory might already exist
        }
      }

      try {
        // Read file data
        const fileBuffer = new Uint8Array(fileInfo.size);
        let bytesRead = 0;
        
        // Read in chunks to handle large files efficiently
        while (bytesRead < fileInfo.size) {
          const chunkSize = Math.min(65536, fileInfo.size - bytesRead);
          const chunkBuffer = new Uint8Array(chunkSize);
          const readResult = await Deno.stdin.read(chunkBuffer);
          
          if (readResult === null) {
            throw new Error(`Unexpected end of stream at ${bytesRead}/${fileInfo.size} bytes`);
          }
          
          // Copy to the main buffer
          fileBuffer.set(chunkBuffer.subarray(0, readResult), bytesRead);
          bytesRead += readResult;
        }
        
        // Write to PyFS
        (pyodide as { FS: { writeFile: (path: string, data: Uint8Array) => void } }).FS.writeFile(resolvedPath, fileBuffer);
        
        results.push({
          success: true,
          operation: "write",
          path: fileInfo.path,
          size: bytesRead,
          binary: fileInfo.binary
        });
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        results.push({
          success: false,
          error: errorMsg,
          operation: "write",
          path: fileInfo.path
        });
      }
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
  } = {}
): Promise<PyodideResult> {
  const output: string[] = [];
  const err_output: string[] = [];
  const originalLog = console.log;
  console.log = (..._args: unknown[]) => {}

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

    initPyodide(pyodide);

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

    if (options.sessionBytes && !options.sessionMetadata) {
      console.error("sessionMetadata is required when providing sessionBytes");
      return { success: false, error: "sessionMetadata is required when providing sessionBytes" };
    }

    // Import prepared environment module
    const prepare_env = (pyodide as { pyimport: (name: string) => unknown }).pyimport("prepare_env");

    let fileSystemResults: Record<string, unknown>[] = [];
    
    if (!Deno.stdin.isTerminal()) {
      fileSystemResults = await processStreamedFiles(pyodide);
    }

    // Prepare packages to install (include dill)
    const defaultPackages = options.stateful ? ["dill"] : [];
    const additionalPackagesToInstall = options.sessionBytes
      ? [...new Set([...defaultPackages, ...sessionMetadata.packages])]
      : defaultPackages;

    const installErrors: string[] = []

    const installedPackages = await (prepare_env as {
      install_imports: (
        code: string,
        packages: string[],
        callback: (event: string, data: string) => void
      ) => Promise<unknown[]>;
    }).install_imports(
      pythonCode,
      additionalPackagesToInstall,
      (event_type: string, data: string) => {
        if (event_type === "failed") {
          installErrors.push(data)
        }
      }
    );

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
      await (prepare_env as { load_session_bytes: (data: Uint8Array) => Promise<void> })
        .load_session_bytes(sessionData);
    }

    const packages = installedPackages.map((pkg: unknown) => 
      (pkg as { get?: (key: string) => string }).get?.("package") as string
    );

    // Restore the original console.log function
    console.log = originalLog;
    
    // Run the Python code
    const rawValue = await (pyodide as { runPythonAsync: (code: string) => Promise<unknown> }).runPythonAsync(pythonCode);
    // Dump result to string
    const jsonValue = await (prepare_env as { dumps: (value: unknown) => Promise<string> })
      .dumps(rawValue);

    // Update session metadata with installed packages
    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      // Save session state to sessionBytes
      sessionData = await (prepare_env as { dump_session_bytes: () => Promise<Uint8Array> })
        .dump_session_bytes();
    }

    // Process stdout - join array to string for consistent handling
    const stdoutString = output.join('\n');
    
    // Return the result with stdout and stderr output
    const result: PyodideResult = {
      success: true, 
      result: rawValue,
      jsonResult: jsonValue,
      stdout: stdoutString ? [stdoutString] : [],
      stderr: err_output,
      sessionMetadata: sessionMetadata,
    };

    if (options.stateful && sessionData) {
      result["sessionBytes"] = sessionData;
    }

    // Add filesystem info
    result["fileSystemOperations"] = fileSystemResults;
    result["fileSystemInfo"] = { 
      type: "memfs",
      mountPoint: "/sandbox",
      workingDirectory: "", 
      mounted: true
    };

    return result;
  } catch (error: unknown) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : String(error),
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
  -h, --help                     Display help
  -V, --version                  Display version
`);     
    return;
  }

  if (flags.version) {
    console.log(pkgVersion)
    return
  }

  // Get Python code from file or command line argument
  let pythonCode = "";

  if (flags.file) {
    try {
      // Resolve relative or absolute file path
      const filePath = flags.file.startsWith("/")
        ? flags.file
        : join(Deno.cwd(), flags.file);
      pythonCode = await Deno.readTextFile(filePath);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`Error reading file ${flags.file}:`, errorMessage);
      Deno.exit(1);
    }
  } else {
    // Process code from command line (replacing escaped newlines)
    pythonCode = flags.code?.replace(/\\n/g, "\n") ?? "";
  }

  if (!pythonCode) {
    console.error(
      "Error: You must provide Python code using either -c/--code or -f/--file option.\nUse --help for usage information."
    );
    Deno.exit(1);
  }
  
  // Run the code
  const result = await runPython(pythonCode, {
    stateful: flags.stateful,
    sessionBytes: flags["session-bytes"],
    sessionMetadata: flags["session-metadata"],
  });

  // Create output JSON with stdout, stderr, and result
  const outputJson: Record<string, unknown> = {
    stdout: result.stdout?.join('\n') || "",
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

  // Output as JSON to stdout
  console.log(JSON.stringify(outputJson));

  // Exit with error code if Python execution failed
  if (!result.success) {
    Deno.exit(1);
  }
}

// If this module is run directly
if (import.meta.main) {
  main().catch((err) => {
    console.error("Unhandled error:", err);
    Deno.exit(1);
  });
}

export { runPython, resolvePathInSandbox, type FileSystemOperation, type FileSystemOptions };