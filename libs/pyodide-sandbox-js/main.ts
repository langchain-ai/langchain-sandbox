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

class FileSystemOperation(TypedDict):
    operation: Literal["read", "write", "list", "mkdir", "exists", "remove", "copy"]
    path: str
    content: Union[str, bytes, None]
    encoding: str
    destination: Union[str, None]
    
def perform_fs_operation(op) -> dict:
    """Performs filesystem operations safely within the sandbox environment.
    
    Supports the following operations:
      - read: Reads file content with text or binary encoding
      - write: Writes content to a file, creating parent directories if needed
      - list: Lists directory contents with metadata (name, type, size, etc)
      - mkdir: Creates directories recursively
      - exists: Checks if a file or directory exists
      - remove: Deletes files or directories (recursive)
      - copy: Copies files or directories to a destination path
      
    Returns:
      A dictionary with operation result ('success' boolean and data or 'error' message)
    """
    try:
        # Convert JsProxy to Python dict if needed
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
                return {"success": False, "error": "File not found"}
                
        elif operation == "write":
            # Ensure parent directory exists
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
            return {"success": True}
            
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
                return {"success": False, "error": "Directory not found"}
                
        elif operation == "mkdir":
            os.makedirs(path, exist_ok=True)
            return {"success": True}
            
        elif operation == "exists":
            return {"success": True, "exists": os.path.exists(path)}
            
        elif operation == "remove":
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                return {"success": True}
            else:
                return {"success": False, "error": "Path not found"}
                
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
                return {"success": False, "error": "Source path not found"}
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_document_store(base_path: str = "/sandbox/documents") -> dict:
    """Creates a document store structure for LangChain.
    
    Sets up a directory structure suitable for LangChain document processing:
      - raw: For storing original documents
      - processed: For storing processed documents
      - embeddings: For storing document embeddings
      - metadata: For storing document metadata
      
    Also creates an index.json file to track documents and collections.
    
    Args:
      base_path: Root directory for the document store
      
    Returns:
      Dictionary with creation status and structure information
    """
    try:
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/raw", exist_ok=True)
        os.makedirs(f"{base_path}/processed", exist_ok=True)
        os.makedirs(f"{base_path}/embeddings", exist_ok=True)
        os.makedirs(f"{base_path}/metadata", exist_ok=True)
        
        # Create index file
        index_file = f"{base_path}/index.json"
        initial_index = {
            "created": datetime.datetime.now().isoformat(),
            "version": "1.0",
            "documents": {},
            "collections": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        with open(index_file, 'w') as f:
            json.dump(initial_index, f, indent=2)
            
        return {
            "success": True,
            "base_path": base_path,
            "structure": ["raw", "processed", "embeddings", "metadata"],
            "index_file": index_file
        }
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
    message_callback: Callable = lambda event_type, data: None,
) -> List[InstallEntry]:
    """Installs Python packages required for the provided code or import list.
    
    Takes either:
      - Python source code: Analyzes imports using Pyodide's find_imports
      - A list of import names: Uses the list directly
      
    Additionally installs any packages specified in additional_packages.
    
    Args:
      source_code_or_imports: Python code string or list of import names
      additional_packages: Extra packages to install regardless of imports
      message_callback: Function called with status updates during installation
      
    Returns:
      List of package entries that were installed
    """
    if isinstance(source_code_or_imports, str):
        try:
            imports: list[str] = find_imports(source_code_or_imports)
        except SyntaxError:
            return []
    else:
        imports: list[str] = source_code_or_imports

    to_install = find_imports_to_install(imports)
    for package in additional_packages:
        if package not in [entry["package"] for entry in to_install]:
            to_install.append(dict(module=package, package=package))

    if to_install:
        try:
            import micropip
        except ModuleNotFoundError:
            await pyodide_js.loadPackage("micropip")
            import micropip

        for entry in to_install:
            try:
                await micropip.install(entry["package"])
            except Exception as e:
                message_callback("failed", entry["package"])
                break
    return to_install

def load_session_bytes(session_bytes: bytes):
    """Loads a serialized session state from bytes.
    
    Uses dill to restore a previously serialized Python session state,
    including all variables, functions and class definitions.
    
    Args:
      session_bytes: Bytes object containing the serialized session
    """
    import dill
    import io
    buffer = io.BytesIO(session_bytes.to_py())
    dill.session.load_session(filename=buffer)

def dump_session_bytes() -> bytes:
    """Serializes the current session state to bytes.
    
    Uses dill to capture the current Python session state,
    including all variables, functions and class definitions.
    
    Returns:
      Bytes object containing the serialized session
    """
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
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [robust_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): robust_serialize(value) for key, value in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [robust_serialize(item) for item in obj]
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return {"type": "not_serializable", "repr": repr(obj)}

def dumps(result: Any) -> str:
    """Serializes a Python object to a JSON string.
    
    Uses robust_serialize to handle complex Python objects before JSON serialization.
    """
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
  mountPoint?: string;
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
  };
}

interface FileSystemOperation {
  operation: "read" | "write" | "list" | "mkdir" | "exists" | "remove" | "copy";
  path: string;
  content?: string | Uint8Array;
  encoding?: string;
  destination?: string;
}

interface ExecutionResult {
  success: boolean;
  stdout: string | null;
  stderr: string | null;
  result: any;
  session?: {
    metadata?: SessionMetadata;
    bytes?: Uint8Array;
  };
  filesystem?: {
    info?: {
      type: string;
      mountPoint: string;
    };
    operations?: any[];
  };
}

async function initPyodide(pyodide: any, options: FileSystemOptions = {}): Promise<void> {
  const sys = pyodide.pyimport("sys");
  const pathlib = pyodide.pyimport("pathlib");

  const dirPath = "/tmp/pyodide_worker_runner/";
  sys.path.append(dirPath);
  pathlib.Path(dirPath).mkdir();
  pathlib.Path(dirPath + "prepare_env.py").write_text(prepareEnvCode);

  if (options.enableFileSystem) {
    const mountPoint = options.mountPoint || "/sandbox";
    
    try {
      pyodide.FS.mkdirTree(mountPoint);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (!errorMessage.includes("exists")) {
        console.warn(`⚠️ Failed to create mount point ${mountPoint}:`, error);
      }
    }
  }
}

async function performFileSystemOperations(
  pyodide: any,
  operations: FileSystemOperation[]
): Promise<any[]> {
  const results: any[] = [];
  const prepare_env = pyodide.pyimport("prepare_env");

  for (const op of operations) {
    try {
      const result = prepare_env.perform_fs_operation(op);
      const jsResult = result.toJs({ dict_converter: Object.fromEntries });
      results.push(jsResult);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      results.push({ success: false, error: errorMessage });
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
    })
    await pyodide.loadPackage(["micropip"], {
      messageCallback: () => {},
      errorCallback: (msg: string) => {
        output.push(`install error: ${msg}`)
      },
    });
    
    await initPyodide(pyodide, options.fileSystemOptions);

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
    const prepare_env = pyodide.pyimport("prepare_env");
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
        error: `Failed to install required Python packages: ${installErrors.join(", ")}.`
      };
    }

    if (options.sessionBytes) {
      sessionData = Uint8Array.from(JSON.parse(options.sessionBytes));
      await prepare_env.load_session_bytes(sessionData);
    }

    let fileSystemResults: any[] = [];
    if (options.fileSystemOperations) {
      fileSystemResults = await performFileSystemOperations(pyodide, options.fileSystemOperations);
    }

    const packages = installedPackages.map((pkg: any) => pkg.get("package"));

    console.log = originalLog;
    
    const rawValue = await pyodide.runPythonAsync(pythonCode);
    
    const jsonValue = await prepare_env.dumps(rawValue);

    sessionMetadata.packages = [
      ...new Set([...sessionMetadata.packages, ...packages]),
    ];
    sessionMetadata.lastModified = new Date().toISOString();

    if (options.stateful) {
      sessionData = await prepare_env.dump_session_bytes() as Uint8Array;
    }
    
    const result: PyodideResult = {
      success: true, 
      result: rawValue,
      jsonResult: jsonValue,
      stdout: output,
      stderr: err_output,
      sessionMetadata: sessionMetadata,
      fileSystemOperations: fileSystemResults,
    };
    
    if (options.stateful && sessionData) {
      result["sessionBytes"] = sessionData;
    }

    if (options.fileSystemOptions?.enableFileSystem) {
      result["fileSystemInfo"] = {
        type: "memfs",
        mountPoint: options.fileSystemOptions.mountPoint || "/sandbox",
      };
    }
    
    return result;
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return { 
      success: false, 
      error: errorMessage,
      stdout: output,
      stderr: err_output
    };
  }
}

async function main(): Promise<void> {
  const flags = parseArgs(Deno.args, {
    string: ["code", "file", "session-bytes", "session-metadata", "fs-operations", "mount-point"],
    alias: {
      c: "code",
      f: "file",
      h: "help",
      V: "version",
      s: "stateful",
      b: "session-bytes",
      m: "session-metadata",
      "fs": "fs-operations",
      "mp": "mount-point",
    },
    boolean: ["help", "version", "stateful", "enable-filesystem"],
    default: { 
      help: false, 
      version: false, 
      stateful: false, 
      "enable-filesystem": false,
      "mount-point": "/sandbox"
    },
  });

  if (flags.help) {
    console.log(`
pyodide-sandbox ${pkgVersion}
Run Python code in a sandboxed environment using Pyodide

OPTIONS:
  -c, --code <code>            Python code to execute
  -f, --file <path>            Path to Python file to execute
  -s, --stateful               Use a stateful session
  -b, --session-bytes <bytes>  Session bytes
  -m, --session-metadata       Session metadata
  --enable-filesystem          Enable filesystem operations (MEMFS)
  --fs-operations <json>       JSON array of filesystem operations
  --mount-point <path>         Mount point path (default: /sandbox)
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
    enableFileSystem: flags["enable-filesystem"],
    fsOperations: flags["fs-operations"],
    mountPoint: flags["mount-point"],
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

  const result = await runPython(pythonCode, {
    stateful: options.stateful,
    sessionBytes: options.sessionBytes,
    sessionMetadata: options.sessionMetadata,
    fileSystemOptions: {
      enableFileSystem: options.enableFileSystem,
      mountPoint: options.mountPoint,
    },
    fileSystemOperations: fileSystemOperations,
  });

  const executionResult: ExecutionResult = {
    success: result.success,
    stdout: result.stdout?.join('') || null,
    stderr: result.success ? (result.stderr?.join('') || null) : result.error || null,
    result: result.success ? JSON.parse(result.jsonResult || 'null') : null,
  };

  // Only include session data if stateful execution was used
  if (result.sessionBytes || result.sessionMetadata) {
    executionResult.session = {
      metadata: result.sessionMetadata,
      ...(result.sessionBytes ? { bytes: result.sessionBytes } : {})
    };
  }

  // Only include filesystem information if filesystem was enabled
  if (options.enableFileSystem) {
    executionResult.filesystem = {
      info: result.fileSystemInfo,
      // Only include operations results if operations were actually performed
      ...(fileSystemOperations.length > 0 ? { operations: result.fileSystemOperations } : {})
    };
  }

  console.log(JSON.stringify(executionResult));

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

export { runPython, type FileSystemOperation, type FileSystemOptions };