import { assertEquals, assertNotEquals } from "@std/assert";
import { runPython, resolvePathInSandbox } from "./main.ts";

Deno.test("runPython simple test", async () => {
  const result = await runPython("x = 2 + 3; x", {});
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
});

Deno.test("runPython with stdout", async () => {
  const result = await runPython("x = 5; print(x); x", {});
  assertEquals(result.success, true);
  assertEquals(result.stdout?.join('').trim(), "5");
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
  assertEquals(result.stderr?.length, 0);
});

Deno.test("runPython with error - division by zero", async () => {
  const result = await runPython("x = 1/0", {});
  assertEquals(result.success, false);
  assertNotEquals(result.error?.length, 0);
  assertEquals(result.error?.includes("ZeroDivisionError"), true);
});

Deno.test("runPython with error - syntax error", async () => {
  const result = await runPython("x = 5; y = x +", {});
  assertEquals(result.success, false);
  assertNotEquals(result.error?.length, 0);
  assertEquals(result.error?.includes("SyntaxError"), true);
});

Deno.test("resolvePathInSandbox - basic resolution", () => {
  assertEquals(resolvePathInSandbox("config.json"), "/sandbox/config.json");
  assertEquals(resolvePathInSandbox("./logs/app.log"), "/sandbox/logs/app.log");
  assertEquals(resolvePathInSandbox("../shared/data.txt"), "/sandbox/../shared/data.txt");
  assertEquals(resolvePathInSandbox("/tmp/absolute.txt"), "/tmp/absolute.txt");
});

// Helper function to create stdin data for filesystem operations
function createFilesystemStdin(
  files: Array<{ path: string; content: string | Uint8Array; binary?: boolean }>,
  directories: string[] = []
): Uint8Array {
  // Convert files to the expected format
  const fileInfos = files.map(f => {
    const contentBytes = typeof f.content === 'string' 
      ? new TextEncoder().encode(f.content)
      : f.content;
    
    return {
      path: f.path,
      size: contentBytes.length,
      binary: f.binary || false,
      content: contentBytes
    };
  });

  // Create metadata
  const metadata = {
    files: fileInfos.map(f => ({
      path: f.path,
      size: f.size,
      binary: f.binary
    })),
    directories: directories
  };

  const metadataJson = new TextEncoder().encode(JSON.stringify(metadata));
  
  // Create header: "PSB" + version + metadata size (4 bytes)
  const header = new Uint8Array(8);
  header.set(new TextEncoder().encode("PSB"), 0);
  header[3] = 1; // version
  
  // Set metadata length (big endian)
  const dataView = new DataView(header.buffer);
  dataView.setUint32(4, metadataJson.length, false);

  // Combine header + metadata + file contents
  const totalSize = header.length + metadataJson.length + 
    fileInfos.reduce((sum, f) => sum + f.content.length, 0);
  
  const result = new Uint8Array(totalSize);
  let offset = 0;
  
  result.set(header, offset);
  offset += header.length;
  
  result.set(metadataJson, offset);
  offset += metadataJson.length;
  
  for (const fileInfo of fileInfos) {
    result.set(fileInfo.content, offset);
    offset += fileInfo.content.length;
  }
  
  return result;
}

// Mock Deno.stdin for filesystem tests
async function runPythonWithFiles(
  code: string,
  files: Array<{ path: string; content: string | Uint8Array; binary?: boolean }> = [],
  directories: string[] = [],
  options: Record<string, unknown> = {}
) {
  if (files.length === 0 && directories.length === 0) {
    return await runPython(code, options);
  }

  // Create the stdin data
  const stdinData = createFilesystemStdin(files, directories);
  
  // Mock stdin for this test
  const originalIsTerminal = Deno.stdin.isTerminal;
  const originalRead = Deno.stdin.read;
  let dataOffset = 0;
  
  // Mock isTerminal to return false (indicating we have stdin data)
  Deno.stdin.isTerminal = () => false;
  
  // Mock stdin.read to return our data
  Deno.stdin.read = (buffer: Uint8Array): Promise<number | null> => {
    if (dataOffset >= stdinData.length) {
      return Promise.resolve(null);
    }
    
    const remaining = stdinData.length - dataOffset;
    const toRead = Math.min(buffer.length, remaining);
    
    buffer.set(stdinData.subarray(dataOffset, dataOffset + toRead));
    dataOffset += toRead;
    
    return Promise.resolve(toRead);
  };
  
  try {
    return await runPython(code, options);
  } finally {
    // Restore original functions
    Deno.stdin.isTerminal = originalIsTerminal;
    Deno.stdin.read = originalRead;
  }
}

Deno.test("FileSystem - operations", async () => {
  const files = [
    {
      path: "config.json",
      content: '{"app": "test", "version": "1.0"}'
    },
    {
      path: "data/output.txt",
      content: "Hello World\nLine 2"
    }
  ];
  
  const directories = ["data"];

  const result = await runPythonWithFiles(`
import os
import json

# Read config file
with open("config.json", "r") as f:
    config = json.load(f)

# Read data file
with open("data/output.txt", "r") as f:
    content = f.read()

# List files
root_files = os.listdir(".")
data_files = os.listdir("data")

result = {
    "config": config,
    "content": content.strip(),
    "root_files": sorted(root_files),
    "data_files": sorted(data_files),
    "working_dir": os.getcwd()
}

result
  `, files, directories);

  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  
  assertEquals(resultObj.config.app, "test");
  assertEquals(resultObj.content, "Hello World\nLine 2");
  assertEquals(resultObj.root_files, ["config.json", "data"]);
  assertEquals(resultObj.data_files, ["output.txt"]);
  assertEquals(resultObj.working_dir, "/sandbox");
});

Deno.test("FileSystem - binary operations", async () => {
  // Create binary content - "Binary data" encoded as bytes
  const binaryContent = new TextEncoder().encode("Binary data");
  
  const files = [
    {
      path: "test.bin",
      content: binaryContent,
      binary: true
    }
  ];

  const result = await runPythonWithFiles(`
import os

# Read binary file
with open("test.bin", "rb") as f:
    binary_content = f.read()

# Decode content
decoded = binary_content.decode('utf-8')

result = {
    "file_exists": os.path.exists("test.bin"),
    "decoded_content": decoded,
    "is_binary_match": decoded == "Binary data",
    "working_dir": os.getcwd()
}

result
  `, files);

  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  assertEquals(resultObj.file_exists, true);
  assertEquals(resultObj.decoded_content, "Binary data");
  assertEquals(resultObj.is_binary_match, true);
  assertEquals(resultObj.working_dir, "/sandbox");
});

Deno.test("FileSystem - memfs directory structure", async () => {
  const files = [
    {
      path: "project/src/main.py",
      content: "print('Hello from memfs!')"
    },
    {
      path: "project/README.md",
      content: "# My Project\nRunning in memfs"
    }
  ];
  
  const directories = ["project", "project/src"];

  const result = await runPythonWithFiles(`
import os

# Navigate and check structure
project_exists = os.path.exists("project")
src_exists = os.path.exists("project/src")
main_py_exists = os.path.exists("project/src/main.py")
readme_exists = os.path.exists("project/README.md")

# Read files
with open("project/src/main.py", "r") as f:
    main_content = f.read()

with open("project/README.md", "r") as f:
    readme_content = f.read()

# List structure
project_files = sorted(os.listdir("project"))
src_files = sorted(os.listdir("project/src"))

result = {
    "project_exists": project_exists,
    "src_exists": src_exists,
    "main_py_exists": main_py_exists,
    "readme_exists": readme_exists,
    "main_content": main_content.strip(),
    "readme_content": readme_content.strip(),
    "project_files": project_files,
    "src_files": src_files,
    "working_dir": os.getcwd()
}

result
  `, files, directories);

  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  
  assertEquals(resultObj.project_exists, true);
  assertEquals(resultObj.src_exists, true);
  assertEquals(resultObj.main_py_exists, true);
  assertEquals(resultObj.readme_exists, true);
  assertEquals(resultObj.main_content, "print('Hello from memfs!')");
  assertEquals(resultObj.readme_content, "# My Project\nRunning in memfs");
  assertEquals(resultObj.project_files, ["README.md", "src"]);
  assertEquals(resultObj.src_files, ["main.py"]);
  assertEquals(resultObj.working_dir, "/sandbox");
});