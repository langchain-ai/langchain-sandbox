import { assertEquals, assertNotEquals } from "@std/assert";
import { runPython, resolvePathInSandbox, type FileSystemOperation } from "./main.ts";

Deno.test("runPython simple test", async () => {
  const result = await runPython("x = 2 + 3; x", {});
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
});

Deno.test("runPython with stdout", async () => {
  const result = await runPython("x = 5; print(x); x", {});
  assertEquals(result.success, true);
  assertEquals(result.stdout?.join(''), "5");
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

Deno.test("FileSystem - operations", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "write",
      path: "config.json",
      content: '{"app": "test", "version": "1.0"}',
    },
    {
      operation: "mkdir",
      path: "data",
    },
    {
      operation: "write",
      path: "data/output.txt",
      content: "Hello World\nLine 2",
    }
  ];

  const result = await runPython(`
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
  `, {
    fileSystemOperations: operations
  });

  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  
  assertEquals(resultObj.config.app, "test");
  assertEquals(resultObj.content, "Hello World\nLine 2");
  assertEquals(resultObj.root_files, ["config.json", "data"]);
  assertEquals(resultObj.data_files, ["output.txt"]);
  assertEquals(resultObj.working_dir, "/sandbox");
});

Deno.test("FileSystem - binary operations", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "write",
      path: "test.bin",
      content: "QmluYXJ5IGRhdGE=", // Base64 for "Binary data"
      encoding: "binary"
    }
  ];

  const result = await runPython(`
import os
import base64

# Read binary file
with open("test.bin", "rb") as f:
    binary_content = f.read()

# Decode content
try:
    decoded = binary_content.decode('utf-8')
except UnicodeDecodeError:
    decoded = base64.b64decode(binary_content).decode('utf-8')

result = {
    "file_exists": os.path.exists("test.bin"),
    "decoded_content": decoded,
    "is_binary_match": decoded == "Binary data",
    "working_dir": os.getcwd()
}

result
  `, {
    fileSystemOperations: operations
  });

  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  assertEquals(resultObj.file_exists, true);
  assertEquals(resultObj.decoded_content, "Binary data");
  assertEquals(resultObj.is_binary_match, true);
  assertEquals(resultObj.working_dir, "/sandbox");
});

Deno.test("FileSystem - memfs directory structure", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "mkdir",
      path: "project",
    },
    {
      operation: "mkdir", 
      path: "project/src",
    },
    {
      operation: "write",
      path: "project/src/main.py",
      content: "print('Hello from memfs!')",
    },
    {
      operation: "write",
      path: "project/README.md",
      content: "# My Project\nRunning in memfs",
    }
  ];

  const result = await runPython(`
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
  `, {
    fileSystemOperations: operations
  });

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