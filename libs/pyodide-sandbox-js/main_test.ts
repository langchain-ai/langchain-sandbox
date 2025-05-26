import { assertEquals, assertNotEquals, assertExists } from "@std/assert";
import { runPython, type FileSystemOperation } from "./main.ts";

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
  // Check that error contains SyntaxError
  assertEquals(result.error?.includes("SyntaxError"), true);
});

Deno.test("runPython with error - name error", async () => {
  const result = await runPython("undefined_variable", {});
  assertEquals(result.success, false);
  assertExists(result.error);
  // Check that error contains NameError
  assertEquals(result.error?.includes("NameError"), true);
});

Deno.test("filesystem - write and read text file", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "write",
      path: "/sandbox/test.txt",
      content: "Hello, World!",
    }
  ];
  
  const result = await runPython(`
with open("/sandbox/test.txt", "r") as f:
    content = f.read()
content
  `, {
    fileSystemOperations: operations
  });
  
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), "Hello, World!");
});

Deno.test("filesystem - directory operations", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "mkdir",
      path: "/sandbox/testdir",
    },
    {
      operation: "write",
      path: "/sandbox/testdir/file.txt",
      content: "File in directory",
    }
  ];
  
  const result = await runPython(`
import os
dir_exists = os.path.isdir("/sandbox/testdir")
file_path = "/sandbox/testdir/file.txt"
file_exists = os.path.exists(file_path)
content = open(file_path).read() if file_exists else ""
{"dir_exists": dir_exists, "file_exists": file_exists, "content": content}
  `, {
    fileSystemOperations: operations
  });
  
  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  assertEquals(resultObj.dir_exists, true);
  assertEquals(resultObj.file_exists, true);
  assertEquals(resultObj.content, "File in directory");
});

Deno.test("filesystem - list directory contents", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "mkdir",
      path: "/sandbox/listdir",
    },
    {
      operation: "write",
      path: "/sandbox/listdir/file1.txt",
      content: "File 1",
    },
    {
      operation: "write",
      path: "/sandbox/listdir/file2.txt",
      content: "File 2",
    }
  ];
  
  const result = await runPython(`
import os
files = os.listdir("/sandbox/listdir")
sorted(files)
  `, {
    fileSystemOperations: operations
  });
  
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), ["file1.txt", "file2.txt"]);
});

Deno.test("filesystem - custom mount point", async () => {
  const operations: FileSystemOperation[] = [
    {
      operation: "write",
      path: "/customdir/test.txt",
      content: "Custom mount point",
    }
  ];
  
  const result = await runPython(`
import os
path = "/customdir/test.txt"
exists = os.path.exists(path)
content = open(path).read() if exists else ""
{"exists": exists, "content": content}
  `, {
    fileSystemOptions: { mountPoint: "/customdir" },
    fileSystemOperations: operations
  });
  
  assertEquals(result.success, true);
  const resultObj = JSON.parse(result.jsonResult || "null");
  assertEquals(resultObj.exists, true);
  assertEquals(resultObj.content, "Custom mount point");
});

Deno.test("filesystem - binary file operations with explicit encoding", async () => {
  // Create binary data as base64 string
  const binaryContent = "QmluYXJ5IGRhdGE="; // base64 for "Binary data"
  
  const operations: FileSystemOperation[] = [
    {
      operation: "write",
      path: "/sandbox/explicit.bin",
      content: binaryContent,
      encoding: "binary"  // Explicitly set binary encoding
    }
  ];
  
  const result = await runPython(`
with open("/sandbox/explicit.bin", "rb") as f:
    content = f.read()
content.decode('utf-8')  # Should be "Binary data"
  `, {
    fileSystemOperations: operations
  });
  
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), "Binary data");
});
