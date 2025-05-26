import { assert, assertEquals, assertNotEquals, assertExists } from "@std/assert";
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


Deno.test("filesystem - basic operations", async () => {
  const operations: FileSystemOperation[] = [
    { operation: "mkdir", path: "/sandbox/test", encoding: "utf-8" },
    { operation: "write", path: "/sandbox/test/hello.txt", content: "Hello, MEMFS!", encoding: "utf-8" },
    { operation: "read", path: "/sandbox/test/hello.txt", encoding: "utf-8" },
    { operation: "list", path: "/sandbox/test", encoding: "utf-8" }
  ];

  const result = await runPython(`
import os

# Test file and directory operations
if os.path.exists("/sandbox/test/hello.txt"):
    with open("/sandbox/test/hello.txt", "r") as f:
        content = f.read()

# Return file info
{
  "file_exists": os.path.exists("/sandbox/test/hello.txt"),
  "dir_exists": os.path.exists("/sandbox/test"),
  "content": content if 'content' in locals() else None
}
  `, {
    fileSystemOptions: { enableFileSystem: true },
    fileSystemOperations: operations
  });

  assertEquals(result.success, true);
  assertEquals(result.fileSystemOperations?.length, 4);
  
  // Check operations results
  assertEquals(result.fileSystemOperations?.[0].success, true); // mkdir
  assertEquals(result.fileSystemOperations?.[1].success, true); // write
  assertEquals(result.fileSystemOperations?.[2].success, true); // read
  assertEquals(result.fileSystemOperations?.[2].content, "Hello, MEMFS!"); // read content
  
  // Check list operation
  const listResult = result.fileSystemOperations?.[3];
  assertEquals(listResult.success, true);
  assertEquals(listResult.items.length, 1);
  assertEquals(listResult.items[0].name, "hello.txt");
  
  // Verify Python code could access the files
  const resultData = JSON.parse(result.jsonResult || "null");
  assertEquals(resultData.file_exists, true);
  assertEquals(resultData.dir_exists, true);
  assertEquals(resultData.content, "Hello, MEMFS!");
});

Deno.test("filesystem - document store creation", async () => {
  const result = await runPython(`
import os
from prepare_env import create_document_store

store_result = create_document_store("/sandbox/docs")

# Verify structure was created
expected_dirs = ["raw", "processed", "embeddings", "metadata"]
all_exist = True
for dir_name in expected_dirs:
    path = f"/sandbox/docs/{dir_name}"
    if not os.path.exists(path):
        all_exist = False
        break

{
    "success": store_result["success"],
    "base_path": store_result["base_path"],
    "dirs_exist": all_exist,
    "index_exists": os.path.exists("/sandbox/docs/index.json")
}
  `, {
    fileSystemOptions: { enableFileSystem: true }
  });

  assertEquals(result.success, true);
  const storeResult = JSON.parse(result.jsonResult || "null");
  assertEquals(storeResult.success, true);
  assertEquals(storeResult.base_path, "/sandbox/docs");
  assertEquals(storeResult.dirs_exist, true);
  assertEquals(storeResult.index_exists, true);
});

Deno.test("filesystem - file manipulation workflow", async () => {
  const setupOps: FileSystemOperation[] = [
    { operation: "mkdir", path: "/sandbox/workflow", encoding: "utf-8" },
    { operation: "write", path: "/sandbox/workflow/doc1.txt", content: "Document one content", encoding: "utf-8" },
    { operation: "write", path: "/sandbox/workflow/doc2.txt", content: "Document two content", encoding: "utf-8" },
  ];

  const result = await runPython(`
import os

processed_dir = "/sandbox/workflow/processed"
os.makedirs(processed_dir, exist_ok=True)

documents = []
for filename in os.listdir("/sandbox/workflow"):
    if filename.endswith('.txt'):
        filepath = os.path.join("/sandbox/workflow", filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        word_count = len(content.split())
        processed_content = content.lower().strip()
        
        processed_path = os.path.join(processed_dir, f"processed_{filename}")
        with open(processed_path, 'w') as f:
            f.write(processed_content)
        
        backup_path = os.path.join("/sandbox/workflow", f"backup_{filename}")
        
        with open(backup_path, 'w') as f:
            f.write(content)
        
        documents.append({
            "source": filename,
            "word_count": word_count,
            "processed_path": processed_path,
            "backup_path": backup_path
        })

{
    "total_documents": len(documents),
    "documents": documents,
    "processed_files": os.listdir(processed_dir)
}
  `, {
    fileSystemOptions: { enableFileSystem: true },
    fileSystemOperations: setupOps
  });

  assertEquals(result.success, true);
  const summary = JSON.parse(result.jsonResult || "null");
  assertEquals(summary.total_documents, 2);
  assertEquals(summary.documents.length, 2);
  assertEquals(summary.processed_files.length, 2);
  
  for (const doc of summary.documents) {
    assert(doc.word_count > 0);
    assert(doc.processed_path.includes("processed_"));
    assert(doc.backup_path.includes("backup_"));
  }
});

Deno.test("filesystem - copy operations", async () => {
  const operations: FileSystemOperation[] = [
    { operation: "mkdir", path: "/sandbox/copy_test", encoding: "utf-8" },
    { operation: "write", path: "/sandbox/copy_test/original.txt", content: "Original content", encoding: "utf-8" },
    { operation: "copy", path: "/sandbox/copy_test/original.txt", destination: "/sandbox/copy_test/copy.txt", encoding: "utf-8" },
    { operation: "read", path: "/sandbox/copy_test/copy.txt", encoding: "utf-8" },
    { operation: "list", path: "/sandbox/copy_test", encoding: "utf-8" }
  ];

  const result = await runPython(`
import os
print("Testing copy operations")
print("Files in copy_test:", os.listdir("/sandbox/copy_test"))

with open("/sandbox/copy_test/original.txt", "r") as f_orig:
    original_content = f_orig.read()
    
with open("/sandbox/copy_test/copy.txt", "r") as f_copy:
    copy_content = f_copy.read()
    
{
    "files": os.listdir("/sandbox/copy_test"),
    "original_content": original_content,
    "copy_content": copy_content,
    "match": original_content == copy_content
}
  `, {
    fileSystemOptions: { enableFileSystem: true },
    fileSystemOperations: operations
  });

  assertEquals(result.success, true);
  assertEquals(result.fileSystemOperations?.length, 5);
  
  // Check copy operation success
  assertEquals(result.fileSystemOperations?.[2].success, true);
  
  // Verify copied file has the same content
  const readResult = result.fileSystemOperations?.[3];
  assertEquals(readResult.success, true);
  assertEquals(readResult.content, "Original content");
  
  // Check verification from Python side
  const resultData = JSON.parse(result.jsonResult || "null");
  assertEquals(resultData.files.length, 2);
  assertEquals(resultData.match, true);
  assertEquals(resultData.original_content, resultData.copy_content);
});
