"""Test pyodide sandbox functionality."""

import os
from pathlib import Path

import pytest

from langchain_sandbox import PyodideSandbox, PyodideSandboxTool, SyncPyodideSandbox

current_dir = Path(__file__).parent


@pytest.fixture
def pyodide_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch PKG_NAME to point to a local deno typescript file."""
    if os.environ.get("RUN_INTEGRATION", "").lower() == "true":
        # Skip this test if running in integration mode
        return
    local_script = str(current_dir / "../../../pyodide-sandbox-js/main.ts")
    monkeypatch.setattr("langchain_sandbox.pyodide.PKG_NAME", local_script)


def get_default_sandbox(stateful: bool = False) -> PyodideSandbox:
    """Get default PyodideSandbox instance for testing."""
    return PyodideSandbox(
        stateful=stateful,
        allow_read=True,
        allow_write=True,
        allow_net=True,
        allow_env=False,
        allow_run=False,
        allow_ffi=False,
    )


def get_default_sync_sandbox(stateful: bool = False) -> SyncPyodideSandbox:
    """Get default SyncPyodideSandbox instance for testing."""
    return SyncPyodideSandbox(
        stateful=stateful,
        allow_read=True,
        allow_write=True,
        allow_net=True,
        allow_env=False,
        allow_run=False,
        allow_ffi=False,
    )


async def test_stdout_sessionless(pyodide_package: None) -> None:
    """Test without a session ID."""
    sandbox = get_default_sandbox()
    # Execute a simple piece of code synchronously
    result = await sandbox.execute("x = 5; print(x); x")
    assert result.status == "success"
    assert result.stdout == "5"
    assert result.result == 5
    assert result.stderr is None
    assert result.session_bytes is None


async def test_session_state_persistence_basic(pyodide_package: None) -> None:
    """Simple test to verify that a session ID is used to persist state.

    We'll assign a variable in one execution and check if it's available in the next.
    """
    sandbox = get_default_sandbox(stateful=True)

    result1 = await sandbox.execute("y = 10; print(y)")
    result2 = await sandbox.execute(
        "print(y)",
        session_bytes=result1.session_bytes,
        session_metadata=result1.session_metadata,
    )

    # Check session state persistence
    assert result1.status == "success", f"Encountered error: {result1.stderr}"
    assert result1.stdout == "10"
    assert result1.result is None
    assert result2.status == "success", f"Encountered error: {result2.stderr}"
    assert result2.stdout == "10"
    assert result1.result is None


async def test_pyodide_sandbox_error_handling(pyodide_package: None) -> None:
    """Test PyodideSandbox error handling."""
    sandbox = get_default_sandbox()

    # Test syntax error
    result = await sandbox.execute("x = 5; y = x +")
    assert result.status == "error"
    assert "SyntaxError" in result.stderr

    # Test undefined variable error
    result = await sandbox.execute("undefined_variable")
    assert result.status == "error"
    assert "NameError" in result.stderr


async def test_pyodide_sandbox_timeout(pyodide_package: None) -> None:
    """Test PyodideSandbox timeout handling."""
    sandbox = get_default_sandbox()

    # Test timeout with infinite loop
    # Using a short timeout to avoid long test runs
    result = await sandbox.execute("while True: pass", timeout_seconds=0.5)
    assert result.status == "error"
    assert "timed out" in result.stderr.lower()


def test_sync_stdout_sessionless(pyodide_package: None) -> None:
    """Test synchronous execution without a session ID."""
    sandbox = get_default_sync_sandbox()
    # Execute a simple piece of code synchronously
    result = sandbox.execute("x = 5; print(x); x")
    assert result.status == "success"
    assert result.stdout == "5"
    assert result.result == 5
    assert result.stderr is None
    assert result.session_bytes is None


def test_sync_session_state_persistence_basic(pyodide_package: None) -> None:
    """Test session state persistence in synchronous mode."""
    sandbox = get_default_sync_sandbox(stateful=True)

    result1 = sandbox.execute("y = 10; print(y)")
    result2 = sandbox.execute(
        "print(y)",
        session_bytes=result1.session_bytes,
        session_metadata=result1.session_metadata,
    )

    # Check session state persistence
    assert result1.status == "success", f"Encountered error: {result1.stderr}"
    assert result1.stdout == "10"
    assert result1.result is None
    assert result2.status == "success", f"Encountered error: {result2.stderr}"
    assert result2.stdout == "10"
    assert result1.result is None


def test_sync_pyodide_sandbox_error_handling(pyodide_package: None) -> None:
    """Test synchronous PyodideSandbox error handling."""
    sandbox = get_default_sync_sandbox()

    # Test syntax error
    result = sandbox.execute("x = 5; y = x +")
    assert result.status == "error"
    assert "SyntaxError" in result.stderr

    # Test undefined variable error
    result = sandbox.execute("undefined_variable")
    assert result.status == "error"
    assert "NameError" in result.stderr


def test_sync_pyodide_sandbox_timeout(pyodide_package: None) -> None:
    """Test synchronous PyodideSandbox timeout handling."""
    sandbox = get_default_sync_sandbox()

    # Test timeout with infinite loop
    # Using a short timeout to avoid long test runs
    result = sandbox.execute("while True: pass", timeout_seconds=0.5)
    assert result.status == "error"
    assert "timed out" in result.stderr.lower()


def test_pyodide_sandbox_tool() -> None:
    """Test synchronous invocation of PyodideSandboxTool."""
    tool = PyodideSandboxTool(stateful=False, allow_net=True)
    result = tool.invoke("x = 5; print(x)")
    assert result == "5"
    result = tool.invoke("x = 5; print(1); print(2)")
    assert result == "12"


def test_pyodide_timeout() -> None:
    """Test synchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(stateful=False, timeout_seconds=0.1, allow_net=True)
    result = tool.invoke("while True: pass")
    assert result == "Error during execution: Execution timed out after 0.1 seconds"


async def test_async_pyodide_sandbox_tool() -> None:
    """Test synchronous invocation of PyodideSandboxTool."""
    tool = PyodideSandboxTool(stateful=False, allow_net=True)
    result = await tool.ainvoke("x = 5; print(x)")
    assert result == "5"
    result = await tool.ainvoke("x = 5; print(1); print(2)")
    # TODO: Need to preserve newlines in the output # noqa: FIX002, TD002
    # https://github.com/langchain-ai/langchain-sandbox/issues/26
    assert result == "12"


async def test_async_pyodide_timeout() -> None:
    """Test synchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(stateful=False, timeout_seconds=0.1, allow_net=True)
    result = await tool.ainvoke("while True: pass")
    assert result == "Error during execution: Execution timed out after 0.1 seconds"


async def test_attach_binary_file(pyodide_package: None) -> None:
    """Test attaching and reading a binary file."""
    sandbox = PyodideSandbox(
        allow_read=True,
        allow_write=True,
    )

    simple_binary = bytes([0x01, 0x02, 0x03, 0x04, 0x05])

    sandbox.attach_file("test_binary.bin", simple_binary)

    code = """
import os
import base64

file_path = "/sandbox/test_binary.bin"
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        content = f.read()

    print(f"File exists: True")
    print(f"Content length: {len(content)}")
    print(f"Content bytes: {', '.join(str(b) for b in content)}")
else:
    print("File exists: False")
"""

    result = await sandbox.execute(code)

    assert result.status == "success", f"Error in execution: {result.stderr}"
    assert "File exists: True" in result.stdout
    assert "Content length: 5" in result.stdout
    assert "Content bytes: 1, 2, 3, 4, 5" in result.stdout


async def test_clear_files_after_execution(pyodide_package: None) -> None:
    """Test clearing files after execution."""
    sandbox = get_default_sandbox()

    sandbox.attach_file("temp.txt", "Temporary content")

    result1 = await sandbox.execute(
        'print(open("/sandbox/temp.txt").read())',
        clear_files=True
    )
    assert result1.status == "success"
    assert "Temporary content" in result1.stdout

    assert len(sandbox.file_operations) == 0

    result2 = await sandbox.execute("""
import os
if os.path.exists("/sandbox/temp.txt"):
    print("File still exists")
else:
    print("File is gone")
""")
    assert result2.status == "success"
    assert "File is gone" in result2.stdout


async def test_tool_with_file_attachment(pyodide_package: None) -> None:
    """Test using PyodideSandboxTool with file attachment."""
    tool = PyodideSandboxTool(allow_read=True, allow_write=True, allow_net=True)

    tool.attach_file("data.csv", "id,value\n1,100\n2,200\n3,300")
    tool.attach_file("config.json", '{"max_value": 250, "min_value": 50}')

    code = """
import csv
import json

with open("/sandbox/data.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open("/sandbox/config.json", "r") as f:
    config = json.load(f)

# Filter data based on config
filtered = []
for row in rows:
    value = int(row["value"])
    if config["min_value"] <= value <= config["max_value"]:
        filtered.append(row)

print(f"Filtered data:")
for row in filtered:
    print(f"id: {row['id']}, value: {row['value']}")
"""

    result = await tool.ainvoke(code)

    assert "Filtered data:" in result
    assert "id: 1, value: 100" in result
    assert "id: 2, value: 200" in result
    # Value 300 should be excluded by filter
    assert "id: 3, value: 300" not in result


async def test_directory_operations(pyodide_package: None) -> None:
    """Test directory creation and file operations within directories."""
    sandbox = get_default_sandbox()

    sandbox.attach_file("nested/dir/file.txt", "Content in nested directory")

    code = """
import os
from pathlib import Path

dir_exists = os.path.isdir("/sandbox/nested/dir")
file_exists = os.path.exists("/sandbox/nested/dir/file.txt")
content = Path("/sandbox/nested/dir/file.txt").read_text() if file_exists else ""

print(f"Directory exists: {dir_exists}")
print(f"File exists: {file_exists}")
print(f"Content: {content}")
"""

    result = await sandbox.execute(code)
    assert result.status == "success"
    assert "Directory exists: True" in result.stdout
    assert "File exists: True" in result.stdout
    assert "Content: Content in nested directory" in result.stdout


def test_sync_file_operations(pyodide_package: None) -> None:
    """Test synchronous file operations."""
    sandbox = get_default_sync_sandbox()

    sandbox.attach_files({
        "data.txt": "Text file content",
        "config.json": '{"enabled": true}'
    })

    code = """
import json
from pathlib import Path

text_content = Path("/sandbox/data.txt").read_text()
json_content = json.loads(Path("/sandbox/config.json").read_text())

print(f"Text content: {text_content}")
print(f"JSON enabled: {json_content['enabled']}")
"""

    result = sandbox.execute(code)
    assert result.status == "success"
    assert "Text content: Text file content" in result.stdout
    assert "JSON enabled: True" in result.stdout


async def test_attach_files_with_explicit_binary_flag(pyodide_package: None) -> None:
    """Test attaching files with explicit binary flag in dictionary format."""
    sandbox = get_default_sandbox()

    text_content = "Hello world"
    binary_content = b"\x00\x01\x02\x03"

    sandbox.attach_files({
        "text_file.txt": {"content": text_content, "binary": False},
        "binary_file.bin": {"content": binary_content, "binary": True}
    })

    code = """
from pathlib import Path
import os

# Check text file
text_path = "/sandbox/text_file.txt"
if os.path.exists(text_path):
    with open(text_path, "r") as f:
        text_content = f.read()
    print(f"Text content: {text_content}")

# Check binary file
bin_path = "/sandbox/binary_file.bin"
if os.path.exists(bin_path):
    with open(bin_path, "rb") as f:
        bin_content = f.read()
    print(f"Binary exists: True")
    print(f"Binary length: {len(bin_content)}")
    print(f"Binary bytes: {', '.join(str(b) for b in bin_content)}")
"""

    result = await sandbox.execute(code)
    assert result.status == "success"
    assert "Text content: Hello world" in result.stdout
    assert "Binary exists: True" in result.stdout
    assert "Binary length: 4" in result.stdout
    assert "Binary bytes: 0, 1, 2, 3" in result.stdout
