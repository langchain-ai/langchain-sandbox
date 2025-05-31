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


@pytest.fixture
def mock_csv_data() -> str:
    """Sample sales data for testing."""
    return """date,product_id,category,quantity,price,customer_id,region
2024-01-15,P001,Electronics,2,499.99,C123,North
2024-01-16,P002,Furniture,1,899.50,C124,South
2024-01-16,P003,Clothing,5,59.99,C125,East
2024-01-17,P001,Electronics,1,499.99,C126,West
2024-01-18,P004,Electronics,3,299.99,C127,North"""


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


def test_pyodide_sandbox_tool() -> None:
    """Test synchronous invocation of PyodideSandboxTool."""
    tool = PyodideSandboxTool(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
    )
    result = tool.invoke({"code": "x = 5; print(x)"})
    assert result == "5"
    result = tool.invoke({"code": "x = 5; print(1); print(2)"})
    assert result == "1\n2"


def test_pyodide_timeout() -> None:
    """Test synchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
        timeout_seconds=0.1,
    )
    result = tool.invoke({"code": "while True: pass"})
    assert "timed out after 0.1 seconds" in result


async def test_async_pyodide_sandbox_tool() -> None:
    """Test asynchronous invocation of PyodideSandboxTool."""
    tool = PyodideSandboxTool(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
    )
    result = await tool.ainvoke({"code": "x = 5; print(x)"})
    assert result == "5"
    result = await tool.ainvoke({"code": "x = 5; print(1); print(2)"})
    assert result == "1\n2"


async def test_async_pyodide_timeout() -> None:
    """Test asynchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
        timeout_seconds=0.1,
    )
    result = await tool.ainvoke({"code": "while True: pass"})
    assert "timed out after 0.1 seconds" in result


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


async def test_filesystem_basic_operations(pyodide_package: None) -> None:
    """Test basic filesystem operations."""
    sandbox = PyodideSandbox(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
    )

    # Attach files
    sandbox.attach_file("test.txt", "Hello, World!")
    sandbox.attach_file("data.json", '{"key": "value"}')
    sandbox.create_directory("output")

    code = """
import os
import json

# Read files
with open("test.txt", "r") as f:
    txt_content = f.read()

with open("data.json", "r") as f:
    json_data = json.load(f)

# Create new file
with open("output/result.txt", "w") as f:
    f.write("Processing complete!")

# List files
root_files = sorted(os.listdir("."))
output_files = sorted(os.listdir("output"))

print(f"Text: {txt_content}")
print(f"JSON key: {json_data['key']}")
print(f"Root files: {root_files}")
print(f"Output files: {output_files}")

# Read the created file to verify it was written
with open("output/result.txt", "r") as f:
    created_content = f.read()
print(f"Created file content: {created_content}")
"""

    result = await sandbox.execute(code)
    assert result.status == "success", f"Execution failed: {result.stderr}"
    assert "Hello, World!" in result.stdout
    assert "value" in result.stdout
    assert "Processing complete!" in result.stdout


def test_filesystem_tool_usage() -> None:
    """Test filesystem with PyodideSandboxTool."""
    tool = PyodideSandboxTool(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
    )

    # Attach CSV data
    csv_data = "name,age\nAlice,30\nBob,25"
    tool.attach_file("users.csv", csv_data)

    code = """
import csv

users = []
with open("users.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        users.append(row)

for user in users:
    print(f"{user['name']} is {user['age']} years old")
"""

    result = tool.invoke({"code": code})
    assert "Alice is 30 years old" in result
    assert "Bob is 25 years old" in result


async def test_binary_file_operations(pyodide_package: None) -> None:
    """Test binary file operations."""
    sandbox = PyodideSandbox(
        enable_filesystem=True,
        allow_net=True,
        allow_read=True,
        allow_write=True,
    )

    # Create some binary data
    binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    sandbox.attach_binary_file("image.png", binary_data)

    code = """
import base64

# Read binary file
with open("image.png", "rb") as f:
    data = f.read()

# Check if it's the PNG header
is_png = data.startswith(b'\\x89PNG')
size = len(data)

print(f"Is PNG: {is_png}")
print(f"Size: {size} bytes")
print(f"Original size: {len(data)}")  # Debug
"""

    result = await sandbox.execute(code)
    assert result.status == "success", f"Execution failed: {result.stderr}"
    assert "Is PNG: True" in result.stdout
    # Verify the size matches the binary data size
    assert f"Size: {len(binary_data)} bytes" in result.stdout
