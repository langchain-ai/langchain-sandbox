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


async def test_stdout_sessionless(pyodide_package: None) -> None:
    """Test without a session ID."""
    sandbox = get_default_sandbox()
    # Execute a simple piece of code asynchronously
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
    assert result2.result is None


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
    assert result2.result is None


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


def test_pyodide_sandbox_tool(pyodide_package: None) -> None:
    """Test synchronous invocation of PyodideSandboxTool."""
    # allow_read=True is required for Deno to access Pyodide WASM files
    tool = PyodideSandboxTool(
        stateful=False,
        allow_net=True,
        allow_read=True,
    )
    result = tool.invoke({"code": "x = 5; print(x)"})
    assert result == "5"
    result = tool.invoke({"code": "x = 5; print(1); print(2)"})
    assert result == "1\n2"


def test_pyodide_timeout() -> None:
    """Test synchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(
        stateful=False,
        allow_net=True,
        timeout_seconds=0.1,
    )
    result = tool.invoke({"code": "while True: pass"})
    assert "timed out after 0.1 seconds" in result


async def test_async_pyodide_sandbox_tool(pyodide_package: None) -> None:
    """Test asynchronous invocation of PyodideSandboxTool."""
    # allow_read=True is required for Deno to access Pyodide WASM files
    tool = PyodideSandboxTool(
        stateful=False,
        allow_net=True,
        allow_read=True,
    )
    result = await tool.ainvoke({"code": "x = 5; print(x)"})
    assert result == "5"
    result = await tool.ainvoke({"code": "x = 5; print(1); print(2)"})
    assert result == "1\n2"


async def test_async_pyodide_timeout() -> None:
    """Test asynchronous invocation of PyodideSandboxTool with timeout."""
    tool = PyodideSandboxTool(
        stateful=False,
        allow_net=True,
        timeout_seconds=0.1,
    )
    result = await tool.ainvoke({"code": "while True: pass"})
    assert "timed out after 0.1 seconds" in result


async def test_filesystem_basic_operations(pyodide_package: None) -> None:
    """Test basic filesystem operations."""
    # allow_read=True is required for Deno to access Pyodide WASM files
    sandbox = PyodideSandbox(
        allow_net=True,
        allow_read=True,
        files={"test.txt": "Hello, World!", "data.json": '{"key": "value"}'},
        directories=["output"],
    )

    code = """
import os
import json

# Read files
with open("test.txt", "r") as f:
    txt_content = f.read()

with open("data.json", "r") as f:
    json_data = json.load(f)

# Create new file in pre-created directory
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


def test_filesystem_tool_usage(pyodide_package: None) -> None:
    """Test filesystem with PyodideSandboxTool."""
    # Attach CSV data using files parameter in constructor
    csv_data = "name,age\nAlice,30\nBob,25"
    tool = PyodideSandboxTool(
        allow_net=True, allow_read=True, files={"users.csv": csv_data}
    )

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
    # Create some binary data
    binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"

    # allow_read=True is required for Deno to access Pyodide WASM files
    sandbox = PyodideSandbox(
        allow_net=True, allow_read=True, files={"image.png": binary_data}
    )

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
"""

    result = await sandbox.execute(code)
    assert result.status == "success", f"Execution failed: {result.stderr}"
    assert "Is PNG: True" in result.stdout
    # Verify the size matches the binary data size
    assert f"Size: {len(binary_data)} bytes" in result.stdout


async def test_large_file_attachment(pyodide_package: None) -> None:
    """Test attaching a large file to the sandbox."""
    # Generate a test file with a simple pattern
    size_mb = 5  # 5MB is sufficient to test streaming
    size_bytes = size_mb * 1024 * 1024

    # Generate test content
    large_data = bytes([i % 256 for i in range(size_bytes)])

    # allow_read=True is required for Deno to access Pyodide WASM files
    sandbox = PyodideSandbox(
        allow_net=True, allow_read=True, files={"large_file.bin": large_data}
    )

    # Verify that the file was attached correctly
    code = """
import os

file_path = "large_file.bin"
exists = os.path.exists(file_path)
size = os.path.getsize(file_path) if exists else 0

print(f"File exists: {exists}")
print(f"File size: {size} bytes")
print("Verification completed successfully!")
"""

    # Execute the code that verifies the file
    result = await sandbox.execute(code)

    assert result.status == "success", f"Failed to verify file: {result.stderr}"
    assert "File exists: True" in result.stdout
    assert f"File size: {size_bytes} bytes" in result.stdout
    assert "Verification completed successfully!" in result.stdout


def test_description_custom_without_files(pyodide_package: None) -> None:
    """Test custom description without files."""
    custom_description = "Use Python to analyze data. No fancy stuff."

    tool = PyodideSandboxTool(allow_net=True, description=custom_description)

    # Verify the custom description is used and doesn't have file info
    assert tool.description == custom_description
    assert "ATTACHED FILES AVAILABLE" not in tool.description


def test_description_custom_with_files(pyodide_package: None) -> None:
    """Test custom description with files."""
    custom_description = "Custom Python sandbox with {available_files}"

    # Create tool with files in constructor
    tool = PyodideSandboxTool(
        allow_net=True,
        description=custom_description,
        files={"data.csv": "a,b\n1,2", "config.json": '{"setting": true}'},
    )

    # Verify description contains both custom text and file info
    assert "Custom Python sandbox with" in tool.description
    assert "ATTACHED FILES AVAILABLE" in tool.description
    assert "data.csv" in tool.description
    assert "config.json" in tool.description


def test_description_default(pyodide_package: None) -> None:
    """Test default description behavior."""
    tool = PyodideSandboxTool(allow_net=True)

    # Check default description
    assert "A secure Python code sandbox with filesystem support" in tool.description
    assert "ATTACHED FILES AVAILABLE" not in tool.description

    # Create a new tool with files to test description update
    tool_with_files = PyodideSandboxTool(
        allow_net=True, files={"test.txt": "Hello world"}
    )

    # Verify description was updated with file info
    assert (
        "A secure Python code sandbox with filesystem support"
        in tool_with_files.description
    )
    assert "ATTACHED FILES AVAILABLE" in tool_with_files.description
    assert "test.txt" in tool_with_files.description


def test_directories_creation(pyodide_package: None) -> None:
    """Test directory creation via constructor."""
    tool = PyodideSandboxTool(
        allow_net=True, allow_read=True, directories=["data", "output", "logs/app"]
    )

    code = """
import os

# Check if directories exist
data_exists = os.path.exists("data") and os.path.isdir("data")
output_exists = os.path.exists("output") and os.path.isdir("output")
logs_exists = os.path.exists("logs") and os.path.isdir("logs")
logs_app_exists = os.path.exists("logs/app") and os.path.isdir("logs/app")

# List root directory
root_items = sorted(os.listdir("."))

print(f"Data directory exists: {data_exists}")
print(f"Output directory exists: {output_exists}")
print(f"Logs directory exists: {logs_exists}")
print(f"Logs/app directory exists: {logs_app_exists}")
print(f"Root items: {root_items}")
"""

    result = tool.invoke({"code": code})
    assert "Data directory exists: True" in result
    assert "Output directory exists: True" in result
    assert "Logs directory exists: True" in result
    assert "Logs/app directory exists: True" in result


def test_combined_files_and_directories(pyodide_package: None) -> None:
    """Test using both files and directories together."""
    tool = PyodideSandboxTool(
        allow_net=True,
        allow_read=True,
        files={"config.json": '{"app": "test"}', "data/input.txt": "Hello World"},
        directories=["output", "logs"],
    )

    code = """
import os
import json

# Read config file
with open("config.json", "r") as f:
    config = json.load(f)

# Read input file  
with open("data/input.txt", "r") as f:
    content = f.read()

# Write to output directory
with open("output/result.txt", "w") as f:
    f.write(f"App: {config['app']}, Content: {content}")

# Check what was created
output_files = os.listdir("output")
root_items = sorted([item for item in os.listdir(".") if not item.startswith(".")])

print(f"Config app: {config['app']}")
print(f"Input content: {content}")
print(f"Output files: {output_files}")
print(f"Root items: {root_items}")
"""

    result = tool.invoke({"code": code})
    assert "Config app: test" in result
    assert "Input content: Hello World" in result
    assert "result.txt" in result
