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
