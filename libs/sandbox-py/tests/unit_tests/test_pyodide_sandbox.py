"""Test pyodide sandbox functionality."""

import os
from pathlib import Path

import pytest

from langchain_sandbox import PyodideSandbox

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
