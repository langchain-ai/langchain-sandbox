"""Test pyodide sandbox functionality."""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_sandbox import PyodideSandbox


@contextmanager
def sandbox_context(  # noqa: PLR0913
    *,
    allow_read: list[str] | str | bool = "node_modules",
    allow_write: list[str] | str | bool = "node_modules",
    allow_net: list[str] | bool = True,
    allow_env: list[str] | bool = False,
    allow_run: list[str] | bool = False,
    allow_ffi: list[str] | bool = False,
) -> Iterator[tuple[PyodideSandbox, str]]:
    """Create a PyodideSandbox instance with a temporary directory for sessions.

    This context manager creates a sandbox with a temporary directory and ensures
    cleanup when the context exits.

    Args:
        allow_read: File system read permissions
        allow_write: File system write permissions
        allow_net: Network access permissions
        allow_env: Environment variable access permissions
        allow_run: Subprocess execution permissions
        allow_ffi: Foreign Function Interface permissions

    Yields:
        A tuple containing (sandbox_instance, temp_directory_path)

    """
    # Create temporary directory
    temp_sessions_dir = tempfile.mkdtemp(prefix="pyodide_test_sessions_")

    try:
        # Set default permissions for temp dir
        actual_read = [] if allow_read is None else allow_read
        actual_write = [] if allow_write is None else allow_write

        # convert str to list
        if isinstance(actual_read, str):
            actual_read = [actual_read]
        if isinstance(actual_write, str):
            actual_write = [actual_write]

        # Ensure the temp directory is always readable and writable
        if isinstance(actual_read, list) and temp_sessions_dir not in actual_read:
            actual_read.append(temp_sessions_dir)

        if isinstance(actual_write, list) and temp_sessions_dir not in actual_write:
            actual_write.append(temp_sessions_dir)

        # Create the sandbox
        sandbox = PyodideSandbox(
            sessions_dir=temp_sessions_dir,
            allow_read=actual_read,
            allow_write=actual_write,
            allow_net=allow_net,
            allow_env=allow_env,
            allow_run=allow_run,
            allow_ffi=allow_ffi,
        )

        # Yield the sandbox and temp dir
        yield sandbox, temp_sessions_dir

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_sessions_dir, ignore_errors=True)


async def test_stdout_sessionless() -> None:
    """Test without a session ID."""
    with sandbox_context() as (sandbox, _):
        # Execute a simple piece of code synchronously
        result = await sandbox.execute("x = 5; print(x); x")
        assert result.status == "success"
        assert result.stdout == "5"
        assert result.result == 5
        assert result.stderr is None


async def test_session_state_persistence_basic() -> None:
    """Simple test to verify that a session ID is used to persist state.

    We'll assign a variable in one execution and check if it's available in the next.
    """
    with sandbox_context() as (sandbox, _):
        # Test with a session ID to ensure state persistence
        session_id = "test_session_1"
        result1 = await sandbox.execute("y = 10; print(y)", session_id=session_id)
        result2 = await sandbox.execute("print(y)", session_id=session_id)

        # Check session state persistence
        assert result1.status == "success", f"Encountered error: {result1.stderr}"
        assert result1.stdout == "10"
        assert result1.result is None
        assert result2.status == "success", f"Encountered error: {result2.stderr}"
        assert result2.stdout == "10"
        assert result1.result is None


async def test_pyodide_sandbox_error_handling() -> None:
    """Test PyodideSandbox error handling."""
    with sandbox_context() as (sandbox, _):
        # Test syntax error
        result = await sandbox.execute("x = 5; y = x +")
        assert result.status == "error"
        assert "SyntaxError" in result.stderr

        # Test undefined variable error
        result = await sandbox.execute("undefined_variable")
        assert result.status == "error"
        assert "NameError" in result.stderr


async def test_pyodide_sandbox_timeout() -> None:
    """Test PyodideSandbox timeout handling."""
    with sandbox_context() as (sandbox, _):
        # Test timeout with infinite loop
        # Using a short timeout to avoid long test runs
        result = await sandbox.execute("while True: pass", timeout_seconds=0.5)
        assert result.status == "error"
        assert "timed out" in result.stderr.lower()


# Currently, we do not support file operations persisting across sessions
async def test_pyodide_sandbox_file_operations() -> None:
    """Test file operations are not persisted across sessions."""
    with sandbox_context() as (sandbox, _):
        # Test file I/O within the allowed session directory
        # Note: In Pyodide, the sessions directory might be mapped differently
        # We are testing the ability to write files via the sandbox,
        # which is what matters
        session_id = "file_test_session"

        # Create a file in the session
        code_write = """
import json
data = {'test': 'value', 'number': 42}
with open('test_data.json', 'w') as f:
    json.dump(data, f)
"""
        result = await sandbox.execute(code_write, session_id=session_id)
        assert result.status == "success"
        assert result.stdout is None
        assert result.stderr is None
        assert result.result is None

        # Confirm that the file does not exist in the session directory
        code_read = """\
import os
if os.path.exists('test_data.json'):
    raise Exception("File should not exist in the session directory.")
print("OK")
"""
        result = await sandbox.execute(code_read, session_id=session_id)
        assert result.status == "error"
        assert result.stdout is None
        assert "File should not exist" in result.stderr
