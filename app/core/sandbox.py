"""
A restricted execution sandbox for running LLM-generated pandas snippets.

This is NOT a full security boundary (true isolation would run this in a
separate container/process with no network and a syscall filter — noted in
the README as a production hardening item). For this assignment it provides
defense-in-depth appropriate to a local analytics tool:
  - no access to builtins like `open`, `exec`, `eval`, `__import__`
  - only whitelisted modules (pandas, numpy, math, statistics) are exposed
  - a wall-clock timeout guards against runaway loops
  - output size is capped before being sent back to the LLM/UI
"""
from __future__ import annotations

import ast
import contextlib
import io
import math
import signal
import statistics
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input",
    "os", "sys", "subprocess", "shutil", "socket", "requests",
}


class SandboxError(Exception):
    pass


class _Timeout:
    """POSIX-only wall clock timeout guard for a code block."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise SandboxError(f"Execution exceeded {self.seconds}s time limit.")

    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        except (ValueError, AttributeError):
            pass  # not on main thread / not POSIX — timeout best-effort

    def __exit__(self, *exc):
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass


# Modules already injected into the sandbox globals — an `import` line for
# exactly one of these is redundant, not dangerous, so it's safe to drop.
# Anything else (import os, import subprocess, from os import system, ...)
# must be LEFT IN the code so `_static_check` below can see and reject it.
# Silently stripping *all* import lines (the previous behavior) made the
# import ban a no-op: `import os\nresult = os.system(...)` would just lose
# its import line and fail later with a confusing NameError instead of
# being blocked outright.
_ALLOWED_IMPORT_MODULES = {"pandas", "numpy", "math", "statistics"}


def _is_whitelisted_import(stmt: str) -> bool:
    """True only for `import <allowed>` / `from <allowed> import ...`."""
    try:
        node = ast.parse(stmt, mode="exec").body[0]
    except (SyntaxError, IndexError):
        return False

    if isinstance(node, ast.Import):
        modules = {n.name.split(".")[0] for n in node.names}
    elif isinstance(node, ast.ImportFrom):
        modules = {(node.module or "").split(".")[0]}
    else:
        return False

    return bool(modules) and modules <= _ALLOWED_IMPORT_MODULES


def _sanitize_code(code: str) -> str:
    """Drop only harmless, already-provided import statements; leave any
    other import line in place so `_static_check` can reject it."""

    cleaned = []

    for line in code.splitlines():
        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            if _is_whitelisted_import(stripped):
                continue  # redundant — pd/np/math/statistics already provided
            # not whitelisted (e.g. "import os") — keep it so the static
            # check below can flag it explicitly.

        cleaned.append(line)

    code = "\n".join(cleaned)

    # Fix deprecated pandas frequency aliases
    code = code.replace(".resample('M')", ".resample('ME')")
    code = code.replace('.resample("M")', '.resample("ME")')
    code = code.replace("freq='M'", "freq='ME'")
    code = code.replace('freq="M"', 'freq="ME"')

    return code


def _static_check(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise SandboxError("Import statements are not allowed in sandboxed code.")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxError(f"Use of '{node.id}' is not allowed in sandboxed code.")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "remove", "rmdir"}:
            raise SandboxError(f"Use of '.{node.attr}' is not allowed in sandboxed code.")


def run_pandas_code(code: str, frames: Dict[str, pd.DataFrame], timeout_s: int = 8) -> Dict[str, Any]:
    """
    Executes `code` with the given named dataframes available as local
    variables, plus pandas/numpy. The code must assign its answer to a
    variable named `result`.

    Returns a dict: {"success": bool, "result": <repr/records>, "stdout": str, "error": str|None}
    """
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "sorted": sorted, "round": round, "abs": abs, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "str": str, "int": int, "float": float, "bool": bool, "print": print,
        "True": True, "False": False, "None": None,
    }

    local_env: Dict[str, Any] = {name: df.copy() for name, df in frames.items()}
    global_env = {
        "__builtins__": safe_builtins,
        "pd": pd, "np": np, "math": math, "statistics": statistics,
    }

    stdout_buf = io.StringIO()
    result: Dict[str, Any] = {"success": False, "result": None, "stdout": "", "error": None}

    try:
        code = _sanitize_code(code)
        _static_check(code)
        with _Timeout(timeout_s):
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, global_env, local_env)  # noqa: S102 - sandboxed on purpose
        if "result" not in local_env:
            raise SandboxError(
                "Your code must assign the final answer to a variable named 'result'."
            )

        value = local_env["result"]
        result["result"] = _serialize(value)
        result["success"] = True
    except SandboxError as e:
        result["error"] = str(e)
    except Exception as e:  # noqa: BLE001 - surface any runtime error to the LLM
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        result["stdout"] = stdout_buf.getvalue()[-2000:]

    if not result["success"]:
        logger.warning(f"Sandbox execution failed: {result['error']}")
    return result


def _serialize(value: Any, max_rows: int = 30) -> Any:
    """Convert pandas/numpy results into JSON-friendly, size-capped structures."""

    # Handle nested dictionaries
    if isinstance(value, dict):
        return {
            str(k): _serialize(v, max_rows)
            for k, v in list(value.items())[:100]
        }

    # Handle lists
    if isinstance(value, list):
        return [_serialize(v, max_rows) for v in value[:100]]

    # Handle tuples
    if isinstance(value, tuple):
        return tuple(_serialize(v, max_rows) for v in value[:100])

    # DataFrame
    if isinstance(value, pd.DataFrame):
        truncated = value.head(max_rows)
        return {
            "type": "dataframe",
            "shape": list(value.shape),
            "columns": list(value.columns.astype(str)),
            "rows": truncated.replace({np.nan: None}).to_dict(orient="records"),
            "truncated": len(value) > max_rows,
        }

    # Series
    if isinstance(value, pd.Series):
        truncated = value.head(max_rows)
        return {
            "type": "series",
            "length": int(len(value)),
            "values": truncated.replace({np.nan: None}).to_dict(),
            "truncated": len(value) > max_rows,
        }

    # NumPy scalars
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # NumPy array
    if isinstance(value, np.ndarray):
        return value.tolist()[:max_rows]

    # Pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # Pandas NaT
    if value is pd.NaT:
        return None

    return value