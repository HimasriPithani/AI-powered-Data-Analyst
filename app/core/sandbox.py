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
        _static_check(code)
        with _Timeout(timeout_s):
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, global_env, local_env)  # noqa: S102 - sandboxed on purpose
        value = local_env.get("result", None)
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
    if isinstance(value, pd.DataFrame):
        truncated = value.head(max_rows)
        return {
            "type": "dataframe",
            "shape": list(value.shape),
            "columns": list(value.columns.astype(str)),
            "rows": truncated.replace({np.nan: None}).to_dict(orient="records"),
            "truncated": len(value) > max_rows,
        }
    if isinstance(value, pd.Series):
        truncated = value.head(max_rows)
        return {
            "type": "series",
            "length": int(len(value)),
            "values": truncated.replace({np.nan: None}).to_dict(),
            "truncated": len(value) > max_rows,
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()[:max_rows]
    return value

