"""File operations with workspace safety.

All operations are sandboxed to AGENT_WORKSPACE (defaults to current directory).
Any path that resolves outside the workspace raises PermissionError.
"""

import os
from pathlib import Path
from typing import List


def _workspace() -> Path:
    return Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()


def _safe_path(relative_path: str) -> Path:
    """
    Resolve *relative_path* inside the workspace.
    Raises PermissionError if the resolved path escapes the workspace root.
    This prevents path traversal attacks (e.g. '../../etc/passwd').
    """
    workspace = _workspace()
    resolved = (workspace / relative_path).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise PermissionError(
            f"Path '{relative_path}' resolves outside workspace '{workspace}'"
        )
    return resolved


# --------------------------------------------------------------------------- #
# Operations                                                                    #
# --------------------------------------------------------------------------- #

def read_file(path: str) -> str:
    """Read and return the full contents of *path*."""
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    Replace the first occurrence of *old_text* with *new_text* in *path*.
    Raises ValueError if *old_text* is not found.
    """
    p = _safe_path(path)
    content = p.read_text(encoding="utf-8")
    if old_text not in content:
        raise ValueError(
            f"Text not found in '{path}':\n{old_text[:120]!r}..."
        )
    updated = content.replace(old_text, new_text, 1)
    p.write_text(updated, encoding="utf-8")
    return f"Edited {path}"


def delete_file(path: str) -> str:
    """Delete *path* from the workspace."""
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    p.unlink()
    return f"Deleted {path}"


def list_files(directory: str = ".") -> List[str]:
    """Return all files under *directory*, relative to workspace root."""
    workspace = _workspace()
    d = _safe_path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(
        str(f.relative_to(workspace)).replace("\\", "/")
        for f in d.rglob("*")
        if f.is_file()
    )


# --------------------------------------------------------------------------- #
# Dispatch table (used by the agent runner)                                     #
# --------------------------------------------------------------------------- #

FILE_OPS = {
    "read":   lambda p, **kw: read_file(p),
    "write":  lambda p, content="", **kw: write_file(p, content),
    "edit":   lambda p, old_text="", new_text="", **kw: edit_file(p, old_text, new_text),
    "delete": lambda p, **kw: delete_file(p),
    "list":   lambda p=".", **kw: "\n".join(list_files(p)),
}
