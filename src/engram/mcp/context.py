"""Auto-detect project context for memory isolation.

This module provides automatic project detection so that memories
are isolated per-project without requiring manual configuration.

Detection cascade:
1. ENGRAM_PROJECT_ID env var (explicit override)
2. Git remote URL (globally unique: "evanvolgas/engram")
3. Git repo root name (local uniqueness)
4. Current directory name (fallback)

User detection:
1. ENGRAM_USER env var
2. Git user.name config
3. System username
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_context() -> str:
    """Auto-detect project context for memory isolation.

    Returns a string identifier for the current project, used as org_id
    to isolate memories between different codebases.

    Detection order:
    1. ENGRAM_PROJECT_ID env var (explicit override)
    2. Git remote origin URL (globally unique)
    3. Git repository root directory name
    4. Current working directory name

    Returns:
        Project identifier string (e.g., "evanvolgas/engram" or "engram")
    """
    # 1. Explicit override
    if env_override := os.environ.get("ENGRAM_PROJECT_ID"):
        return env_override

    # 2. Git remote (globally unique)
    try:
        remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        # Handle SSH: "git@github.com:evanvolgas/engram.git"
        # Handle HTTPS: "https://github.com/evanvolgas/engram.git"
        if "github.com" in remote:
            path = remote.split("github.com")[-1]
            return path.lstrip(":/").replace(".git", "")
        if "gitlab.com" in remote:
            path = remote.split("gitlab.com")[-1]
            return path.lstrip(":/").replace(".git", "")
        if "bitbucket.org" in remote:
            path = remote.split("bitbucket.org")[-1]
            return path.lstrip(":/").replace(".git", "")
        # Generic: just use the repo name from the URL
        return remote.rstrip("/").split("/")[-1].replace(".git", "")
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass

    # 3. Git repo root name
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return os.path.basename(root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 4. Current directory name
    return os.path.basename(os.getcwd())


@lru_cache(maxsize=1)
def get_default_user() -> str | None:
    """Get default user identifier.

    Detection order:
    1. ENGRAM_USER env var
    2. Git user.name config
    3. System username (USER or USERNAME env var)

    Returns:
        User identifier string, or None if not detectable.
    """
    # 1. Explicit env var
    if env_user := os.environ.get("ENGRAM_USER"):
        return env_user

    # 2. Git user.name
    try:
        git_user = subprocess.check_output(
            ["git", "config", "--get", "user.name"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if git_user:
            # Normalize: "Evan Volgas" -> "evan-volgas"
            return git_user.lower().replace(" ", "-")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 3. System username
    return os.environ.get("USER") or os.environ.get("USERNAME")


def clear_context_cache() -> None:
    """Clear cached context values. Useful for testing."""
    get_project_context.cache_clear()
    get_default_user.cache_clear()
