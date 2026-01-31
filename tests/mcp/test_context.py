"""Tests for MCP context auto-detection."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

import pytest

from engram.mcp.context import (
    clear_context_cache,
    get_default_user,
    get_project_context,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear context caches before and after each test."""
    clear_context_cache()
    yield
    clear_context_cache()


class TestGetProjectContext:
    """Tests for get_project_context()."""

    def test_explicit_env_override(self):
        """ENGRAM_PROJECT_ID env var takes precedence."""
        with patch.dict(os.environ, {"ENGRAM_PROJECT_ID": "my-custom-project"}):
            clear_context_cache()
            assert get_project_context() == "my-custom-project"

    def test_github_ssh_remote(self):
        """Parses GitHub SSH remote URL correctly."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "git@github.com:evanvolgas/engram.git\n"
                result = get_project_context()
                assert result == "evanvolgas/engram"

    def test_github_https_remote(self):
        """Parses GitHub HTTPS remote URL correctly."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "https://github.com/evanvolgas/engram.git\n"
                result = get_project_context()
                assert result == "evanvolgas/engram"

    def test_gitlab_remote(self):
        """Parses GitLab remote URL correctly."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "git@gitlab.com:myorg/myproject.git\n"
                result = get_project_context()
                assert result == "myorg/myproject"

    def test_bitbucket_remote(self):
        """Parses Bitbucket remote URL correctly."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "git@bitbucket.org:team/repo.git\n"
                result = get_project_context()
                assert result == "team/repo"

    def test_fallback_to_git_root(self):
        """Falls back to git root directory name when no remote."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:

                def side_effect(cmd, **kwargs):
                    # cmd is a list like ["git", "config", "--get", "remote.origin.url"]
                    cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                    if "remote.origin.url" in cmd_str:
                        raise subprocess.CalledProcessError(1, cmd)
                    if "show-toplevel" in cmd_str:
                        return "/home/user/projects/my-repo\n"
                    raise subprocess.CalledProcessError(1, cmd)

                mock.side_effect = side_effect
                result = get_project_context()
                assert result == "my-repo"

    def test_fallback_to_cwd(self):
        """Falls back to current directory when not a git repo."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.side_effect = subprocess.CalledProcessError(1, "git")
                with patch("engram.mcp.context.os.getcwd", return_value="/home/user/my-project"):
                    result = get_project_context()
                    assert result == "my-project"

    def test_caching(self):
        """Result is cached after first call."""
        with patch.dict(os.environ, {"ENGRAM_PROJECT_ID": "cached-project"}):
            clear_context_cache()
            result1 = get_project_context()
            # Change env var - should still return cached value
            os.environ["ENGRAM_PROJECT_ID"] = "different-project"
            result2 = get_project_context()
            assert result1 == result2 == "cached-project"


class TestGetDefaultUser:
    """Tests for get_default_user()."""

    def test_explicit_env_var(self):
        """ENGRAM_USER env var takes precedence."""
        with patch.dict(os.environ, {"ENGRAM_USER": "evan"}):
            clear_context_cache()
            assert get_default_user() == "evan"

    def test_git_user_name(self):
        """Falls back to git user.name."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ENGRAM_USER if present
            os.environ.pop("ENGRAM_USER", None)
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "Evan Volgas\n"
                result = get_default_user()
                assert result == "evan-volgas"

    def test_git_user_name_simple(self):
        """Handles simple git usernames."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ENGRAM_USER", None)
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.return_value = "evan\n"
                result = get_default_user()
                assert result == "evan"

    def test_fallback_to_system_user(self):
        """Falls back to system USER env var."""
        with patch.dict(os.environ, {"USER": "sysuser"}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.side_effect = subprocess.CalledProcessError(1, "git")
                result = get_default_user()
                assert result == "sysuser"

    def test_fallback_to_username_windows(self):
        """Falls back to USERNAME env var (Windows)."""
        with patch.dict(os.environ, {"USERNAME": "winuser"}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.side_effect = subprocess.CalledProcessError(1, "git")
                result = get_default_user()
                assert result == "winuser"

    def test_returns_none_when_no_user(self):
        """Returns None when no user can be detected."""
        with patch.dict(os.environ, {}, clear=True):
            clear_context_cache()
            with patch("engram.mcp.context.subprocess.check_output") as mock:
                mock.side_effect = subprocess.CalledProcessError(1, "git")
                result = get_default_user()
                assert result is None

    def test_caching(self):
        """Result is cached after first call."""
        with patch.dict(os.environ, {"ENGRAM_USER": "cached-user"}):
            clear_context_cache()
            result1 = get_default_user()
            os.environ["ENGRAM_USER"] = "different-user"
            result2 = get_default_user()
            assert result1 == result2 == "cached-user"


class TestClearContextCache:
    """Tests for clear_context_cache()."""

    def test_clears_both_caches(self):
        """Clears both project and user caches."""
        with patch.dict(os.environ, {"ENGRAM_PROJECT_ID": "project1", "ENGRAM_USER": "user1"}):
            clear_context_cache()
            p1 = get_project_context()
            u1 = get_default_user()

            # Change env vars
            os.environ["ENGRAM_PROJECT_ID"] = "project2"
            os.environ["ENGRAM_USER"] = "user2"

            # Still cached
            assert get_project_context() == p1
            assert get_default_user() == u1

            # Clear and re-fetch
            clear_context_cache()
            assert get_project_context() == "project2"
            assert get_default_user() == "user2"
