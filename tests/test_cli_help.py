"""Regression tests verifying `--help` CLI invocation across all scripts and src CLI modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATHS = sorted(
    p for p in (REPO_ROOT / "scripts").glob("*.py") if p.name != "__init__.py"
)

SRC_CLI_MODULES = (
    "src.data.arctic_shift_vn",
    "src.data.pheme_loader",
    "src.nlp.dual_edge_scorer",
    "src.nlp.extract_features",
)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_discovered_cli_entrypoint_counts() -> None:
    assert len(SCRIPT_PATHS) >= 17
    assert len(SRC_CLI_MODULES) == 4


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_direct_help(script_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"Direct `--help` failed for {script_path.name} (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ValueError" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "usage:" in proc.stdout.lower()
    if script_path.name == "demo_research_walkthrough.py":
        assert "MINH HỌA CƠ CHẾ HỌC THUẬT" not in proc.stdout


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: f"m_{p.stem}")
def test_script_module_help(script_path: Path) -> None:
    mod_name = f"scripts.{script_path.stem}"
    proc = subprocess.run(
        [sys.executable, "-m", mod_name, "--help"],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"Module `-m {mod_name} --help` failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ValueError" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "usage:" in proc.stdout.lower()


@pytest.mark.parametrize("module_name", SRC_CLI_MODULES)
def test_src_module_help(module_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"Module `-m {module_name} --help` failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "ValueError" not in proc.stderr
    assert "ModuleNotFoundError" not in proc.stderr
    assert "usage:" in proc.stdout.lower()
    if module_name == "src.nlp.extract_features":
        assert "100% Teacher" in proc.stdout
