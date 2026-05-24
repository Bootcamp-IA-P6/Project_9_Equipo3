"""Fixtures compartidas para tests del proyecto."""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _project_cwd():
    """Los módulos y configs usan rutas relativas al root del repo."""
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(prev)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def features_config(project_root: Path) -> str:
    return str(project_root / "configs" / "features.yaml")


@pytest.fixture(scope="session")
def models_config(project_root: Path) -> str:
    return str(project_root / "configs" / "models.yaml")


@pytest.fixture(scope="session")
def best_params_config(project_root: Path) -> str:
    return str(project_root / "configs" / "best_params.yaml")
