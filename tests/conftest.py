"""Shared test fixtures for the XmLLM project."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from src.app.settings import Settings, get_settings

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    """Keep the lru_cache on get_settings() from leaking between tests.

    Tests that change env vars mid-test would otherwise read a cached
    Settings instance from an earlier run.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Temporary storage root for tests that need filesystem access."""
    storage = tmp_path / "data"
    storage.mkdir()
    return storage


@pytest.fixture
def test_settings(tmp_storage: Path) -> Settings:
    """Settings configured for testing — uses tmp_path for storage."""
    return Settings(
        app_mode="local",
        storage_root=tmp_storage,
        db_name="test.db",
        log_level="debug",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with lifespan — ensures DB/FileStore/JobService are initialized."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "data"))
    from src.app import settings as settings_mod
    monkeypatch.setattr(
        settings_mod,
        "get_settings",
        lambda: settings_mod.Settings(storage_root=tmp_path / "data"),
    )
    from fastapi.testclient import TestClient
    from src.app.main import app
    with TestClient(app) as c:
        yield c
