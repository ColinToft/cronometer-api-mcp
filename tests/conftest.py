"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _default_diary_groups(monkeypatch):
    """Every test starts with the default diary group names, whatever the
    shell exports; a test that needs others sets CRONOMETER_DIARY_GROUPS
    through monkeypatch, which undoes it afterwards."""
    monkeypatch.delenv("CRONOMETER_DIARY_GROUPS", raising=False)
