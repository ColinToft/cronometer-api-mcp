import pytest


@pytest.fixture(autouse=True)
def _default_diary_groups(monkeypatch):
    """Default group names, whatever the shell exports."""
    monkeypatch.delenv("CRONOMETER_DIARY_GROUPS", raising=False)
