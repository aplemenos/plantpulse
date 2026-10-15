import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def monkeypatch_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pytest.MonkeyPatch]:
    # Force the deterministic rule-based agent path and the in-process cache in tests,
    # regardless of what's set in the developer's shell.
    mp = pytest.MonkeyPatch()
    mp.delenv("ANTHROPIC_API_KEY", raising=False)
    mp.delenv("REDIS_URL", raising=False)
    # Keep test traces out of the experiment developers browse.
    mp.setenv("PLANTPULSE_TRACING", "1")
    mp.setenv("PLANTPULSE_AGENT_EXPERIMENT", "plantpulse-agent-tests")
    # Conversations go to a throwaway database, not var/sessions.db.
    mp.setenv("PLANTPULSE_SESSIONS_DB", str(tmp_path_factory.mktemp("sessions") / "sessions.db"))
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def client(monkeypatch_session: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as c:  # context manager runs the lifespan (model loading)
        yield c
