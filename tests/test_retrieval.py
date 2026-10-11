from pathlib import Path
from typing import Any

import fastembed
import numpy as np
import pytest

from ml import vector_store
from ml.retrieval_eval import evaluate
from ml.vector_store import BM25, VectorStore, tokenize


@pytest.fixture(scope="module")
def store() -> VectorStore:
    return VectorStore.load()


def test_tokenize_stems_and_drops_stop_words() -> None:
    assert tokenize("The turbine is vibrating") == tokenize("turbine vibration")


def test_bm25_ranks_the_document_with_the_rare_term_first() -> None:
    bm25 = BM25(["inverter fan replaced", "IGBT module failed on inverter", "sensor recalibrated"])
    scores = bm25.scores("IGBT")
    assert scores.argmax() == 1 and scores[0] == scores[2] == 0


def test_bm25_ignores_unknown_words() -> None:
    assert BM25(["inverter fault"]).scores("zebra").sum() == 0


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
def test_site_filter_applies_in_every_mode(store: VectorStore, mode: str) -> None:
    hits = store.search("inverter tripped", k=10, site_id="solar-02", mode=mode)  # type: ignore[arg-type]
    assert hits and {h["site_id"] for h in hits} == {"solar-02"}


def test_hybrid_returns_k_unique_logs(store: VectorStore) -> None:
    hits = store.search("gearbox vibration", k=5)
    assert len(hits) == 5 and len({h["log_id"] for h in hits}) == 5


def test_hybrid_is_not_worse_than_either_ranking_alone_on_first_hits(store: VectorStore) -> None:
    hybrid = evaluate(store, "hybrid")
    assert hybrid["hit_at_3"] >= max(evaluate(store, "vector")["hit_at_3"], evaluate(store, "bm25")["hit_at_3"])


class _FakeTextEmbedding:
    """Stands in for fastembed's model so the bge code path runs without a download."""

    def __init__(self, model_name: str = "", cache_dir: str | None = None) -> None:
        pass

    @staticmethod
    def _vec(text: str) -> "np.ndarray[Any, np.dtype[np.float32]]":
        v = np.zeros(16, dtype=np.float32)
        for token in tokenize(text):
            v[hash(token) % 16] += 1.0
        return v

    def passage_embed(self, texts: list[str]) -> list["np.ndarray[Any, np.dtype[np.float32]]"]:
        return [self._vec(t) for t in texts]

    def query_embed(self, query: str) -> list["np.ndarray[Any, np.dtype[np.float32]]"]:
        return [self._vec(query)]


def test_bge_index_is_reloaded_with_bge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTextEmbedding)
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setenv("PLANTPULSE_EMBEDDER", "bge")

    built = VectorStore.build()
    loaded = VectorStore.load()
    assert built.info["embedder"] == loaded.info["embedder"] == "bge"
    assert loaded.search("gearbox vibration", k=3, site_id="wind-02")


def test_auto_falls_back_to_tfidf_when_bge_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unavailable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ValueError("Could not load model from any source.")

    monkeypatch.setattr(fastembed, "TextEmbedding", Unavailable)
    monkeypatch.setenv("PLANTPULSE_EMBEDDER", "auto")
    assert vector_store.select_embedder(["inverter fault", "sensor drift", "curtailment"]).name == "tfidf"
