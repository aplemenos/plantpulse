"""
Semantic search over the synthetic maintenance-log corpus.

Embedding backend is behind a small interface (`Embedder`) so the neural
option can be swapped in with a one-line change. This sandbox's network
policy blocks huggingface.co (no model downloads), so the shipped default
is `TfidfSvdEmbedder` - a classical TF-IDF + truncated-SVD (LSA) embedder
that is fully offline and deterministic. In a normal environment,
`FastEmbedEmbedder` (sentence-transformer style, via fastembed/ONNX) is a
drop-in replacement with the exact same `.embed(texts) -> np.ndarray`
contract - see the class below, left in place but unused by default.

FAISS is used for the vector index regardless of embedding backend.

Build:  python -m ml.vector_store
Reuse:  from ml.vector_store import VectorStore; VectorStore.load().search("...")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

import faiss
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline

from ml.data_generation import MaintenanceLog

ROOT = Path(__file__).resolve().parent.parent
LOGS_PATH = ROOT / "data" / "maintenance_logs.json"
STORE_DIR = ROOT / "ml" / "artifacts" / "vector_store"
EMBED_DIM = 128

Vectors = NDArray[np.float32]


class SearchHit(MaintenanceLog):
    score: float


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Vectors: ...


class TfidfSvdEmbedder:
    """Offline embedder: TF-IDF -> L2-normalized TruncatedSVD (LSA), dim=EMBED_DIM.

    No network calls, no GPU, sub-millisecond per query once fitted -
    the right tradeoff here, where the goal is a correct, demonstrable
    retrieval pipeline rather than state-of-the-art semantic recall.
    """

    def __init__(self, n_components: int = EMBED_DIM) -> None:
        self.n_components = n_components
        self._pipeline: Pipeline | None = None

    def fit(self, texts: list[str]) -> TfidfSvdEmbedder:
        n_components = min(self.n_components, max(2, len(texts) - 1))
        self._pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english"),
            TruncatedSVD(n_components=n_components, random_state=42),
        )
        self._pipeline.fit(texts)
        return self

    def embed(self, texts: list[str]) -> Vectors:
        if self._pipeline is None:
            raise RuntimeError("TfidfSvdEmbedder must be fitted or loaded before embedding")
        return self._pipeline.transform(texts).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path) -> TfidfSvdEmbedder:
        inst = cls()
        inst._pipeline = cast(Pipeline, joblib.load(path))
        return inst


class FastEmbedEmbedder:
    """Neural embedder via fastembed/ONNX (BAAI/bge-small-en-v1.5, 384-dim).

    Unused by default in this sandbox (huggingface.co is network-blocked
    here) but kept as the documented swap-in for a normal deployment:
    `VectorStore.build(embedder=FastEmbedEmbedder())`.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        from fastembed import (
            TextEmbedding,  # imported lazily - optional dependency path
        )

        self._model = TextEmbedding(model_name=self.MODEL_NAME)

    def embed(self, texts: list[str]) -> Vectors:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


class VectorStore:
    def __init__(self, index: faiss.Index, records: list[MaintenanceLog], embedder: Embedder) -> None:
        self._index = index
        self._records = records
        self._embedder = embedder

    @classmethod
    def build(cls, embedder: Embedder | None = None) -> VectorStore:
        records = _read_records(LOGS_PATH)
        texts = [r["text"] for r in records]

        embedder = embedder or TfidfSvdEmbedder().fit(texts)
        vectors = embedder.embed(texts)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(STORE_DIR / "index.faiss"))
        with open(STORE_DIR / "records.json", "w") as f:
            json.dump(records, f, indent=2)
        if isinstance(embedder, TfidfSvdEmbedder):
            embedder.save(STORE_DIR / "embedder.joblib")
        with open(STORE_DIR / "metadata.json", "w") as f:
            json.dump(
                {"embedder": type(embedder).__name__, "n_records": len(records),
                 "dim": int(vectors.shape[1])}, f, indent=2,
            )

        print(f"[vector_store] indexed {len(records)} maintenance logs "
              f"via {type(embedder).__name__}, dim={vectors.shape[1]}")
        return cls(index, records, embedder)

    @classmethod
    def load(cls) -> VectorStore:
        index = faiss.read_index(str(STORE_DIR / "index.faiss"))
        records = _read_records(STORE_DIR / "records.json")
        embedder = TfidfSvdEmbedder.load(STORE_DIR / "embedder.joblib")
        return cls(index, records, embedder)

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        qvec = self._embedder.embed([query])
        faiss.normalize_L2(qvec)
        scores, idxs = self._index.search(qvec, k)
        out: list[SearchHit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            rec = self._records[idx]
            out.append(
                SearchHit(
                    log_id=rec["log_id"],
                    site_id=rec["site_id"],
                    event_type=rec["event_type"],
                    date=rec["date"],
                    text=rec["text"],
                    score=round(score, 4),
                )
            )
        return out


def _read_records(path: Path) -> list[MaintenanceLog]:
    with open(path) as f:
        return cast(list[MaintenanceLog], json.load(f))


if __name__ == "__main__":
    store = VectorStore.build()
    for q in [
        "power output dropped suddenly but weather looks normal",
        "turbine vibration increasing over several days",
        "grid operator asked us to reduce output",
    ]:
        print(f"\nquery: {q!r}")
        for r in store.search(q, k=3):
            print(f"  [{r['score']}] {r['text']}")
