"""
Hybrid search over the maintenance-log corpus (the retrieval half of the
agent's RAG).

Each query is ranked two ways and the rankings are fused:

  - semantic: a dense embedding in a FAISS index. Finds "inverter got too
    hot" for a log that says "over-temperature fault".
  - keyword: BM25 over stemmed tokens. Finds exact part names and codes
    ("IGBT", "pyranometer") that embeddings tend to blur.

Fusion is Reciprocal Rank Fusion (score = sum of 1 / (60 + rank)), which
needs no score calibration between the two very different scales. Searches
can be restricted to one site.

Embedders (PLANTPULSE_EMBEDDER):
  - "bge"   : BAAI/bge-small-en-v1.5 via fastembed (ONNX, 384-dim, CPU).
  - "tfidf" : TF-IDF + truncated SVD (LSA). Offline and deterministic.
  - "auto"  : bge if the model can be downloaded, else tfidf. (default)
The embedder used is recorded in metadata.json and reused at load time, so
the query side always matches the index.

Build:  python -m ml.vector_store
Reuse:  from ml.vector_store import VectorStore; VectorStore.load().search("...")
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol, cast

import faiss
import joblib
import numpy as np
from numpy.typing import NDArray
from py_rust_stemmers import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from typing_extensions import TypedDict

from ml.data_generation import MaintenanceLog

ROOT = Path(__file__).resolve().parent.parent
LOGS_PATH = ROOT / "data" / "maintenance_logs.json"
STORE_DIR = ROOT / "ml" / "artifacts" / "vector_store"
FASTEMBED_CACHE = ROOT / ".cache" / "fastembed"
EMBED_DIM = 128
RRF_K = 60  # standard RRF constant; damps the influence of any single top rank

Vectors = NDArray[np.float32]
EmbedderName = Literal["bge", "tfidf"]
SearchMode = Literal["hybrid", "vector", "bm25"]


class SearchHit(MaintenanceLog):
    score: float


class StoreInfo(TypedDict):
    embedder: EmbedderName
    n_records: int
    dim: int


class Embedder(Protocol):
    name: EmbedderName

    def embed(self, texts: list[str]) -> Vectors: ...
    def embed_query(self, query: str) -> Vectors: ...


class TfidfSvdEmbedder:
    """Offline embedder: TF-IDF -> TruncatedSVD (LSA), dim=EMBED_DIM.

    No network, no GPU, deterministic. Weak at synonyms ("too hot" vs
    "over-temperature"), which is what the neural embedder is for.
    """

    name: EmbedderName = "tfidf"

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

    def embed_query(self, query: str) -> Vectors:
        return self.embed([query])

    def save(self, path: Path) -> None:
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: Path) -> TfidfSvdEmbedder:
        inst = cls()
        inst._pipeline = cast(Pipeline, joblib.load(path))
        return inst


class FastEmbedEmbedder:
    """Neural embedder: BAAI/bge-small-en-v1.5 via fastembed (ONNX, 384-dim, CPU).

    The model (~67 MB) is downloaded once into .cache/fastembed.
    """

    name: EmbedderName = "bge"
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy, and only needed for "bge"

        FASTEMBED_CACHE.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=self.MODEL_NAME, cache_dir=str(FASTEMBED_CACHE))

    def embed(self, texts: list[str]) -> Vectors:
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, query: str) -> Vectors:
        return np.array(list(self._model.query_embed(query)), dtype=np.float32)


def select_embedder(texts: list[str]) -> Embedder:
    choice = os.environ.get("PLANTPULSE_EMBEDDER", "auto")
    if choice not in ("auto", "bge", "tfidf"):
        raise ValueError(f"PLANTPULSE_EMBEDDER must be auto, bge or tfidf, got {choice!r}")
    if choice in ("auto", "bge"):
        try:
            return FastEmbedEmbedder()
        except (ImportError, ValueError, OSError) as exc:  # fastembed missing, or model download blocked
            if choice == "bge":
                raise
            print(f"[vector_store] bge unavailable ({type(exc).__name__}: {exc}); falling back to tfidf")
    return TfidfSvdEmbedder().fit(texts)


_STEMMER = SnowballStemmer("english")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stop words, Snowball-stem: 'vibrating' and 'vibration' both -> 'vibrat'."""
    return [_STEMMER.stem_word(t) for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25:
    """Okapi BM25 keyword scoring. Small enough to keep in memory and rebuild at load."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(t)) for t in texts]
        self._lengths = np.array([sum(d.values()) for d in self._docs], dtype=np.float64)
        self._avg_len = float(self._lengths.mean()) if len(texts) else 0.0
        doc_freq: Counter[str] = Counter()
        for d in self._docs:
            doc_freq.update(d.keys())
        n = len(texts)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: str) -> NDArray[np.float64]:
        out = np.zeros(len(self._docs), dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self._lengths / (self._avg_len or 1.0))
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self._docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class VectorStore:
    def __init__(self, index: faiss.Index, records: list[MaintenanceLog], embedder: Embedder) -> None:
        self._index = index
        self._records = records
        self._embedder = embedder
        self._bm25 = BM25([r["text"] for r in records])

    @property
    def info(self) -> StoreInfo:
        return {"embedder": self._embedder.name, "n_records": len(self._records), "dim": int(self._index.d)}

    @classmethod
    def build(cls, embedder: Embedder | None = None) -> VectorStore:
        records = _read_records(LOGS_PATH)
        texts = [r["text"] for r in records]

        embedder = embedder or select_embedder(texts)
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
        store = cls(index, records, embedder)
        (STORE_DIR / "metadata.json").write_text(json.dumps(store.info, indent=2))

        print(f"[vector_store] indexed {len(records)} maintenance logs via {embedder.name}, "
              f"dim={vectors.shape[1]}")
        return store

    @classmethod
    def load(cls) -> VectorStore:
        index = faiss.read_index(str(STORE_DIR / "index.faiss"))
        records = _read_records(STORE_DIR / "records.json")
        info = cast(StoreInfo, json.loads((STORE_DIR / "metadata.json").read_text()))
        embedder: Embedder = (
            FastEmbedEmbedder() if info["embedder"] == "bge"
            else TfidfSvdEmbedder.load(STORE_DIR / "embedder.joblib")
        )
        return cls(index, records, embedder)

    def _vector_ranking(self, query: str) -> list[tuple[int, float]]:
        qvec = self._embedder.embed_query(query)
        faiss.normalize_L2(qvec)
        scores, idxs = self._index.search(qvec, len(self._records))
        return [(i, s) for i, s in zip(idxs[0].tolist(), scores[0].tolist()) if i >= 0]

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.scores(query)
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(
        self, query: str, k: int = 3, site_id: str | None = None, mode: SearchMode = "hybrid"
    ) -> list[SearchHit]:
        """Top-k logs for `query`, optionally only from `site_id`.

        Scores are cosine similarity (vector), BM25 (bm25) or RRF (hybrid);
        only the order is comparable across modes.
        """
        allowed = {i for i, r in enumerate(self._records) if site_id is None or r["site_id"] == site_id}

        rankings: list[list[tuple[int, float]]] = []
        if mode in ("hybrid", "vector"):
            rankings.append([(i, s) for i, s in self._vector_ranking(query) if i in allowed])
        if mode in ("hybrid", "bm25"):
            rankings.append([(i, s) for i, s in self._bm25_ranking(query) if i in allowed])

        if mode == "hybrid":
            fused: dict[int, float] = {}
            for ranking in rankings:
                for rank, (i, _) in enumerate(ranking, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        else:
            ranked = rankings[0]

        return [SearchHit(**self._records[i], score=round(score, 4)) for i, score in ranked[:k]]


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
            print(f"  [{r['log_id']} {r['score']}] {r['text']}")
