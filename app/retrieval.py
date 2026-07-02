"""
Two genuinely different retrieval mechanisms, dispatched by mode:

- recommend / refine -> hybrid BM25 + embedding search, fused by
  reciprocal rank fusion. Query text built from the re-derived
  `constraints` dict, not raw conversation text.

- compare -> exact/fuzzy NAME lookup against the catalog, no ranking.
  Users say "OPQ" and mean "Occupational Personality Questionnaire
  OPQ32r" — alias resolution via substring + fuzzy match, not semantic
  search, because semantic search on a 2-3 word product nickname
  against a 600-item catalog is unreliable.

Candidates returned by EITHER path are the only items the generator is
allowed to reference. This is the hallucination guard.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.catalog import load_filtered_catalog, searchable_text

INDEX_DIR = Path(__file__).parent.parent / "catalog" / "processed"
EMB_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CATALOG_CACHE_PATH = INDEX_DIR / "catalog_filtered.json"

_model = None
_catalog: list[dict] | None = None
_embeddings: np.ndarray | None = None
_bm25: BM25Okapi | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index() -> None:
    """Run offline / at build time, not at request time."""
    catalog = load_filtered_catalog()
    texts = [searchable_text(item) for item in catalog]

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    np.save(EMB_PATH, embeddings)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print(f"Indexed {len(catalog)} records -> {EMB_PATH}, {BM25_PATH}")


def _load_index():
    global _catalog, _embeddings, _bm25
    if _catalog is None:
        _catalog = load_filtered_catalog()
    if _embeddings is None:
        if not EMB_PATH.exists():
            raise FileNotFoundError("Run `python -m app.retrieval build` first (offline index build).")
        _embeddings = np.load(EMB_PATH)
    if _bm25 is None:
        with open(BM25_PATH, "rb") as f:
            _bm25 = pickle.load(f)
    return _catalog, _embeddings, _bm25


def hybrid_search(query_text: str, top_k: int = 15) -> list[dict]:
    catalog, embeddings, bm25 = _load_index()

    model = _get_model()
    q_emb = model.encode([query_text], normalize_embeddings=True)[0]
    vec_scores = embeddings @ q_emb
    vec_rank = np.argsort(-vec_scores)

    bm25_scores = bm25.get_scores(query_text.lower().split())
    bm25_rank = np.argsort(-bm25_scores)

    # reciprocal rank fusion
    k = 60
    fused: dict[int, float] = {}
    for rank, idx in enumerate(vec_rank):
        fused[idx] = fused.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(bm25_rank):
        fused[idx] = fused.get(idx, 0) + 1 / (k + rank + 1)

    top_indices = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [catalog[i] for i in top_indices]


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()


def exact_entity_lookup(entity_names: list[str]) -> list[dict]:
    """For compare mode. Substring match on normalized name — catches
    'OPQ' matching 'Occupational Personality Questionnaire OPQ32r'
    because the query term appears as a substring of the real name.
    Does NOT catch abbreviations that aren't substrings (e.g. some
    users might say 'the personality test' with zero lexical overlap —
    that case returns no match and the generator must say so, not guess)."""
    catalog, _, _ = _load_index()
    results = []
    for entity in entity_names:
        norm_entity = _normalize(entity)
        matches = [
            item for item in catalog
            if norm_entity in _normalize(item.get("name", ""))
        ]
        results.extend(matches)
    # de-dupe, preserve order
    seen = set()
    deduped = []
    for item in results:
        key = item.get("name")
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped

def warmup():
    """Load all retrieval resources into memory during application startup."""
    _get_model()      # Load SentenceTransformer
    _load_index()     # Load catalog, embeddings and BM25 index

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
