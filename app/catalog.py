"""
Catalog loader + Individual-Test-vs-Job-Solution filter.

Field names CONFIRMED against 3 real sample records (2026-07-02):
entity_id, name, link, scraped_at, job_levels, job_levels_raw, languages,
languages_raw, duration, duration_raw, status, remote, adaptive,
description, keys. There is NO `url` field (it's `link`) and NO
`test_type` field (see nodes.py::CATEGORY_TO_CODE for the derived mapping
and its known gap).

Filter heuristic (still UNVERIFIED against the full 600+ record set — none
of the 3 confirmed samples were Job Solution candidates, so this doesn't
validate the heuristic, only the field names it reads): bundled Job
Solutions were observed on an earlier partial view to mention "Precise
Fit" in description text and/or contain "Solution" in the name. Audit the
filtered-out and filtered-in lists manually before trusting this.

status filter: only "status": "ok" records are kept. Other status values
haven't been observed in the 3 confirmed samples but may exist elsewhere
in the full file — don't assume "ok" is universal without checking.
"""
from __future__ import annotations
import json
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "catalog" / "raw" / "shl_product_catalog.json"
FILTERED_PATH = Path(__file__).parent.parent / "catalog" / "processed" / "catalog_filtered.json"


def _is_job_solution(item: dict) -> bool:
    name = (item.get("name") or "").lower()
    desc = (item.get("description") or "").lower()
    if "precise fit" in desc:
        return True
    if "solution" in name:
        return True
    return False


def load_raw_catalog() -> list[dict]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {RAW_PATH}. Download it manually — "
            "this environment could not fetch it automatically — and place "
            "it there before running the indexer."
        )
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_filtered_catalog() -> list[dict]:
    raw = load_raw_catalog()

    not_ok = [item for item in raw if item.get("status") != "ok"]
    ok_only = [item for item in raw if item.get("status") == "ok"]
    if not_ok:
        print(f"WARNING: {len(not_ok)} records have non-'ok' status, excluded. "
              f"Statuses seen: {set(i.get('status') for i in not_ok)}")

    filtered = [item for item in ok_only if not _is_job_solution(item)]
    FILTERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FILTERED_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
    print(f"Raw: {len(raw)} | status=ok: {len(ok_only)} | "
          f"Filtered (Individual Test only): {len(filtered)} "
          f"| Excluded as Job Solutions: {len(ok_only) - len(filtered)}")
    return filtered


def load_filtered_catalog() -> list[dict]:
    if not FILTERED_PATH.exists():
        return build_filtered_catalog()
    with open(FILTERED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def searchable_text(item: dict) -> str:
    """Text used for embedding + BM25. Deliberately excludes structured
    fields (job_levels, languages, remote) — those should be metadata
    filters, not fuzzed into semantic search."""
    name = item.get("name", "")
    desc = item.get("description", "")
    return f"{name}. {desc}".strip()


if __name__ == "__main__":
    build_filtered_catalog()
