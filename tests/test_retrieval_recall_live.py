"""Live integration test for top_k=5 retrieval recall over indexed documents in Qdrant.

Requires running ami-rag stack (API server on port 8009, Qdrant, embed & rerank).
Automatically skips if API server is unreachable.
"""

import json
import os
import time
from pathlib import Path

import httpx
import pytest

CASES_FILE = Path(__file__).parent / "retrieval_cases.json"
BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8009").rstrip("/")
API_KEY = os.getenv("RAG_API_KEY", "")
TIMEOUT = float(os.getenv("RAG_TIMEOUT", "60.0"))


def _load_cases() -> list[dict]:
    with open(CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def _check_service_ready() -> bool:
    try:
        res = httpx.get(f"{BASE_URL}/readyz", timeout=3.0)
        return res.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _check_service_ready(),
        reason=f"ami-rag API server is not accessible or not ready at {BASE_URL}",
    ),
]


def _build_params():
    cases = _load_cases()
    params = []
    for c in cases:
        marks = []
        if c.get("xfail"):
            marks.append(pytest.mark.xfail(reason=c.get("xfail_reason") or "Expected low recall"))
        params.append(pytest.param(c, id=f"{c['modality']}-{c['id']}", marks=marks))
    return params


def _check_hit(case: dict, documents: list[dict]) -> tuple[bool, int]:
    """Hit khi top-k chứa tài liệu mong đợi (`expected_doc_id`), mọi modality.

    Không so khớp `expected_chunk_id`: chunk_id là hash MD5 nên đổi sau mỗi lần
    reindex / đổi caption, làm test báo MISS giả dù nội dung vẫn truy xuất đúng.
    """
    for rank, doc in enumerate(documents, start=1):
        doc_id = (doc.get("doc") or {}).get("document_id")
        if doc_id == case["expected_doc_id"]:
            return True, rank
    return False, -1


def run_case(case: dict, top_k: int = 5) -> dict:
    """Run one case against the live API and report the outcome.

    Shared by the pytest tests below and by `scripts/run_fusion_bench.py`, so a
    strategy comparison and the official assertions score cases identically.
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "messages": [{"role": "user", "content": case["query"]}],
        "top_k": top_k,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        res = client.post(f"{BASE_URL}/v2/rag/", json=payload, headers=headers)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if res.status_code != 200:
        return {
            "id": case["id"],
            "modality": case["modality"],
            "status": "error",
            "hit": False,
            "rank": -1,
            "latency_ms": elapsed_ms,
            "detail": res.text[:200],
        }

    body = res.json()
    documents = body.get("documents", [])
    hit, rank = _check_hit(case, documents)
    return {
        "id": case["id"],
        "modality": case["modality"],
        "status": "hit" if hit else "miss",
        "hit": hit,
        "rank": rank,
        "latency_ms": elapsed_ms,
        "server_latency_ms": (body.get("meta") or {}).get("latency_ms"),
        "fusion": (body.get("meta") or {}).get("fusion"),
        "documents": documents,
    }


@pytest.mark.parametrize("case", _build_params())
def test_retrieval_top5_recall(case: dict):
    result = run_case(case, top_k=5)
    documents = result["documents"]
    details = [
        {
            "rank": idx + 1,
            "modality": d.get("modality"),
            "reference_id": d.get("reference_id"),
            "doc_id": (d.get("doc") or {}).get("document_id"),
            "score": round(d.get("score") or 0.0, 4),
        }
        for idx, d in enumerate(documents)
    ]

    expected_target = f"doc_id={case['expected_doc_id']}"

    assert result["status"] != "error", f"API error: {result['detail']}"
    assert len(documents) <= 5, f"Expected at most 5 documents, got {len(documents)}"

    assert result["hit"], (
        f"Case '{case['id']}' ({case['modality']}) MISS in top 5.\n"
        f"Query: {case['query']}\n"
        f"Expected: {expected_target}\n"
        f"Returned ({len(documents)} docs): {details}"
    )


@pytest.mark.parametrize("case", _build_params())
def test_retrieval_top20_recall(case: dict):
    result = run_case(case, top_k=20)
    documents = result["documents"]
    expected_target = f"doc_id={case['expected_doc_id']}"

    assert result["status"] != "error", f"API error: {result['detail']}"
    assert len(documents) <= 20, f"Expected at most 20 documents, got {len(documents)}"
    assert result["hit"], (
        f"Case '{case['id']}' ({case['modality']}) MISS in top 20.\n"
        f"Query: {case['query']}\n"
        f"Expected: {expected_target}"
    )


@pytest.mark.parametrize("case", _build_params())
def test_retrieval_top40_recall(case: dict):
    result = run_case(case, top_k=40)
    documents = result["documents"]
    expected_target = f"doc_id={case['expected_doc_id']}"

    assert result["status"] != "error", f"API error: {result['detail']}"
    assert len(documents) <= 40, f"Expected at most 40 documents, got {len(documents)}"
    assert result["hit"], (
        f"Case '{case['id']}' ({case['modality']}) MISS in top 40.\n"
        f"Query: {case['query']}\n"
        f"Expected: {expected_target}"
    )


def test_retrieval_summary_report():
    """Summary test collecting Hit@5, Hit@20, Hit@40 and MRR stats across all modalities."""
    cases = _load_cases()
    ks = [5, 20, 40]
    results = {
        k: {
            "text": {"hit": 0, "total": 0, "rr_sum": 0.0},
            "table": {"hit": 0, "total": 0, "rr_sum": 0.0},
            "image": {"hit": 0, "total": 0, "rr_sum": 0.0},
        }
        for k in ks
    }

    for case in cases:
        modality = case["modality"]
        # Query top_k=40 once, then compute hits across 5, 20, 40
        res = run_case(case, top_k=40)
        rank = res["rank"] if res["hit"] else -1
        for k in ks:
            results[k][modality]["total"] += 1
            if 1 <= rank <= k:
                results[k][modality]["hit"] += 1
                results[k][modality]["rr_sum"] += 1.0 / rank

    print("\n" + "=" * 75)
    print("           RETRIEVAL RECALL (HIT@5, HIT@20, HIT@40) & MRR SUMMARY REPORT")
    print("=" * 75)
    print(f"{'K':<5} | {'MODALITY':<8} | {'HIT / TOTAL':<12} | {'HIT RATE':<10} | {'MRR':<8}")
    print("-" * 75)
    for k in ks:
        k_res = results[k]
        total_hits = sum(s["hit"] for s in k_res.values())
        total_cases = sum(s["total"] for s in k_res.values())
        total_rr = sum(s["rr_sum"] for s in k_res.values())
        for modality, stats in k_res.items():
            hit = stats["hit"]
            tot = stats["total"]
            pct = (hit / tot * 100) if tot > 0 else 0.0
            mrr = (stats["rr_sum"] / tot) if tot > 0 else 0.0
            print(f"@{k:<4} | {modality.upper():<8} | {hit}/{tot:<10} | {pct:6.1f}%    | {mrr:.4f}")
        overall_pct = (total_hits / total_cases * 100) if total_cases > 0 else 0.0
        overall_mrr = (total_rr / total_cases) if total_cases > 0 else 0.0
        print(f"@{k:<4} | {'OVERALL':<8} | {total_hits}/{total_cases:<10} | {overall_pct:6.1f}%    | {overall_mrr:.4f}")
        print("-" * 75)
    print("=" * 75 + "\n")

    # Assertions for Hit@5 baseline
    assert results[5]["text"]["hit"] >= 8, "Text retrieval should hit at least 8/10 at top 5"
    assert results[5]["table"]["hit"] >= 3, "Table retrieval should hit at least 3/9 at top 5"
    assert results[5]["image"]["hit"] >= 1, "Image retrieval should hit at least 1/9 at top 5"

    # Assertions for Hit@20 expansion
    assert results[20]["table"]["hit"] >= 5, "Table retrieval should hit at least 5/9 at top 20"

    # Assertions for Hit@40 expansion
    assert results[40]["table"]["hit"] >= 7, "Table retrieval should hit at least 7/9 at top 40"
    assert results[40]["text"]["hit"] == 10, "Text retrieval should achieve 10/10 at top 40"
