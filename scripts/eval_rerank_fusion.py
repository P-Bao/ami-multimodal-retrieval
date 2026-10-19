"""Eval rerank fusion (raw vs rrf) trên bộ test JSONL - chạy trên server.

So sánh xếp hạng theo điểm tuyệt đối (raw - hành vi cũ) với fusion RRF 2 nhóm
modality (image vs phần còn lại) trên CÙNG một lần retrieve mỗi query:

    embed (gateway) -> vector search (Qdrant) -> rerank full scores (gateway)
    -> rank raw + rank rrf (fuse_modality_scores, sweep được cấu hình)

Bộ test JSONL mỗi dòng một case:
    {"query": "...", "expected_chunk_id": "chunk-..."}           (1 target)
    {"query": "...", "expected_chunk_ids": ["chunk-a", ...]}     (nhiều target)
Trường tuỳ chọn: "expected_modality" ("image"|"text"). Đổi tên trường qua
--query-field / --id-field nếu bộ test hiện có dùng tên khác.

Metrics: Recall@1/3/5 + MRR theo nhóm (image / text / overall) cho cả raw và
rrf; paired win/tie/loss (rrf tốt hơn raw khi hạng rrf của target < hạng raw);
phân bố điểm relevant/irrelevant theo nhóm (p10/p50/p90) để chỉnh floor/weight.
In cảnh báo khi số case mỗi nhóm < 10.

Usage:
    python scripts/eval_rerank_fusion.py --dataset eval_set.jsonl
    python scripts/eval_rerank_fusion.py --dataset eval_set.jsonl \
        --visual-weight 0.9 --rrf-k 10 --sweep --output result.json

Không gọi LLM. Cần gateway embed/rerank + Qdrant theo .env (EMBED_SERVER_URL,
RERANK_BASE_URL, QDRANT_URL).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ami_rag.core.rerank_client import (
    build_rerank_documents,
    build_rerank_func,
    fuse_modality_scores,
)
from ami_rag.settings import get_settings, resolve_rerank_backend

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_rerank_fusion")

RECALL_KS = (1, 3, 5)


def _is_match(chunk: dict, targets: set[str]) -> bool:
    cid = chunk.get("chunk_id")
    did = chunk.get("doc_id") or (chunk.get("doc") or {}).get("document_id")
    return bool((cid and cid in targets) or (did and did in targets))


def load_cases(
    path: Path, query_field: str, id_field: str, skip_xfail: bool = False
) -> list[dict]:
    """Đọc bộ test JSON (array) hoặc JSONL: mỗi case {query, expected_chunk_ids, modality}."""
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    rows: list[dict] = []
    if raw_text.startswith("["):
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                rows = data
        except json.JSONDecodeError:
            pass

    if not rows:
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("dòng %d lỗi JSON: %s", line_no, exc)

    cases: list[dict] = []
    for line_no, row in enumerate(rows, start=1):
        if skip_xfail and row.get("xfail"):
            continue
        query = row.get(query_field) or ""
        targets = (
            row.get(f"{id_field}s")
            or ([row.get(id_field)] if row.get(id_field) else None)
            or row.get("expected_doc_ids")
            or ([row.get("expected_doc_id")] if row.get("expected_doc_id") else None)
            or []
        )
        targets = [t for t in targets if t]
        if not query or not targets:
            logger.warning("dòng %d thiếu query/target, bỏ qua", line_no)
            continue
        modality = row.get("expected_modality") or row.get("modality") or "text"
        cases.append(
            {
                "query": query,
                "targets": set(targets),
                "modality": modality,
                "id": row.get("id"),
                "xfail": bool(row.get("xfail")),
            }
        )
    return cases


def _is_visual(chunk: dict) -> bool:
    return (chunk.get("modality") or "text") == "image"


async def retrieve(pipeline, rerank_func, query: str, top_k: int) -> list[dict]:
    """embed -> vector search -> rerank full scores. Trả [{chunk, raw, fused}].

    ``fused`` tính 1 lần duy nhất theo tham số fusion mặc định của settings;
    sweep cấu hình khác tính lại từ raw bằng ``refuse`` (không gọi lại rerank).
    """
    settings = get_settings()
    vector = await pipeline.embedder.embed_query(query)
    candidates_wanted = top_k * max(1, settings.RETRIEVAL_OVERFETCH)
    hits = await pipeline.vector_store.search(
        pipeline.collection,
        vector,
        max(settings.RETRIEVAL_CHUNK_TOP_K, candidates_wanted),
    )
    chunks = [hit.payload for hit in hits]
    if not chunks:
        return []

    multimodal = (
        getattr(settings, "RERANK_MULTIMODAL", True)
        and resolve_rerank_backend(settings) == "vllm"
    )
    documents = await build_rerank_documents(chunks, pipeline.asset_store, multimodal=multimodal)
    results = await rerank_func(query=query, documents=documents, top_n=len(documents))

    scored: list[dict] = []
    for item in results or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(chunks):
            scored.append({"chunk": chunks[idx], "raw": float(item.get("relevance_score") or 0.0)})
    return scored


def refuse(scored: list[dict], **fusion_params) -> list[dict]:
    """Tính lại fused score từ raw (không gọi lại rerank), sorted theo fused desc."""
    chunks = [s["chunk"] for s in scored]
    fused = fuse_modality_scores(
        [{"index": i, "relevance_score": s["raw"]} for i, s in enumerate(scored)],
        chunks,
        **fusion_params,
    )
    by_index = {f["index"]: f["fused_score"] for f in fused}
    for i, s in enumerate(scored):
        s["fused"] = by_index.get(i, s["raw"] - 1.0)
    return sorted(scored, key=lambda s: -s["fused"])


def rank_of_targets(scored: list[dict], targets: set[str]) -> int | None:
    """Hạng (1-based) tốt nhất của target trong danh sách đã sort; None nếu vắng."""
    for rank, s in enumerate(scored, start=1):
        if _is_match(s["chunk"], targets):
            return rank
    return None


def raw_ranking(scored: list[dict]) -> list[dict]:
    return sorted(scored, key=lambda s: -s["raw"])


def _mrr(ranks: list[int | None]) -> float:
    return statistics.mean(1.0 / r for r in ranks if r) if any(ranks) else 0.0


def _recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return statistics.mean(1.0 if (r is not None and r <= k) else 0.0 for r in ranks)


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return round(ordered[idx], 4)

    return {"p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9), "n": len(values)}


def evaluate(cases: list[dict], runs: list[dict]) -> dict:
    """runs: [{name, ranks_by_case, ...}] -> report metrics theo nhóm + overall."""
    report: dict = {"groups": {}, "overall": {}, "paired": {}, "distributions": {}}
    groups = sorted({c["modality"] for c in cases})

    for scope, subset in [("overall", cases)] + [(g, [c for c in cases if c["modality"] == g]) for g in groups]:
        entry: dict = {"n_cases": len(subset)}
        if len(subset) < 10:
            entry["warning"] = f"n={len(subset)} < 10: số liệu không đủ tin cậy"
        for run in runs:
            ranks = [run["ranks"][c["query"]] for c in subset]
            entry[run["name"]] = {
                "mrr": round(_mrr(ranks), 4),
                **{f"recall@{k}": round(_recall_at_k(ranks, k), 4) for k in RECALL_KS},
            }
        report[scope if scope == "overall" else "groups"][scope] = entry

    raw_run = next(r for r in runs if r["name"] == "raw")
    rrf_run = next(r for r in runs if r["name"] == "rrf")
    paired = {"rrf_better": 0, "raw_better": 0, "tie": 0}
    for case in cases:
        r_rank = rrf_run["ranks"][case["query"]]
        w_rank = raw_run["ranks"][case["query"]]
        if r_rank is None and w_rank is None:
            paired["tie"] += 1
        elif r_rank is None:
            paired["raw_better"] += 1
        elif w_rank is None or r_rank < w_rank:
            paired["rrf_better"] += 1
        elif w_rank < r_rank:
            paired["raw_better"] += 1
        else:
            paired["tie"] += 1
    report["paired"] = paired

    for group in groups + ["overall"]:
        subset = cases if group == "overall" else [c for c in cases if c["modality"] == group]
        rel: dict[bool, list[float]] = {True: [], False: []}
        for case in subset:
            for s in case["scored"]:
                rel[_is_match(s["chunk"], case["targets"])].append(s["raw"])
        report["distributions"][group] = {
            "relevant": _quantiles(rel[True]),
            "irrelevant": _quantiles(rel[False]),
        }
    return report


def print_report(report: dict, fusion_params: dict) -> None:
    print(f"\n=== fusion params: {fusion_params} ===")
    for scope, label in [("overall", "OVERALL")] + [
        (g, f"GROUP {g}") for g in report["groups"]
    ]:
        entry = report["overall" if scope == "overall" else "groups"][scope]
        warn = f"  [{entry['warning']}]" if "warning" in entry else ""
        print(f"\n{label} (n={entry['n_cases']}){warn}")
        for name in ("raw", "rrf"):
            m = entry.get(name, {})
            print(
                f"  {name:>4}: MRR={m.get('mrr', 0):.4f} "
                + " ".join(f"R@{k}={m.get(f'recall@{k}', 0):.3f}" for k in RECALL_KS)
            )
    paired = report["paired"]
    print(
        f"\npaired (rrf vs raw): rrf_better={paired['rrf_better']} "
        f"raw_better={paired['raw_better']} tie={paired['tie']}"
    )
    print("\ndistributions (raw score):")
    for group, dist in report["distributions"].items():
        for kind in ("relevant", "irrelevant"):
            d = dist[kind]
            if d:
                print(
                    f"  {group:>8}/{kind:<10}: p10={d['p10']} p50={d['p50']} "
                    f"p90={d['p90']} (n={d['n']})"
                )


async def main_async(args: argparse.Namespace) -> None:
    from ami_rag.core.factory import get_pipeline

    cases = load_cases(
        Path(args.dataset),
        args.query_field,
        args.id_field,
        skip_xfail=args.skip_xfail,
    )
    if not cases:
        print("Bộ test rỗng - không chạy.")
        return
    print(f"{len(cases)} case từ {args.dataset}")

    pipeline = await get_pipeline()
    rerank_func = build_rerank_func(get_settings())
    print(f"collection: {pipeline.collection} | rerank backend: {resolve_rerank_backend(get_settings())}")

    default_params = {
        "rrf_k": args.rrf_k,
        "visual_weight": args.visual_weight,
        "visual_floor": args.visual_floor,
        "text_floor": args.text_floor,
    }
    sweep_params = [default_params]
    if args.sweep:
        for weight in (0.7, 0.8, 0.9, 1.0, 1.2):
            for rrf_k in (10, 30, 60):
                params = dict(default_params, visual_weight=weight, rrf_k=rrf_k)
                if params != default_params:
                    sweep_params.append(params)

    # 1 lần retrieve mỗi query: embed + search + rerank full scores
    for case in cases:
        case["scored"] = await retrieve(pipeline, rerank_func, case["query"], args.top_k)

    runs: list[dict] = []
    raw_ranks: dict[str, int | None] = {}
    for case in cases:
        raw_ranks[case["query"]] = rank_of_targets(raw_ranking(case["scored"]), case["targets"])
    runs.append({"name": "raw", "ranks": raw_ranks})

    all_reports = []
    for params in sweep_params:
        rrf_ranks: dict[str, int | None] = {}
        for case in cases:
            ranked = refuse(case["scored"], **params)
            rrf_ranks[case["query"]] = rank_of_targets(ranked, case["targets"])
        runs_rrf = [runs[0], {"name": "rrf", "ranks": rrf_ranks}]
        report = evaluate(cases, runs_rrf)
        report["fusion_params"] = params
        all_reports.append(report)
        print_report(report, params)

    if args.output:
        payload = {
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "reports": all_reports,
            "cases": [
                {
                    "query": c["query"],
                    "expected_modality": c["modality"],
                    "targets": sorted(c["targets"]),
                    "scored": [
                        {
                            "chunk_id": s["chunk"].get("chunk_id"),
                            "modality": s["chunk"].get("modality") or "text",
                            "raw": s["raw"],
                        }
                        for s in c["scored"]
                    ],
                }
                for c in cases
            ],
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nđã ghi {args.output}")

    await pipeline.embedder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval rerank fusion raw vs rrf")
    parser.add_argument("--dataset", required=True, help="đường dẫn JSONL bộ test")
    parser.add_argument("--query-field", default="query")
    parser.add_argument("--id-field", default="expected_chunk_id")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--visual-weight", type=float, default=0.8)
    parser.add_argument("--visual-floor", type=float, default=0.01)
    parser.add_argument("--text-floor", type=float, default=0.05)
    parser.add_argument("--sweep", action="store_true", help="quét weight x rrf_k")
    parser.add_argument("--skip-xfail", action="store_true", help="bỏ qua các case đánh dấu xfail")
    parser.add_argument("--output", help="ghi kết quả JSON ra file")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
