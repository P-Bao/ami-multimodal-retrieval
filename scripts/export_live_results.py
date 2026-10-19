"""Chạy toàn bộ retrieval cases qua Live API (/v2/rag/) và xuất kết quả ra file JSON.

Usage:
    python scripts/export_live_results.py --output result_live_raw.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_retrieval_recall_live import _load_cases, run_case


def main():
    parser = argparse.ArgumentParser(description="Export live API retrieval results")
    parser.add_argument("--output", required=True, help="Đường dẫn file JSON kết quả đầu ra")
    parser.add_argument("--top-k", type=int, default=5, help="Số document top_k")
    args = parser.parse_args()

    cases = _load_cases()
    print(f"Bắt đầu chạy {len(cases)} cases qua Live API (top_k={args.top_k})...")

    results = []
    hits = {"text": 0, "table": 0, "image": 0}
    totals = {"text": 0, "table": 0, "image": 0}

    for idx, case in enumerate(cases, start=1):
        res = run_case(case, top_k=args.top_k)
        results.append(res)
        mod = case["modality"]
        totals[mod] = totals.get(mod, 0) + 1
        if res.get("hit"):
            hits[mod] = hits.get(mod, 0) + 1
        status = f"HIT  (rank {res.get('rank')})" if res.get("hit") else "MISS"
        print(f"[{idx:2d}/{len(cases):2d}] [{mod:5s}] {case['id']:30s} -> {status} (fusion={res.get('fusion')})")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi kết quả ra {out_path.resolve()}")
    print("Tóm tắt Hit@5:")
    for mod in ("text", "table", "image"):
        print(f"  {mod.upper():<6}: {hits.get(mod, 0)}/{totals.get(mod, 0)}")


if __name__ == "__main__":
    main()
