from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redrob_ranker import CandidateRanker


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates for the Senior AI Engineer JD.")
    parser.add_argument("--candidates", default="../candidates.jsonl", help="Path to candidates.jsonl or sample_candidates.json.")
    parser.add_argument("--out", default="../submission.csv", help="Output CSV path.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to emit.")
    args = parser.parse_args()

    ranker = CandidateRanker(top_k=args.top_k)
    results = ranker.rank_file(args.candidates)
    ranker.write_submission(results, args.out)
    print(f"Wrote {min(args.top_k, len(results))} ranked candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
