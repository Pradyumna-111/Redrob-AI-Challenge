# Redrob AI Challenge

Offline candidate ranking submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Output

The final ranked candidate file is:

```bash
submission.csv
```

It contains the required 100 ranked candidates with columns:

```text
candidate_id,rank,score,reasoning
```

## Reproduce

Place the full `candidates.jsonl` file at the repository root, then run:

```bash
python redrob-ai-hackathon/rank.py --candidates candidates.jsonl --out submission.csv
```

Validate:

```bash
python validate_submission.py submission.csv
```

## Approach

See `redrob-ai-hackathon/METHODOLOGY.md`.
