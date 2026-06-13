# Redrob AI Hackathon Ranker

Offline, deterministic ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Method

The ranker is built for the released Senior AI Engineer founding-team JD. It scores every candidate with recruiter-style components:

- role/title fit for AI, ML, data, backend, search, and ranking work
- evidence of production retrieval, recommendation, ranking, semantic search, evaluation, and vector systems
- skill quality using proficiency, duration, endorsements, and career-text evidence
- product-company/domain fit, with a penalty for service-only careers where the JD says fit is poor
- experience-band fit, favoring the 5-9 year range but allowing strong outliers
- Redrob behavioral availability: recency, response rate, response time, notice period, open-to-work, verification, GitHub, saves, and interview completion
- trap resistance for keyword-stuffed non-technical profiles, impossible expert-zero-duration skills, inactive profiles, and CV/speech-only AI backgrounds without NLP/IR evidence

No network, hosted LLM, GPU, or heavyweight dependency is required during ranking.

## Run

From this folder:

```bash
python rank.py --candidates ../candidates.jsonl --out ../submission.csv
```

Validate from the repository root:

```bash
python validate_submission.py submission.csv
```

## Tests

```bash
python -m unittest discover -s tests
```

## Files

- `rank.py` - CLI entry point
- `src/redrob_ranker/features.py` - constants and feature helpers
- `src/redrob_ranker/ranker.py` - scoring, ranking, CSV writing, and explanations
- `tests/test_ranker.py` - smoke tests for key ranking behavior
