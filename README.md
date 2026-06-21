# Redrob AI Challenge - Explainable Candidate Ranking Engine

An offline, deterministic candidate ranking system built for the Redrob Intelligent Candidate Discovery & Ranking Challenge. The system evaluates 100,000 candidate profiles against the released Senior AI Engineer - Founding Team job description and produces an explainable top-100 shortlist.

The implementation is designed around the challenge's real production constraints: CPU-only execution, no network access, no hosted LLM calls during ranking, less than five minutes of runtime, and a reproducible CSV submission.

## Problem

Traditional applicant tracking systems often reward literal keyword overlap. That can elevate profiles with long AI skill lists even when their career history does not show relevant production experience. It can also overlook candidates whose experience demonstrates retrieval, ranking, recommendation, or evaluation work without using the newest framework names.

This project addresses that gap by combining structured profile evidence, career-history context, product-domain exposure, behavioral availability, location, and explicit suspicious-profile penalties.

## Solution Overview

For every candidate, the ranker:

1. Normalizes profile, career, education, certification, and skill text.
2. Measures role and title alignment with the Senior AI Engineer JD.
3. Scores required and preferred skills using proficiency, duration, endorsements, and career-text corroboration.
4. Looks for production evidence involving retrieval, search, ranking, recommendation, evaluation metrics, deployment, scale, latency, and ownership.
5. Evaluates experience band, product-company context, career growth, location, and Redrob behavioral signals.
6. Applies multiplicative penalties for keyword stuffing and suspicious or poor-fit profiles.
7. Keeps the best 100 candidates in a bounded heap and sorts them deterministically.
8. Writes a validator-compatible CSV with candidate-specific reasoning.

This is an explainable hybrid ranking engine, not a hosted generative-AI workflow. Avoiding external model calls is deliberate: it keeps ranking fast, auditable, deterministic, and compliant with the challenge constraints.

## Ranking Formula

The base score is a weighted combination of eight components:

| Component | Weight | What it measures |
| --- | ---: | --- |
| Role/title fit | 24% | Alignment with AI, ML, data, backend, search, and ranking roles |
| Skill fit | 20% | Required/preferred skills with proficiency and evidence quality |
| Project relevance | 16% | Retrieval, ranking, recommendation, evaluation, and production signals |
| Product/domain fit | 12% | Product-company, software, fintech, SaaS, HR-tech, and talent context |
| Experience fit | 10% | Preference for the JD's 5-9 year range while allowing strong outliers |
| Behavioral availability | 8% | Activity, responsiveness, notice period, open-to-work, verification, and engagement |
| Location fit | 5% | Pune/Noida, tier-1 Indian cities, and relocation readiness |
| Career growth | 5% | Seniority, leadership evidence, tenure, and job-hop pattern |

The final score is:

```text
final_score = clamp(weighted_base_score * trap_penalty, 0, 1)
```

### Evidence-aware skill scoring

The skill component does more than count names. Direct skill entries are weighted using:

- proficiency: beginner, intermediate, advanced, or expert
- duration of use, capped to prevent extreme values dominating
- endorsements, also capped
- corroborating mentions in career history and profile text

The required skill vocabulary includes Python, NLP, embeddings, retrieval, ranking, search, recommendation systems, vector databases, FAISS, Elasticsearch, OpenSearch, LLMs, LoRA, fine-tuning, and RAG. Preferred supporting skills include MLOps, PyTorch, XGBoost, Docker, Kubernetes, AWS, GCP, Kafka, Spark, Redis, and related production tooling.

### Behavioral availability

The ranker incorporates Redrob platform signals because a strong candidate who is inactive or unreachable is less actionable for a recruiter. Signals include:

- last-active recency and open-to-work status
- recruiter response rate and response time
- profile completeness and verification
- notice period and willingness to relocate
- GitHub activity, recruiter saves, profile views, and search appearances
- interview completion and historical offer acceptance

### Trap and quality controls

The score is reduced for profiles that exhibit one or more high-risk patterns:

- non-technical titles paired with unusually dense AI keyword lists
- many expert skills with near-zero stated duration
- numerous AI keywords without production project evidence
- very junior profiles with implausibly broad AI expertise
- computer-vision or speech-only backgrounds without NLP/IR evidence
- service-only career histories without product or production-search evidence
- long inactivity combined with very low recruiter response
- 150+ day notice periods
- junior LangChain-only experience without deeper ML/search evidence

These penalties are multiplicative, so multiple inconsistencies compound instead of being hidden by a large keyword score.

## Explainability

Every shortlisted candidate receives a concise explanation generated from the same features used in scoring. Explanations can include:

- current title and total years of experience
- relevant AI/search skills found in the profile
- production retrieval/ranking evidence
- product-company or domain alignment
- supporting engineering stack
- concerns such as notice period, availability, or suspicious-profile risk

The system does not invent facts or use a language model to generate unsupported claims. Reasoning is assembled only from parsed candidate fields and computed component thresholds.

## Architecture

```text
candidates.jsonl
      |
      v
Streaming JSONL reader
      |
      v
Profile normalization and evidence extraction
      |
      +--> role/title fit
      +--> skills and project evidence
      +--> domain, experience, growth, location
      +--> Redrob behavioral availability
      +--> trap and quality penalties
      |
      v
Weighted score + multiplicative penalty
      |
      v
Top-100 min-heap and deterministic sort
      |
      v
submission.csv + candidate-specific reasoning
```

The full dataset is processed as a stream, so memory use does not scale with all 100,000 candidate objects. Only the current candidate and the top-100 heap need to remain in memory.

## Repository Structure

```text
.
|-- README.md
|-- candidate_schema.json
|-- sample_candidates.json
|-- submission.csv
|-- submission_metadata.yaml
|-- validate_submission.py
`-- redrob-ai-hackathon/
    |-- METHODOLOGY.md
    |-- README.md
    |-- rank.py
    |-- requirements.txt
    |-- src/redrob_ranker/
    |   |-- __init__.py
    |   |-- features.py
    |   `-- ranker.py
    `-- tests/
        `-- test_ranker.py
```

The full `candidates.jsonl` dataset is intentionally excluded from Git because it is large and organizer-provided.

## Requirements

- Python 3.10 or newer
- No third-party runtime dependencies
- CPU only
- No GPU
- No network access during ranking

## Run the Ranker

Place `candidates.jsonl` at the repository root, then run:

```bash
python redrob-ai-hackathon/rank.py --candidates candidates.jsonl --out submission.csv
```

Optional arguments:

```bash
python redrob-ai-hackathon/rank.py \
  --candidates sample_candidates.json \
  --out sample_ranked_submission.csv \
  --top-k 50
```

## Validate the Submission

```bash
python validate_submission.py submission.csv
```

A valid final file contains exactly 100 data rows with this column order:

```text
candidate_id,rank,score,reasoning
```

## Run Tests

```bash
python -m unittest discover -s redrob-ai-hackathon/tests
```

The focused tests verify two central behaviors:

- a production ML profile outranks a keyword-stuffed non-technical profile
- an active and responsive candidate outranks an otherwise similar inactive candidate

## Results

- Full candidate pool processed: 100,000 profiles
- Output: top 100 ranked candidates
- Observed full-run time on the development machine: approximately 82 seconds
- Submission format: validated successfully with `validate_submission.py`
- Runtime dependencies: Python standard library only
- Ranking-time network calls: none
- Ranking-time GPU use: none

The top-ranked candidates are dominated by Lead AI Engineer, Senior/Staff Machine Learning Engineer, Applied ML Engineer, AI Engineer, Senior Data Scientist, and recommendation/search-oriented profiles with production evidence.

## Reproducibility and Determinism

- The same input produces the same scores and ordering.
- Candidate IDs provide deterministic tie-breaking.
- Scores are written in non-increasing rank order.
- The ranking step has no hidden APIs, manual edits, or external state.
- The final CSV is generated directly from the code in this repository.

## Demo and Submission Assets

- GitHub repository: https://github.com/Pradyumna-111/Redrob-AI-Challenge
- Colab sandbox: https://colab.research.google.com/drive/1inKETpmgo6F6ZinZ2OXOFrxhDlB2J-7I?usp=sharing
- Final output: `submission.csv`
- Detailed methodology: `redrob-ai-hackathon/METHODOLOGY.md`

## Limitations and Future Improvements

The current implementation uses explicit domain knowledge rather than trained semantic embeddings. This is a deliberate baseline that satisfies the offline compute budget and remains easy to audit. Future iterations could add:

- precomputed local sentence-transformer embeddings
- BM25 or TF-IDF retrieval as a first-stage candidate generator
- a learned-to-rank model trained on recruiter relevance labels
- calibration against NDCG, MRR, MAP, and online engagement feedback
- fairness monitoring and protected-attribute audits
- automated feature-drift and ranking-regression reports

Any such extension should preserve the current guarantees: no unsupported reasoning, deterministic offline reproduction, bounded runtime, and transparent signal attribution.
