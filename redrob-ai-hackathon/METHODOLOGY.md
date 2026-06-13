# Methodology

This solution ranks candidates for Redrob AI's Senior AI Engineer founding-team role using a deterministic offline scoring model. The JD makes clear that keyword overlap is a trap, so the ranker separates "AI words listed as skills" from evidence that the candidate shipped search, retrieval, ranking, recommendation, or evaluation systems in production.

## Scoring Components

- **Title and role fit:** prioritizes AI Engineer, ML Engineer, Applied ML Engineer, Data Scientist, Recommendation Systems Engineer, and software/backend profiles with ML-search evidence.
- **Skill fit:** scores required skills such as Python, NLP, embeddings, retrieval, ranking, vector databases, FAISS, Milvus, Qdrant, OpenSearch, Elasticsearch, LLMs, LoRA, and RAG. Skill scores use proficiency, duration, endorsements, and career-text corroboration.
- **Project relevance:** looks for production signals around semantic search, hybrid retrieval, vector search, recommendation systems, ranking metrics, NDCG/MRR/MAP, A/B testing, scale, latency, deployment, and on-call ownership.
- **Domain fit:** rewards product-company and software/fintech/SaaS experience, and down-weights service-only career histories when the profile lacks strong product or production-search evidence.
- **Experience fit:** favors the JD's 5-9 year range while allowing strong near-band candidates.
- **Behavioral availability:** incorporates Redrob signals for recency, response rate, response time, notice period, open-to-work, verification, GitHub activity, recruiter saves, profile completeness, and interview completion.
- **Location fit:** favors Pune/Noida and tier-1 Indian cities while allowing relocation candidates.
- **Trap resistance:** applies penalties for keyword-stuffed non-technical profiles, impossible expert skills with near-zero duration, inactive profiles with very low recruiter response, service-only profiles without product evidence, CV/speech-only AI backgrounds without NLP/IR evidence, and junior LangChain-only profiles.

## Reproducibility

The ranking step uses only the Python standard library. It streams `candidates.jsonl`, keeps a top-100 heap in memory, writes `submission.csv`, and makes no network or GPU calls. On the provided machine it processed the full candidate pool in about 82 seconds.

```bash
python redrob-ai-hackathon/rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

## Review Notes

Reasoning strings are generated from the same evidence used for scoring. Each row mentions concrete profile facts such as title, years of experience, relevant skills, project/domain evidence, supporting stack, and concerns like notice period or availability where applicable.
