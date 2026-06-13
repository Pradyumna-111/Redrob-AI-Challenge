from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redrob_ranker import CandidateRanker


def base_candidate(candidate_id: str, title: str, years: float, skill_names: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "profile": {
            "current_title": title,
            "headline": title,
            "summary": "Built production retrieval and ranking systems for recruiters.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": years,
            "current_company": "CRED",
            "current_industry": "Fintech",
        },
        "career_history": [
            {
                "company": "CRED",
                "title": title,
                "duration_months": 72,
                "industry": "Fintech",
                "description": "Shipped embeddings based semantic search, vector retrieval, ranking evaluation with NDCG and A/B tests to production users.",
            }
        ],
        "education": [],
        "skills": [
            {"name": name, "proficiency": "advanced", "endorsements": 20, "duration_months": 36}
            for name in skill_names
        ],
        "redrob_signals": {
            "profile_completeness_score": 92,
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 30,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12,
            "saved_by_recruiters_30d": 8,
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "github_activity_score": 70,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.6,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


class CandidateRankerTests(unittest.TestCase):
    def test_product_ml_profile_beats_keyword_stuffed_non_tech_profile(self) -> None:
        ranker = CandidateRanker()
        strong = base_candidate(
            "CAND_0000001",
            "AI Engineer",
            6.5,
            ["Python", "NLP", "Qdrant", "Embeddings", "Ranking", "Fine-tuning LLMs"],
        )
        stuffed = base_candidate(
            "CAND_0000002",
            "Marketing Manager",
            7.0,
            ["Python", "NLP", "Qdrant", "Embeddings", "Ranking", "Fine-tuning LLMs", "Milvus", "LoRA"],
        )
        stuffed["career_history"][0]["description"] = "Managed campaigns and wrote AI-assisted content."

        self.assertGreater(ranker.score_candidate(strong).score, ranker.score_candidate(stuffed).score)

    def test_inactive_candidate_is_penalized(self) -> None:
        ranker = CandidateRanker()
        active = base_candidate("CAND_0000003", "ML Engineer", 6.0, ["Python", "NLP", "Milvus", "Ranking"])
        inactive = base_candidate("CAND_0000004", "ML Engineer", 6.0, ["Python", "NLP", "Milvus", "Ranking"])
        inactive["redrob_signals"]["last_active_date"] = "2025-06-01"
        inactive["redrob_signals"]["recruiter_response_rate"] = 0.05

        self.assertGreater(ranker.score_candidate(active).score, ranker.score_candidate(inactive).score)


if __name__ == "__main__":
    unittest.main()
