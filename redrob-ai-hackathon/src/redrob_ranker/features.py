from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 6, 13)

SERVICE_COMPANIES = {
    "accenture",
    "capgemini",
    "cognizant",
    "infosys",
    "mindtree",
    "tcs",
    "wipro",
}

PRODUCT_COMPANIES = {
    "cred",
    "freshworks",
    "flipkart",
    "hooli",
    "inmobi",
    "paytm",
    "razorpay",
    "swiggy",
    "zerodha",
}

TIER1_CITIES = {
    "bangalore",
    "bengaluru",
    "delhi",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "mumbai",
    "noida",
    "pune",
}

TECH_TITLES = {
    "ai engineer",
    "machine learning engineer",
    "ml engineer",
    "data scientist",
    "data engineer",
    "backend engineer",
    "software engineer",
    "full stack developer",
    "analytics engineer",
}

NON_TECH_TITLES = {
    "accountant",
    "civil engineer",
    "content writer",
    "customer support",
    "graphic designer",
    "hr manager",
    "marketing manager",
    "mechanical engineer",
    "operations manager",
    "sales executive",
}

REQUIRED_SKILLS = {
    "python",
    "nlp",
    "embeddings",
    "sentence-transformers",
    "vector search",
    "semantic search",
    "retrieval",
    "ranking",
    "recommendation systems",
    "recommender systems",
    "search",
    "qdrant",
    "milvus",
    "pinecone",
    "weaviate",
    "faiss",
    "elasticsearch",
    "opensearch",
    "llms",
    "fine-tuning llms",
    "lora",
    "rag",
}

PREFERRED_SKILLS = {
    "airflow",
    "apache beam",
    "aws",
    "bentoml",
    "databricks",
    "docker",
    "feature engineering",
    "gcp",
    "kafka",
    "kubernetes",
    "mlops",
    "postgres",
    "pytorch",
    "redis",
    "spark",
    "statistical modeling",
    "terraform",
    "xgboost",
}

PROJECT_TERMS = {
    "a/b test",
    "ab test",
    "bm25",
    "candidate",
    "embedding",
    "hybrid retrieval",
    "index refresh",
    "llm",
    "map",
    "mrr",
    "ndcg",
    "offline benchmark",
    "rank",
    "recommendation",
    "recruiter",
    "retrieval",
    "search",
    "semantic",
    "vector",
}

CV_SPEECH_TERMS = {
    "computer vision",
    "image classification",
    "gan",
    "gans",
    "robotics",
    "speech recognition",
    "tts",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.35,
    "intermediate": 0.60,
    "advanced": 0.82,
    "expert": 1.00,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_text(value: str | None) -> str:
    value = value or ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9+#./-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def months_since(value: str | None) -> float:
    parsed = parse_date(value)
    if not parsed:
        return 999.0
    return max(0.0, (REFERENCE_DATE - parsed).days / 30.4375)


def log_norm(value: float, cap: float) -> float:
    if value <= 0:
        return 0.0
    return clamp(math.log1p(value) / math.log1p(cap))


@dataclass(frozen=True)
class CandidateText:
    title: str
    current_company: str
    location: str
    country: str
    industry: str
    summary: str
    all_text: str
    career_text: str
    skill_names: list[str]


def candidate_text(candidate: dict[str, Any]) -> CandidateText:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    certifications = candidate.get("certifications", [])

    title = normalize_text(profile.get("current_title"))
    current_company = normalize_text(profile.get("current_company"))
    location = normalize_text(profile.get("location"))
    country = normalize_text(profile.get("country"))
    industry = normalize_text(profile.get("current_industry"))
    summary = normalize_text(profile.get("summary"))
    skill_names = [normalize_text(skill.get("name")) for skill in skills]
    career_bits = []
    for role in career:
        career_bits.extend(
            [
                normalize_text(role.get("title")),
                normalize_text(role.get("company")),
                normalize_text(role.get("industry")),
                normalize_text(role.get("description")),
            ]
        )
    education_bits = [
        normalize_text(f"{edu.get('degree', '')} {edu.get('field_of_study', '')} {edu.get('tier', '')}")
        for edu in education
    ]
    cert_bits = [
        normalize_text(f"{cert.get('name', '')} {cert.get('issuer', '')}")
        for cert in certifications
    ]
    career_text = " ".join(career_bits)
    all_text = " ".join(
        [
            title,
            normalize_text(profile.get("headline")),
            summary,
            current_company,
            industry,
            career_text,
            " ".join(skill_names),
            " ".join(education_bits),
            " ".join(cert_bits),
        ]
    )
    return CandidateText(
        title=title,
        current_company=current_company,
        location=location,
        country=country,
        industry=industry,
        summary=summary,
        all_text=all_text,
        career_text=career_text,
        skill_names=skill_names,
    )
