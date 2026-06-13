from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .features import (
    CV_SPEECH_TERMS,
    NON_TECH_TITLES,
    PREFERRED_SKILLS,
    PRODUCT_COMPANIES,
    PROFICIENCY_WEIGHT,
    PROJECT_TERMS,
    REQUIRED_SKILLS,
    SERVICE_COMPANIES,
    TECH_TITLES,
    TIER1_CITIES,
    candidate_text,
    clamp,
    contains_any,
    log_norm,
    months_since,
)


@dataclass(frozen=True)
class RankingResult:
    candidate_id: str
    score: float
    reasoning: str
    components: dict[str, float]


class CandidateRanker:
    """Deterministic recruiter-style ranker for the released Senior AI Engineer JD.

    The model intentionally avoids hosted LLM calls. It combines structured profile
    evidence, career text, product-company context, Redrob behavioral availability,
    and trap penalties into one reproducible score.
    """

    def __init__(self, top_k: int = 100) -> None:
        self.top_k = top_k

    def rank_file(self, candidates_path: str | Path) -> list[RankingResult]:
        heap: list[tuple[float, str, RankingResult]] = []
        for candidate in self._iter_candidates(candidates_path):
            result = self.score_candidate(candidate)
            item = (result.score, result.candidate_id, result)
            if len(heap) < self.top_k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        return [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]

    def write_submission(self, results: Iterable[RankingResult], out_path: str | Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = list(results)[: self.top_k]
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, result in enumerate(rows, start=1):
                writer.writerow(
                    [
                        result.candidate_id,
                        rank,
                        f"{result.score:.6f}",
                        result.reasoning,
                    ]
                )

    def score_candidate(self, candidate: dict[str, Any]) -> RankingResult:
        text = candidate_text(candidate)
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        skill_score, skill_hits, preferred_hits = self._skill_score(skills, text.all_text)
        title_score = self._title_score(text.title, text.career_text)
        experience_score = self._experience_score(float(profile.get("years_of_experience") or 0))
        project_score = self._project_score(text.career_text)
        domain_score = self._domain_score(text, career)
        growth_score = self._growth_score(profile, career)
        behavior_score = self._behavior_score(signals, text)
        location_score = self._location_score(text, signals)
        trap_penalty = self._trap_penalty(candidate, text, skill_hits, project_score, domain_score)

        base = (
            0.24 * title_score
            + 0.20 * skill_score
            + 0.16 * project_score
            + 0.12 * domain_score
            + 0.10 * experience_score
            + 0.08 * behavior_score
            + 0.05 * location_score
            + 0.05 * growth_score
        )
        score = clamp(base * trap_penalty)
        reasoning = self._reason(candidate, score, skill_hits, preferred_hits, project_score, domain_score, behavior_score, trap_penalty)
        return RankingResult(
            candidate_id=candidate["candidate_id"],
            score=score,
            reasoning=reasoning,
            components={
                "title": title_score,
                "skills": skill_score,
                "projects": project_score,
                "domain": domain_score,
                "experience": experience_score,
                "behavior": behavior_score,
                "location": location_score,
                "growth": growth_score,
                "trap_penalty": trap_penalty,
            },
        )

    def _skill_score(self, skills: list[dict[str, Any]], all_text: str) -> tuple[float, list[str], list[str]]:
        total = 0.0
        possible = 0.0
        required_hits: list[str] = []
        preferred_hits: list[str] = []
        by_name = {str(skill.get("name", "")).lower(): skill for skill in skills}

        for term in sorted(REQUIRED_SKILLS):
            possible += 1.0
            direct = by_name.get(term)
            text_hit = term in all_text
            if direct or text_hit:
                weight = 0.70 if text_hit else 0.0
                if direct:
                    weight = PROFICIENCY_WEIGHT.get(str(direct.get("proficiency")), 0.5)
                    weight += min(float(direct.get("duration_months") or 0), 48.0) / 240.0
                    weight += min(float(direct.get("endorsements") or 0), 40.0) / 400.0
                total += clamp(weight)
                required_hits.append(term)

        for term in sorted(PREFERRED_SKILLS):
            direct = by_name.get(term)
            text_hit = term in all_text
            if direct or text_hit:
                weight = 0.04
                if direct:
                    weight += 0.04 * PROFICIENCY_WEIGHT.get(str(direct.get("proficiency")), 0.5)
                total += weight
                preferred_hits.append(term)

        return clamp(total / max(possible, 1.0)), required_hits, preferred_hits

    def _title_score(self, title: str, career_text: str) -> float:
        if "ai engineer" in title or "machine learning engineer" in title or "ml engineer" in title:
            return 1.0
        if "data scientist" in title:
            return 0.88
        if "data engineer" in title or "analytics engineer" in title:
            return 0.78 if contains_any(career_text, {"ml", "model", "feature", "retrieval", "ranking"}) else 0.62
        if "backend engineer" in title or "software engineer" in title or "full stack" in title:
            return 0.72 if contains_any(career_text, {"ml", "ranking", "search", "recommendation", "embedding"}) else 0.52
        if any(non_tech in title for non_tech in NON_TECH_TITLES):
            return 0.08
        if any(tech in title for tech in TECH_TITLES):
            return 0.55
        return 0.25

    def _experience_score(self, years: float) -> float:
        if 5.0 <= years <= 9.0:
            return 1.0
        if 4.0 <= years < 5.0:
            return 0.82
        if 9.0 < years <= 11.0:
            return 0.74
        if 3.0 <= years < 4.0:
            return 0.48
        if 11.0 < years <= 14.0:
            return 0.42
        return 0.18

    def _project_score(self, career_text: str) -> float:
        hits = sum(1 for term in PROJECT_TERMS if term in career_text)
        production = 0.0
        for term in ["deployed", "production", "real users", "scale", "latency", "on-call"]:
            if term in career_text:
                production += 0.08
        return clamp(hits / 7.0 + production)

    def _domain_score(self, text: Any, career: list[dict[str, Any]]) -> float:
        companies = {text.current_company}
        industries = {text.industry}
        service_months = 0
        total_months = 0
        product_months = 0
        for role in career:
            company = str(role.get("company", "")).lower()
            industry = str(role.get("industry", "")).lower()
            months = int(role.get("duration_months") or 0)
            total_months += months
            companies.add(company)
            industries.add(industry)
            if company in SERVICE_COMPANIES:
                service_months += months
            if company in PRODUCT_COMPANIES or "software" in industry or "fintech" in industry or "saas" in text.all_text:
                product_months += months

        product_ratio = product_months / max(total_months, 1)
        service_ratio = service_months / max(total_months, 1)
        score = 0.25 + 0.65 * product_ratio
        if service_ratio >= 0.95 and product_months == 0:
            score -= 0.45
        if "hr" in text.all_text or "recruiter" in text.all_text or "talent" in text.all_text:
            score += 0.10
        if any(company in PRODUCT_COMPANIES for company in companies):
            score += 0.12
        return clamp(score)

    def _growth_score(self, profile: dict[str, Any], career: list[dict[str, Any]]) -> float:
        years = float(profile.get("years_of_experience") or 0)
        senior_words = {"lead", "senior", "principal", "staff", "architect", "founding"}
        lead_evidence = 0
        short_hops = 0
        for role in career:
            title = str(role.get("title", "")).lower()
            desc = str(role.get("description", "")).lower()
            months = int(role.get("duration_months") or 0)
            if any(word in title or word in desc for word in senior_words) or "managed a team" in desc or "mentoring" in desc:
                lead_evidence += 1
            if months and months < 18:
                short_hops += 1
        score = 0.45 + min(lead_evidence, 3) * 0.15
        if 5 <= years <= 9:
            score += 0.15
        if short_hops >= 3:
            score -= 0.20
        return clamp(score)

    def _behavior_score(self, signals: dict[str, Any], text: Any) -> float:
        response = float(signals.get("recruiter_response_rate") or 0.0)
        response_time = float(signals.get("avg_response_time_hours") or 999.0)
        active_months = months_since(signals.get("last_active_date"))
        notice = float(signals.get("notice_period_days") or 180.0)
        profile = float(signals.get("profile_completeness_score") or 0.0) / 100.0
        github = float(signals.get("github_activity_score") if signals.get("github_activity_score") is not None else -1)
        github_norm = 0.0 if github < 0 else github / 100.0
        saved = log_norm(float(signals.get("saved_by_recruiters_30d") or 0), 20)
        views = log_norm(float(signals.get("profile_views_received_30d") or 0), 100)
        interview = float(signals.get("interview_completion_rate") or 0.0)
        offer = signals.get("offer_acceptance_rate")
        offer_score = 0.45 if offer in (None, -1) else float(offer)
        active_score = clamp(1 - active_months / 6.0)
        response_time_score = clamp(1 - response_time / 168.0)
        notice_score = clamp(1 - notice / 120.0)
        verified = 0.0
        for key in ["verified_email", "verified_phone", "linkedin_connected"]:
            verified += 0.04 if signals.get(key) else 0.0
        open_to_work = 0.12 if signals.get("open_to_work_flag") else 0.0
        return clamp(
            0.20 * response
            + 0.14 * response_time_score
            + 0.16 * active_score
            + 0.12 * notice_score
            + 0.12 * profile
            + 0.08 * github_norm
            + 0.06 * saved
            + 0.04 * views
            + 0.08 * interview
            + 0.04 * offer_score
            + verified
            + open_to_work
        )

    def _location_score(self, text: Any, signals: dict[str, Any]) -> float:
        if "india" not in text.country:
            return 0.18 if signals.get("willing_to_relocate") else 0.05
        if "pune" in text.location or "noida" in text.location:
            return 1.0
        if any(city in text.location for city in TIER1_CITIES):
            return 0.86
        if signals.get("willing_to_relocate"):
            return 0.74
        return 0.55

    def _trap_penalty(
        self,
        candidate: dict[str, Any],
        text: Any,
        skill_hits: list[str],
        project_score: float,
        domain_score: float,
    ) -> float:
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        signals = candidate.get("redrob_signals", {})
        penalty = 1.0

        expert_zero = sum(
            1
            for skill in skills
            if str(skill.get("proficiency")) == "expert" and int(skill.get("duration_months") or 0) <= 1
        )
        if expert_zero >= 3:
            penalty *= 0.25

        ai_keyword_count = len(skill_hits)
        non_tech_title = any(title in text.title for title in NON_TECH_TITLES)
        if non_tech_title and ai_keyword_count >= 5:
            penalty *= 0.28
        elif non_tech_title:
            penalty *= 0.55

        if ai_keyword_count >= 8 and project_score < 0.30:
            penalty *= 0.65

        years = float(profile.get("years_of_experience") or 0)
        if years < 2 and ai_keyword_count >= 6:
            penalty *= 0.45

        if contains_any(text.all_text, CV_SPEECH_TERMS) and not contains_any(text.all_text, {"nlp", "retrieval", "ranking", "search"}):
            penalty *= 0.62

        service_months = sum(
            int(role.get("duration_months") or 0)
            for role in career
            if str(role.get("company", "")).lower() in SERVICE_COMPANIES
        )
        total_months = sum(int(role.get("duration_months") or 0) for role in career)
        if total_months > 0 and service_months / total_months > 0.95 and domain_score < 0.35:
            penalty *= 0.62

        if months_since(signals.get("last_active_date")) > 6 and float(signals.get("recruiter_response_rate") or 0.0) < 0.12:
            penalty *= 0.55

        if float(signals.get("notice_period_days") or 0.0) >= 150:
            penalty *= 0.82

        if "langchain" in text.all_text and project_score < 0.35 and years < 5:
            penalty *= 0.70

        return clamp(penalty, 0.05, 1.0)

    def _reason(
        self,
        candidate: dict[str, Any],
        score: float,
        skill_hits: list[str],
        preferred_hits: list[str],
        project_score: float,
        domain_score: float,
        behavior_score: float,
        trap_penalty: float,
    ) -> str:
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        title = profile.get("current_title", "candidate")
        years = float(profile.get("years_of_experience") or 0.0)
        skills = ", ".join(skill_hits[:4]) if skill_hits else "limited explicit AI/search skills"
        positives = [
            f"{title} with {years:.1f} years",
            f"evidence in {skills}",
        ]
        if project_score >= 0.70:
            positives.append("career text shows ranking/search or retrieval work")
        elif project_score >= 0.40:
            positives.append("some production ML/search-adjacent project evidence")
        if domain_score >= 0.70:
            positives.append("product-company/domain context fits the JD")
        if preferred_hits:
            positives.append(f"supporting stack includes {', '.join(preferred_hits[:3])}")

        concerns: list[str] = []
        response = float(signals.get("recruiter_response_rate") or 0.0)
        notice = int(signals.get("notice_period_days") or 0)
        if behavior_score < 0.45:
            concerns.append(f"availability signals are mixed: response rate {response:.2f}")
        if notice >= 90:
            concerns.append(f"notice period is {notice} days")
        if trap_penalty < 0.8:
            concerns.append("profile has keyword/trap risk, so it was down-weighted")
        if score < 0.45 and not concerns:
            concerns.append("below the ideal Senior AI Engineer bar")

        reason = "; ".join(positives)
        if concerns:
            reason += ". Concern: " + "; ".join(concerns[:2])
        return reason[:700]

    def _iter_candidates(self, candidates_path: str | Path) -> Iterable[dict[str, Any]]:
        path = Path(candidates_path)
        with path.open("r", encoding="utf-8") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                data = json.load(f)
                yield from data
            else:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
