
"""
Greedy team formation that balances:
- Fit to project (candidate-project similarity from M2)
- Diversity / complementarity (penalize similarity to current team from M1 matrix)
- Skill coverage (marginal gain in required skills)

 GenAI touch:
- LLM reranker among top-K candidates at each step (only if OPENAI_API_KEY is set)
  to consider soft-skill/risk signals from provided fields.
"""

from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from shared.interfaces import CandidateProfile, ProjectDescription

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional dependency: OpenAI
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


def _norm_skill(s: str) -> str:
    return (s or "").strip().lower()

def _team_skill_set(team: List[CandidateProfile]) -> set:
    s = set()
    for m in team:
        s.update([_norm_skill(x) for x in (m.skills or []) if x])
    return s

def coverage_gain(team: List[CandidateProfile], candidate: CandidateProfile, required_skills: List[str]) -> float:
    """Marginal gain in required skill coverage by adding candidate (0..1)."""
    req = set([_norm_skill(x) for x in (required_skills or []) if x])
    if not req:
        return 0.0

    team_sk = _team_skill_set(team)
    cand_sk = set([_norm_skill(x) for x in (candidate.skills or []) if x])

    before = len(req & team_sk)
    after = len(req & (team_sk | cand_sk))
    return (after - before) / max(1, len(req))

def avg_similarity_to_team(
    candidate_id: str,
    team_ids: List[str],
    sim_matrix: np.ndarray,
    candidate_ids_order: List[str],
) -> float:
    """Average similarity between candidate and current team members. Returns 0 if missing."""
    if not team_ids:
        return 0.0
    idx = {cid: i for i, cid in enumerate(candidate_ids_order)}
    if candidate_id not in idx:
        return 0.0
    ci = idx[candidate_id]
    sims = []
    for tid in team_ids:
        if tid in idx:
            sims.append(float(sim_matrix[ci, idx[tid]]))
    return float(sum(sims) / len(sims)) if sims else 0.0

def _safe_fit(project_similarities: Dict[str, float], cid: str) -> float:
    try:
        return float(project_similarities.get(cid, 0.0))
    except Exception:
        return 0.0



#LLM reranker (bounded)


def llm_choose_best_candidate(
    team: List[CandidateProfile],
    shortlist: List[CandidateProfile],
    project: ProjectDescription,
    project_similarities: Dict[str, float],
    sim_matrix: np.ndarray,
    candidate_ids_order: List[str],
    model: str = "gpt-4.1-mini",
) -> Optional[str]:
    """
    Pick ONE candidate from shortlist only. Returns candidate_id or None (fallback).
    Auto-disabled if OPENAI_API_KEY missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)

    team_ids = [m.id for m in team]
    missing_required = sorted(
        set([_norm_skill(x) for x in (project.required_skills or []) if x]) - _team_skill_set(team)
    )

    def brief(m: CandidateProfile) -> Dict:
        return {
            "id": m.id,
            "role": m.role,
            "availability_hours": m.availability_hours,
            "collaboration_style": m.collaboration_style,
            "skills": (m.skills or [])[:20],
            "fit": _safe_fit(project_similarities, m.id),
        }

    def cand_brief(c: CandidateProfile) -> Dict:
        team_sk = _team_skill_set(team)
        add_req = sorted(
            set([_norm_skill(x) for x in (project.required_skills or []) if x])
            & (set([_norm_skill(x) for x in (c.skills or []) if x]) - team_sk)
        )
        return {
            **brief(c),
            "avg_similarity_to_team": avg_similarity_to_team(c.id, team_ids, sim_matrix, candidate_ids_order),
            "adds_required_skills": add_req,
        }

    payload = {
        "project": {
            "title": project.title,
            "required_roles": project.required_roles,
            "required_skills": project.required_skills,
            "priority_skills": project.priority_skills,
            "team_size": project.team_size,
        },
        "current_team": [brief(m) for m in team],
        "missing_required_skills": missing_required,
        "shortlist": [cand_brief(c) for c in shortlist],
        "rules": [
            "Pick exactly ONE candidate from shortlist.",
            "Prefer candidates that add missing required skills.",
            "Prefer lower avg_similarity_to_team (less redundancy).",
            "Consider availability and role balance.",
            "Do not invent skills or facts; use only provided data."
        ],
        "output_format": {
            "best_candidate_id": "string (must be in shortlist)",
            "reason": "short string",
            "risks": ["short strings"]
        }
    }

    prompt = (
        "You are a team composition critic. Choose the best next candidate.\n"
        "Return JSON ONLY.\n"
        f"{json.dumps(payload, indent=2)}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        txt = resp.choices[0].message.content.strip()
        a, b = txt.find("{"), txt.rfind("}")
        if a == -1 or b == -1:
            return None
        obj = json.loads(txt[a:b+1])
        chosen = obj.get("best_candidate_id")
        if chosen and chosen in {c.id for c in shortlist}:
            return str(chosen)
        return None
    except Exception as e:
        logger.info(f"LLM rerank skipped (fallback). Reason: {e}")
        return None


# Team Builder

class TeamBuilderGenAI:
    """Greedy team builder with optional LLM reranking among top-K each step."""

    def __init__(
        self,
        alpha: float = 0.55,   # fit
        beta: float = 0.25,    # diversity
        gamma: float = 0.20,   # coverage gain
        fit_threshold: float = 0.0,
        rerank_top_k: int = 5,
        use_llm_rerank: bool = True,
        llm_model: str = "gpt-4.1-mini",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fit_threshold = fit_threshold
        self.rerank_top_k = rerank_top_k

        #never crash if no key; just disable.
        self.use_llm_rerank = bool(use_llm_rerank and os.getenv("OPENAI_API_KEY") and OpenAI is not None)
        self.llm_model = llm_model

        if use_llm_rerank and not self.use_llm_rerank:
            logger.info("LLM reranker disabled (no OPENAI_API_KEY or openai pkg missing).")

    def _score_next(
        self,
        team: List[CandidateProfile],
        cand: CandidateProfile,
        project: ProjectDescription,
        sim_matrix: np.ndarray,
        candidate_ids_order: List[str],
        project_similarities: Dict[str, float],
    ) -> float:
        fit = _safe_fit(project_similarities, cand.id)
        gain_cov = coverage_gain(team, cand, project.required_skills)

        team_ids = [m.id for m in team]
        avg_sim = avg_similarity_to_team(cand.id, team_ids, sim_matrix, candidate_ids_order)
        diversity = 1.0 - avg_sim

        return float((self.alpha * fit) + (self.gamma * gain_cov) + (self.beta * diversity))

    def build_team_greedy(
        self,
        profiles: List[CandidateProfile],
        project: ProjectDescription,
        candidate_sim_matrix: np.ndarray,
        candidate_ids: List[str],
        project_similarities: Dict[str, float],
    ) -> List[CandidateProfile]:

        if not profiles:
            return []

        # gentle fit filter
        filtered = [p for p in profiles if _safe_fit(project_similarities, p.id) >= self.fit_threshold]
        if not filtered:
            filtered = list(profiles)

        # seed = best fit
        filtered.sort(key=lambda p: _safe_fit(project_similarities, p.id), reverse=True)
        team = [filtered[0]]
        remaining = [p for p in filtered[1:]]

        while len(team) < int(project.team_size) and remaining:
            scored: List[Tuple[float, CandidateProfile]] = []
            for cand in remaining:
                scored.append((self._score_next(team, cand, project, candidate_sim_matrix, candidate_ids, project_similarities), cand))
            scored.sort(key=lambda x: x[0], reverse=True)

            shortlist = [c for _, c in scored[: max(1, self.rerank_top_k)]]

            chosen_id = None
            if self.use_llm_rerank and len(shortlist) >= 2:
                chosen_id = llm_choose_best_candidate(
                    team=team,
                    shortlist=shortlist,
                    project=project,
                    project_similarities=project_similarities,
                    sim_matrix=candidate_sim_matrix,
                    candidate_ids_order=candidate_ids,
                    model=self.llm_model,
                )

            if chosen_id:
                chosen = next((c for c in remaining if c.id == chosen_id), None)
                if chosen is None:
                    chosen = scored[0][1]
            else:
                chosen = scored[0][1]

            team.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return team



# Interface function 

def build_teams(
    profiles: List[CandidateProfile],
    project: ProjectDescription,
    candidate_sim_matrix: np.ndarray,
    candidate_ids: List[str],
    project_similarities: Dict[str, float],
) -> List[List[CandidateProfile]]:
    """
    interface: returns List[List[CandidateProfile]].
    """
    builder = TeamBuilderGenAI()
    team = builder.build_team_greedy(
        profiles=profiles,
        project=project,
        candidate_sim_matrix=candidate_sim_matrix,
        candidate_ids=candidate_ids,
        project_similarities=project_similarities,
    )
    return [team] if team else []