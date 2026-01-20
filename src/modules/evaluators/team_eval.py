from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _parse_availability_min(avail_raw: Any) -> float:
    """
    Returns a minimum availability in hours/week.

    Examples:
      "10–20" -> 10
      "10-20" -> 10
      "Unknown" / None -> 0
      15 -> 15
    """
    if avail_raw is None:
        return 0.0
    s = str(avail_raw).strip()
    if not s or s.lower() == "unknown":
        return 0.0
    s = s.replace("–", "-")

    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return 0.0
   
    try:
        return float(nums[0])
    except Exception:
        return 0.0


def _cand_items(profile: dict) -> set[str]:
    tech = profile.get("technical", {}) or {}
    skills = tech.get("skills", []) or []
    tools = tech.get("tools", []) or []
    return {_norm(x) for x in (skills + tools) if isinstance(x, str) and x.strip()}


def _team_items(member_ids: List[str], profile_by_id: Dict[str, dict]) -> set[str]:
    out: set[str] = set()
    for cid in member_ids:
        prof = profile_by_id.get(str(cid), {}) or {}
        out |= _cand_items(prof)
    return out


def _avg_internal_similarity(team_ids: List[str], sim_matrix: Optional[np.ndarray], candidate_ids_order: List[str]) -> float:
    """
    Average pairwise similarity within team.
    (Higher => more redundant)
    """
    if sim_matrix is None or len(team_ids) < 2:
        return 0.0

    idx = {str(cid): i for i, cid in enumerate(candidate_ids_order)}
    pairs = []
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            a, b = str(team_ids[i]), str(team_ids[j])
            if a in idx and b in idx:
                pairs.append(float(sim_matrix[idx[a], idx[b]]))
    return float(sum(pairs) / len(pairs)) if pairs else 0.0


def _contribution_distribution(member_ids: List[str], profile_by_id: Dict[str, dict], required_items: List[str]) -> Tuple[List[dict], float]:
    req = {_norm(x) for x in required_items if isinstance(x, str) and x.strip()}
    rows = []
    for cid in member_ids:
        prof = profile_by_id.get(str(cid), {}) or {}
        items = _cand_items(prof)
        contributed = sorted(list(items & req))
        pct = (len(contributed) / max(1, len(req))) * 100.0
        rows.append(
            {
                "candidate_id": str(cid),
                "n_contributed": len(contributed),
                "contribution_%": round(pct, 1),
                "contributed_items": contributed,
            }
        )
    max_pct = max((r["contribution_%"] for r in rows), default=0.0)
    return rows, float(max_pct)


def _team_hours(member_ids: List[str], profile_by_id: Dict[str, dict]) -> Dict[str, Any]:
    hours_by_member = []
    total = 0.0
    for cid in member_ids:
        prof = profile_by_id.get(str(cid), {}) or {}
        h = _parse_availability_min((prof.get("constraints", {}) or {}).get("weekly_availability_hours"))
        hours_by_member.append({"candidate_id": str(cid), "min_hours_per_week": h})
        total += h
    max_h = max((x["min_hours_per_week"] for x in hours_by_member), default=0.0)
    hero_share = (max_h / total) if total > 0 else 0.0
    return {
        "total_team_min_hours": round(total, 1),
        "hours_by_member": hours_by_member,
        "hero_share_%": round(hero_share * 100.0, 1),
    }


class TeamEvaluator:
    """
    Team evaluation for the current pipeline.

    Reads:
      - candidate_profiles.json  (clean extracted profiles)
      - project_requirements.json
      - teams_ui.json OR teams_summary.json (from UI)
      - candidate_dissimilarity.npy (optional)
      - candidate_ids.json (optional ordering for similarity matrix)

    Outputs:
      - teams_evaluation_summary.json
    """

    def load_json(self, p: Path) -> Any:
        return json.loads(p.read_text(encoding="utf-8"))

    def save_json(self, obj: Any, p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def evaluate(
        self,
        candidate_profiles_path: Path,
        project_requirements_path: Path,
        teams_path: Path,
        out_path: Path,
        candidate_ids_path: Optional[Path] = None,
        candidate_dissim_path: Optional[Path] = None,
        required_team_hours_per_week: float = 0.0,
    ) -> Dict[str, Any]:
        profiles = self.load_json(candidate_profiles_path)
        proj = self.load_json(project_requirements_path)
        teams_obj = self.load_json(teams_path)

        profile_by_id = {str(p.get("id")): p for p in profiles if isinstance(p, dict) and "id" in p}

        tr = (proj.get("technical_requirements", {}) or {})
        required_items = list(tr.get("skills", []) or []) + list(tr.get("tools", []) or [])
        required_roles = set(str(x) for x in (tr.get("roles", []) or []) if x)

       
        sim_matrix = None
        cand_order: List[str] = []
        if candidate_ids_path and candidate_dissim_path and candidate_ids_path.exists() and candidate_dissim_path.exists():
            cand_order = [str(x) for x in self.load_json(candidate_ids_path)]
            dissim = np.load(candidate_dissim_path)
            sim_matrix = 1.0 - dissim
            sim_matrix[sim_matrix < 0.0] = 0.0


        team_member_lists: List[List[str]] = []

        if isinstance(teams_obj, list):
           
            for team_rows in teams_obj:
                member_ids = [str(r.get("Candidate ID") or r.get("candidate_id") or r.get("id")) for r in team_rows]
                member_ids = [x for x in member_ids if x and x != "None"]
                if member_ids:
                    team_member_lists.append(member_ids)

        elif isinstance(teams_obj, dict) and "teams" in teams_obj:
            for t in teams_obj.get("teams", []) or []:
                mids = t.get("member_ids") or t.get("selected_candidate_ids") or []
                mids = [str(x) for x in mids]
                if mids:
                    team_member_lists.append(mids)

        results = []
        for i, member_ids in enumerate(team_member_lists, start=1):
            items = _team_items(member_ids, profile_by_id)
            req = {_norm(x) for x in required_items if isinstance(x, str) and x.strip()}

            covered = sorted(list(req & items))
            missing = sorted(list(req - items))
            cov_pct = 100.0 if not req else 100.0 * len(covered) / len(req)

            roles_present = set()
            for cid in member_ids:
                prof = profile_by_id.get(str(cid), {}) or {}
                role = (prof.get("metadata", {}) or {}).get("dev_type", "Unknown")
                roles_present.add(str(role))
            missing_roles = sorted(list(required_roles - roles_present))

            internal_sim = _avg_internal_similarity(member_ids, sim_matrix, cand_order) if sim_matrix is not None else 0.0

            dist_rows, max_contrib = _contribution_distribution(member_ids, profile_by_id, required_items)
            hours_info = _team_hours(member_ids, profile_by_id)

            feasible = True
            if required_team_hours_per_week and hours_info["total_team_min_hours"] < float(required_team_hours_per_week):
                feasible = False

            results.append(
                {
                    "team_number": i,
                    "size": len(member_ids),
                    "coverage_percent": round(float(cov_pct), 2),
                    "covered_items": covered,
                    "missing_items": missing,
                    "missing_roles": missing_roles,
                    "avg_internal_similarity": round(float(internal_sim), 4),
                    "max_contribution_%": round(float(max_contrib), 1),
                    "hours": hours_info,
                    "feasible_by_hours": feasible,
                    "member_distribution": dist_rows,
                }
            )

        summary = {
            "n_teams": len(results),
            "required_items_count": len({_norm(x) for x in required_items if x}),
            "required_roles_count": len(required_roles),
            "required_team_hours_per_week": float(required_team_hours_per_week),
            "teams": results,
        }

        self.save_json(summary, out_path)
        return summary
