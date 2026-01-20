
# src/modules/RAG/rag_retrieve.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_str(x: Any) -> str:
    s = str(x).strip() if x is not None else ""
    return s if s else "Not specified"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"unknown", "not specified", "nan"}:
        return None
    # allow "10 years", "10.0", etc.
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return int(float(m[0]))
    except Exception:
        return None


def _extract_candidate_ids_from_team_rows(team_rows: Any) -> List[str]:
    ids: List[str] = []
    if not isinstance(team_rows, list):
        return ids

    for row in team_rows:
        if isinstance(row, dict):
            if "Candidate ID" in row:
                ids.append(str(row["Candidate ID"]))
            elif "candidate_id" in row:
                ids.append(str(row["candidate_id"]))
            elif "id" in row:
                ids.append(str(row["id"]))
        else:
            try:
                ids.append(str(row))
            except Exception:
                pass

    # unique keep-order
    seen = set()
    out = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def load_team_candidate_ids(teams_ui_path: Path) -> List[List[str]]:
    teams = _load_json(teams_ui_path)
    out: List[List[str]] = []
    for team_rows in teams:
        out.append(_extract_candidate_ids_from_team_rows(team_rows))
    return out


def retrieve_candidate_inputs(candidate_inputs_path: Path, candidate_ids: List[str]) -> List[Dict[str, str]]:
    inputs = _load_json(candidate_inputs_path)  # dict: id -> text
    docs: List[Dict[str, str]] = []
    for cid in candidate_ids:
        text = inputs.get(str(cid), "")
        docs.append({"candidate_id": str(cid), "text": str(text or "")})
    return docs


def _profiles_by_id(candidate_profiles_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if candidate_profiles_path is None:
        return {}
    if not candidate_profiles_path.exists():
        return {}

    raw = _load_json(candidate_profiles_path)
    if not isinstance(raw, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for p in raw:
        if isinstance(p, dict) and "id" in p and "error" not in p:
            out[str(p["id"])] = p
    return out


def _pick(d: Dict[str, Any], path: List[str], default: Any = "Not specified") -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _counter(values: List[str]) -> Dict[str, int]:
    vals = [v for v in (values or []) if v and v not in {"Unknown", "Not specified"}]
    return dict(Counter(vals))


BELBIN_GLOSSARY = {
    "Teamworker": "Supportive, harmony-focused, helps people collaborate",
    "Shaper": "Drives progress, pushes for decisions, keeps urgency high",
    "Coordinator": "Aligns goals, delegates, organizes the group",
    "Implementer": "Turns ideas into practical steps and steady execution",
    "Monitor-Evaluator": "Analytical, weighs pros/cons, cautious decisions",
    "Plant": "Creative idea generator, novel approaches",
    "Resource-Investigator": "Explores options, brings external ideas/opportunities",
    "Completer-Finisher": "Detail-oriented, quality control, closes tasks strongly",
}


def _project_tech_requirements(project_req: Any) -> Dict[str, Any]:
    """
    Returns normalized project requirements lists:
      roles, skills, tools
    """
    if not isinstance(project_req, dict):
        return {"roles": [], "skills": [], "tools": []}

    tr = project_req.get("technical_requirements", {}) or {}
    roles = [str(x).strip() for x in (tr.get("roles", []) or []) if str(x).strip()]
    skills = [str(x).strip() for x in (tr.get("skills", []) or []) if str(x).strip()]
    tools = [str(x).strip() for x in (tr.get("tools", []) or []) if str(x).strip()]

    # dedupe keep-order
    def dedupe(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = _norm(x)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return {"roles": dedupe(roles), "skills": dedupe(skills), "tools": dedupe(tools)}


def _team_skill_tool_sets(members: List[Dict[str, Any]]) -> Dict[str, set]:
    skills: set[str] = set()
    tools: set[str] = set()

    for m in members:
        tech = (m.get("technical", {}) or {})
        for s in (tech.get("skills", []) or []):
            if s:
                skills.add(_norm(str(s)))
        for t in (tech.get("tools", []) or []):
            if t:
                tools.add(_norm(str(t)))

    return {"skills": skills, "tools": tools}


def _team_metadata_summary(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    dev_types: List[str] = []
    industries: List[str] = []
    exp_years: List[int] = []
    years_code: List[int] = []

    for m in members:
        meta = (m.get("metadata", {}) or {})

        dt = meta.get("dev_type", None)
        if dt and str(dt).strip() and str(dt).strip() != "Unknown":
            dev_types.append(str(dt).strip())

        ind = meta.get("industry", None)
        if ind and str(ind).strip() and str(ind).strip() != "Unknown":
            industries.append(str(ind).strip())

        wy = _to_int(meta.get("work_experience_years", None))
        if wy is not None:
            exp_years.append(int(wy))

        yc = _to_int(meta.get("years_code", None))
        if yc is not None:
            years_code.append(int(yc))

    def _range_stats(xs: List[int]) -> Dict[str, Any]:
        if not xs:
            return {"min": "Not specified", "max": "Not specified", "avg": "Not specified"}
        return {
            "min": int(min(xs)),
            "max": int(max(xs)),
            "avg": round(float(sum(xs) / len(xs)), 2),
        }

    return {
        "dev_type_counts": _counter(dev_types),
        "industry_counts": _counter(industries),
        "work_experience_years": _range_stats(exp_years),
        "years_code": _range_stats(years_code),
    }


def _role_coverage_verifiable(required_roles: List[str], dev_type_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    We do NOT want hallucinations. So we only claim "covered" for roles that
    can be matched *textually* against metadata.dev_type.

    Matching logic: role label is considered matched if it is a substring
    of a dev_type OR dev_type is substring of role label (case-insensitive normalized).
    """
    req = [r for r in required_roles if r.strip()]
    if not req:
        return {
            "coverage_possible": False,
            "matched_roles": [],
            "missing_roles": [],
            "note": "Not specified (no required roles provided by project requirements)",
        }

    dev_types = list(dev_type_counts.keys()) if isinstance(dev_type_counts, dict) else []
    if not dev_types:
        return {
            "coverage_possible": False,
            "matched_roles": [],
            "missing_roles": [],
            "note": "Not specified (team dev_type metadata not available to verify role coverage)",
        }

    matched = []
    missing = []

    dev_norm = [(d, _norm(d)) for d in dev_types]
    for role in req:
        rnorm = _norm(role)
        ok = False
        for d, dnorm in dev_norm:
            if rnorm and (rnorm in dnorm or dnorm in rnorm):
                ok = True
                break
        if ok:
            matched.append(role)
        else:
            missing.append(role)

    return {
        "coverage_possible": True,
        "matched_roles": matched,
        "missing_roles": missing,
        "note": "Role coverage is verified using metadata.dev_type only (no Belbin).",
    }


def build_context_all_teams(
    *,
    project_root: Path,
    teams_ui_path: Path,
    candidate_inputs_path: Path,
    project_req_path: Optional[Path] = None,
    candidate_profiles_path: Optional[Path] = None,
    teams_summary_path: Optional[Path] = None,  # optional extra
) -> Dict[str, Any]:
    """
    Returns ONE context JSON containing:
    - project requirements (if available)
    - for each team:
      - member_ids
      - algorithmic fields from teams_ui rows (Fit, Skills sample, etc.) if present
      - structured extracted signals (Belbin + collaboration + learning_behavior) if profiles exist
      - precomputed per-team dynamics counts
      - NEW: per-team technical summary (skills/tools/dev_type/industry/experience)
      - NEW: deterministic required skills/tools matching (no hallucinations)
    """
    teams_ui = _load_json(teams_ui_path)
    candidate_inputs = _load_json(candidate_inputs_path)
    profiles_map = _profiles_by_id(candidate_profiles_path)

    project_req = None
    if project_req_path is not None and project_req_path.exists():
        try:
            project_req = _load_json(project_req_path)
        except Exception:
            project_req = None

    teams_summary = None
    if teams_summary_path is not None and teams_summary_path.exists():
        try:
            teams_summary = _load_json(teams_summary_path)
        except Exception:
            teams_summary = None

    proj_tr = _project_tech_requirements(project_req) if project_req is not None else {"roles": [], "skills": [], "tools": []}

    teams: List[Dict[str, Any]] = []
    for t_idx, team_rows in enumerate(teams_ui, start=1):
        member_ids = _extract_candidate_ids_from_team_rows(team_rows)

        # lightweight algorithmic/selection info from teams_ui.json rows (if present)
        algo_rows = []
        if isinstance(team_rows, list):
            for r in team_rows:
                if isinstance(r, dict):
                    algo_rows.append(r)

        members: List[Dict[str, Any]] = []
        for cid in member_ids:
            inp_text = str(candidate_inputs.get(str(cid), "") or "")
            prof = profiles_map.get(str(cid), {}) or {}
            member = {
                "candidate_id": str(cid),

                # Structured extracted signals (preferred for dynamics)
                "metadata": _pick(prof, ["metadata"], {}),
                "personality": _pick(prof, ["personality"], {}),
                "collaboration": _pick(prof, ["collaboration"], {}),
                "learning_behavior": _pick(prof, ["learning_behavior"], {}),

                # Technical summary
                "technical": _pick(prof, ["technical"], {}),

                # Constraints
                "constraints": _pick(prof, ["constraints"], {}),

                # Raw input blob (for AI usage quotes, optional grounding)
                "candidate_input_text": inp_text,
            }
            members.append(member)

        # ----------------------------
        # Dynamics snapshot counts
        # ----------------------------
        belbins = [_safe_str(_pick(m, ["personality", "Belbin_team_role"], "Not specified")) for m in members]
        comms = [_safe_str(_pick(m, ["collaboration", "communication_style"], "Not specified")) for m in members]
        conflicts = [_safe_str(_pick(m, ["collaboration", "conflict_style"], "Not specified")) for m in members]
        leads = [_safe_str(_pick(m, ["collaboration", "leadership_preference"], "Not specified")) for m in members]
        deadlines = [_safe_str(_pick(m, ["collaboration", "deadline_discipline"], "Not specified")) for m in members]
        ks = [_safe_str(_pick(m, ["learning_behavior", "knowledge_sharing"], "Not specified")) for m in members]
        lo = [_safe_str(_pick(m, ["learning_behavior", "learning_orientation"], "Not specified")) for m in members]

        snapshot = {
            "belbin_counts": _counter(belbins),
            "communication_counts": _counter(comms),
            "conflict_counts": _counter(conflicts),
            "leadership_counts": _counter(leads),
            "deadline_counts": _counter(deadlines),
            "knowledge_sharing_counts": _counter(ks),
            "learning_orientation_counts": _counter(lo),
        }

        # ----------------------------
        # NEW: Technical summary + deterministic matching
        # ----------------------------
        st = _team_skill_tool_sets(members)
        meta_sum = _team_metadata_summary(members)

        # required matching ONLY if project lists exist
        req_skills = proj_tr.get("skills", []) or []
        req_tools = proj_tr.get("tools", []) or []
        req_roles = proj_tr.get("roles", []) or []

        team_skills = st["skills"]
        team_tools = st["tools"]

        matched_skills = [s for s in req_skills if _norm(s) in team_skills]
        missing_skills = [s for s in req_skills if _norm(s) not in team_skills]

        matched_tools = [t for t in req_tools if _norm(t) in team_tools]
        missing_tools = [t for t in req_tools if _norm(t) not in team_tools]

        role_cov = _role_coverage_verifiable(req_roles, meta_sum.get("dev_type_counts", {}))

        technical_summary = {
            "top_skills": sorted(list(team_skills))[:30],  # normalized strings
            "top_tools": sorted(list(team_tools))[:30],
            "dev_type_counts": meta_sum.get("dev_type_counts", {}),
            "industry_counts": meta_sum.get("industry_counts", {}),
            "work_experience_years": meta_sum.get("work_experience_years", {}),
            "years_code": meta_sum.get("years_code", {}),
            "required_matching": {
                "required_roles": req_roles,
                "required_skills": req_skills,
                "required_tools": req_tools,

                "skills": {
                    "can_evaluate": bool(req_skills),
                    "matched": matched_skills,
                    "missing": missing_skills,
                    "note": "Only evaluated if project requirements provided required skills.",
                },
                "tools": {
                    "can_evaluate": bool(req_tools),
                    "matched": matched_tools,
                    "missing": missing_tools,
                    "note": "Only evaluated if project requirements provided required tools.",
                },
                "roles": role_cov,
            },
        }

        # optional attach teams_summary row if present
        team_summary_row = None
        if isinstance(teams_summary, dict):
            for t in (teams_summary.get("teams", []) or []):
                if int(t.get("team_number", -1)) == int(t_idx):
                    team_summary_row = t
                    break

        teams.append(
            {
                "team_number": t_idx,
                "member_ids": member_ids,
                "algo_rows": algo_rows,
                "members": members,
                "dynamics_snapshot": snapshot,

                # ✅ NEW
                "technical_summary": technical_summary,
                "team_summary": team_summary_row or "Not specified",
            }
        )

    return {
        "project_requirements": project_req if project_req is not None else "Not specified",
        "project_technical_requirements": proj_tr,
        "belbin_glossary": BELBIN_GLOSSARY,
        "teams": teams,
        "note": "Belbin roles describe collaboration style; technical roles come from metadata.dev_type and project requirements roles.",
    }
