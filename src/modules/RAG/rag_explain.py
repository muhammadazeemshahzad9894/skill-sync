

# src/modules/RAG/rag_explain.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .rag_retrieve import build_context_all_teams
from .rag_paths import RagPaths

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


SYSTEM_PROMPT = """
You are SkillSync, a team reporting assistant.

Hard rules:
- Use ONLY the provided CONTEXT JSON.
- Do NOT invent facts about candidates, teams, skills, tools, roles, industries, or experience.
- If something is missing, write "Not specified".
- Belbin roles are NOT technical roles. Never treat Belbin as job roles.
- For team dynamics, prefer these structured fields:
  personality.Belbin_team_role
  collaboration.communication_style / conflict_style / leadership_preference / deadline_discipline
  learning_behavior.learning_orientation / knowledge_sharing
- For technical sections, use ONLY:
  project_technical_requirements (roles/skills/tools)
  and team technical_summary (skills/tools/dev_type/industry/experience) derived from metadata + technical fields.

Goal:
Produce a user-friendly Team Report with:
1) Project requirements first (technical)
2) For EACH team:
   - Technical section first (roles/skills/tools + industry + experience + dev_type)
   - Then Team dynamics section (same style you already produce)
3) Pick ONE recommended team overall (mostly based on dynamics, but can mention technical match lightly).

Return VALID JSON ONLY (no markdown wrapper outside JSON).
""".strip()


USER_PROMPT = """
TASK:
Produce a user-friendly Team Report based on CONTEXT.
Pick ONE recommended team overall.

Output JSON schema (exact keys):
{
  "report_markdown": "...",
  "recommended_team_number": 0
}

REPORT STYLE:
- Keep it friendly and readable.
- Do not remove existing content style (strengths, risks, etc.) — just reorder and add.
- Do NOT claim missing requirements unless that category was provided by project requirements.
- When requirements exist, show the *actual matched items* (not just "provided").

REPORT FORMAT (report_markdown):

Start with:
# Team Report

## Project requirements (technical)
Show:
- Required roles (if provided and non-empty; else say Not specified)
- Required skills (if provided and non-empty; else say Not specified)
- Required tools (if provided and non-empty; else say Not specified)

Then include:
## Belbin Roles Glossary
Use CONTEXT.belbin_glossary. After the glossary, add a clear note line:
"Note: Belbin roles describe teamwork style — they are not job titles / technical project roles."

Then for EACH team (must include all teams):

## Team X

### Technical fit (roles/skills/tools)
1) Required coverage (short, user-friendly):
   First check whether PROJECT REQUIREMENTS actually include each category:
   (a) required roles, (b) required skills, (c) required tools.
   - Only evaluate "coverage" for categories that are explicitly present and non-empty.
   - If a category is missing or empty in project requirements, write:
     "Not specified (not provided by the project description)" and do NOT claim gaps.
   - If categories are present but team data is insufficient to verify, write "Not specified".
   - Refer ONLY to technical roles from project requirements and metadata.dev_type (never Belbin).
   - For skills/tools coverage: if present, explicitly list which required skills/tools are matched and which are missing (short lists).
2) Team technical footprint (free text):
   - Mention top skills and top tools (short).
   - Mention dev-type spread (from metadata.dev_type counts).
   - Mention industry signals (metadata.industry counts) if present.
   - Mention experience range/average (metadata.work_experience_years) if present.

### Team dynamics
Keep your existing structure:
1) What this team is like (2–4 sentences)
2) Dynamics snapshot (one line of counts; concise)
3) Strengths (3–5 bullets)
4) Risks / watch-outs (2–5 bullets)
5) Light technical confirmation (1–2 bullets max, optional)

End with:
# Recommended Team: #X
with 4–7 bullet reasons (mostly dynamics-driven, with light technical mention allowed).

Now return JSON only.

CONTEXT:
""".strip()


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    a = raw.find("{")
    b = raw.rfind("}")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("No JSON object found in model output.")
    return json.loads(raw[a: b + 1])


def run_team_report(
    *,
    api_key: str,
    base_url: str = OPENROUTER_BASE_URL,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.0,
    teams_ui_path: Path,
    candidate_inputs_path: Path,
    project_req_path: Optional[Path] = None,
    teams_summary_path: Optional[Path] = None,  # ✅ NEW (optional)
) -> Dict[str, Any]:
    """
    UI calls THIS.

    Returns:
      {"context": ..., "result": {"report_markdown": "...", "recommended_team_number": N}}
    """
    if not api_key or not api_key.strip():
        raise ValueError("Missing api_key")

    if not teams_ui_path.exists():
        raise FileNotFoundError(f"teams_ui.json not found: {teams_ui_path}")
    if not candidate_inputs_path.exists():
        raise FileNotFoundError(f"candidate_inputs.json not found: {candidate_inputs_path}")

    project_root = Path(__file__).resolve().parents[3]
    paths = RagPaths(project_root)

    candidate_profiles_path = paths.candidate_profiles if paths.candidate_profiles.exists() else None

    # prefer UI-provided project requirements if passed, else RagPaths
    pr_path = project_req_path if project_req_path and project_req_path.exists() else paths.project_requirements

    # NEW: teams summary path (optional)
    ts_path = teams_summary_path if teams_summary_path and teams_summary_path.exists() else paths.teams_summary

    context = build_context_all_teams(
        project_root=project_root,
        teams_ui_path=teams_ui_path,
        candidate_inputs_path=candidate_inputs_path,
        project_req_path=pr_path if pr_path and pr_path.exists() else None,
        candidate_profiles_path=candidate_profiles_path,
        teams_summary_path=ts_path if ts_path and ts_path.exists() else None,
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    user = USER_PROMPT + "\n" + json.dumps(context, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()
    result = _safe_json_loads(raw)

    # UI safety
    if "report_markdown" not in result:
        result["report_markdown"] = "Not specified"
    if "recommended_team_number" not in result:
        result["recommended_team_number"] = 0

    return {"context": context, "result": result}
