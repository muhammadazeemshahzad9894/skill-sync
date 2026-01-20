# src/modules/team_builder/test_builder1_local.py
import sys
from pathlib import Path
import json
from typing import List, Dict, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # cv-sage root
sys.path.append(str(ROOT / "src"))

from shared.interfaces import CandidateProfile, ProjectDescription
from modules.team_builder.builder import build_teams


ART = ROOT / "data" / "artifacts"

CAND_PROFILES_PATH = ART / "extraction" / "candidate_profiles.json"
PROJECT_REQ_PATH = ART / "project" / "project_requirements.json"

CAND_DISSIM_PATH = ART / "candidate_dissimilarity.npy"
CAND_IDS_PATH = ART / "candidate_ids.json"
CAND_PROJ_SIM_PATH = ART / "candidate_project_similarity.json"


def _parse_availability_to_int(avail_raw) -> int:
    """
    "10–20" -> 10
    "Unknown" -> 0
    """
    s = str(avail_raw or "")
    digits = "".join([c if c.isdigit() or c == " " else " " for c in s]).split()
    return int(digits[0]) if digits else 0


def load_profiles() -> List[CandidateProfile]:
    raw = json.loads(CAND_PROFILES_PATH.read_text(encoding="utf-8"))
    profiles: List[CandidateProfile] = []

    for x in raw:
        if not isinstance(x, dict):
            continue

        technical = x.get("technical", {}) or {}
        metadata = x.get("metadata", {}) or {}
        constraints = x.get("constraints", {}) or {}
        collaboration = x.get("collaboration", {}) or {}

        skills = technical.get("skills", []) or []
        tools = technical.get("tools", []) or []

        availability_hours = _parse_availability_to_int(constraints.get("weekly_availability_hours", "Unknown"))

        exp_raw = metadata.get("work_experience_years", 0)
        try:
            experience_years = int(float(exp_raw))
        except Exception:
            experience_years = 0

        profiles.append(
            CandidateProfile(
                id=str(x.get("id", "Unknown")),
                name=str(x.get("name", "Unknown")),
                skills=list(skills),
                skill_levels={},
                role=str(metadata.get("dev_type", "Unknown")),
                experience_years=experience_years,
                collaboration_style=str(collaboration.get("communication_style", "Unknown")),
                availability_hours=availability_hours,
                tools=list(tools),
                domains=[],
            )
        )

    return profiles


def load_project_req() -> dict:
    return json.loads(PROJECT_REQ_PATH.read_text(encoding="utf-8"))


def project_req_to_description(project_req: dict) -> Tuple[ProjectDescription, int]:
    tr = project_req.get("technical_requirements", {}) or {}
    cons = project_req.get("constraints", {}) or {}

    team_size = int(cons.get("team_size", 4))
    min_av = int(cons.get("min_availability_hours_per_week", 0))

    required_skills = list(tr.get("skills", []) or []) + list(tr.get("tools", []) or [])

    project = ProjectDescription(
        id="active_project",
        title="Extracted Project",
        description="",
        required_roles=list(tr.get("roles", []) or []),
        required_skills=required_skills,
        team_size=team_size,
        duration_weeks=0,
        priority_skills=[],
    )
    return project, min_av


def load_candidate_dissimilarity_and_ids() -> Tuple[np.ndarray, List[str]]:
    dissim = np.load(CAND_DISSIM_PATH)
    candidate_ids = json.loads(CAND_IDS_PATH.read_text(encoding="utf-8"))
    return dissim, [str(x) for x in candidate_ids]


def dissimilarity_to_similarity(dissim: np.ndarray) -> np.ndarray:
    sim = 1.0 - dissim
    sim[sim < 0.0] = 0.0
    return sim


def load_candidate_project_similarities_for_active_project() -> Tuple[Dict[str, float], str]:
    payload = json.loads(CAND_PROJ_SIM_PATH.read_text(encoding="utf-8"))
    candidate_ids = [str(x) for x in payload["candidate_ids"]]
    project_ids = [str(x) for x in payload["project_ids"]]
    M = np.array(payload["similarity_matrix"], dtype=float)

    proj_idx = project_ids.index("active_project") if "active_project" in project_ids else 0
    chosen_project_id = project_ids[proj_idx]

    sims = {candidate_ids[i]: float(M[i, proj_idx]) for i in range(len(candidate_ids))}
    return sims, chosen_project_id


def main():
    print("=== SkillSync: Team Builder test (YOUR schemas + YOUR artifacts) ===")

    project_req = load_project_req()
    project, min_av = project_req_to_description(project_req)
    print(f"Project: team_size={project.team_size}, min_availability={min_av}")
    print(f"Project required_skills={len(project.required_skills)}, roles={project.required_roles}")

    profiles = load_profiles()
    print(f"Loaded profiles: {len(profiles)}")

    dissim, cand_ids_order = load_candidate_dissimilarity_and_ids()
    cand_sim_matrix = dissimilarity_to_similarity(dissim)
    print(f"Dissimilarity shape: {dissim.shape} -> similarity shape: {cand_sim_matrix.shape}")

    project_similarities, chosen_project_id = load_candidate_project_similarities_for_active_project()
    print(f"Loaded fit map for project_id={chosen_project_id}, scores={len(project_similarities)}")

    teams = build_teams(
        profiles=profiles,
        project=project,
        candidate_sim_matrix=cand_sim_matrix,
        candidate_ids=cand_ids_order,
        project_similarities=project_similarities,
        min_availability_hours=min_av,
    )

    print(f"\nBuilt teams: {len(teams)}")
    for i, team in enumerate(teams, 1):
        print(f"\nTeam {i} (size={len(team)}):")
        for m in team:
            fit = project_similarities.get(m.id, 0.0)
            print(f"- {m.id} | role={m.role} | avail={m.availability_hours} | fit={fit:.3f} | skills={m.skills[:6]}")

    out = [
        [{"id": m.id, "role": m.role, "availability_hours": m.availability_hours, "skills": m.skills, "tools": m.tools} for m in team]
        for team in teams
    ]
    (ART / "teams_debug.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved -> data/artifacts/teams_debug.json")


if __name__ == "__main__":
    main()
