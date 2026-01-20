import numpy as np

def _norm_skill(s: str) -> str:
    return (s or "").strip().lower()

def _team_skill_set(team):
    s = set()
    for m in team:
        s.update(_norm_skill(x) for x in m.get("technical", {}).get("skills", []))
    return s


def build_candidate_index(candidate_ids: list[str]) -> dict[str, int]:
    return {cid: i for i, cid in enumerate(candidate_ids)}


def avg_dissimilarity_to_team(candidate_id, team, dissim_matrix, id_to_idx):
    if not team or candidate_id not in id_to_idx:
        return 0.0

    ci = id_to_idx[candidate_id]
    vals = [
        float(dissim_matrix[ci, id_to_idx[m["id"]]])
        for m in team
        if m["id"] in id_to_idx
    ]
    return float(sum(vals) / len(vals)) if vals else 0.0


def coverage_gain(team, candidate, required_skills):
    req = {_norm_skill(x) for x in (required_skills or []) if x}
    if not req:
        return 0.0

    team_sk = _team_skill_set(team)
    cand_sk = {
        _norm_skill(x)
        for x in candidate.get("technical", {}).get("skills", [])
    }

    before = len(req & team_sk)
    after = len(req & (team_sk | cand_sk))
    return (after - before) / max(1, len(req))


def score_candidate(
    team,
    candidate,
    project,
    dissim_matrix,
    id_to_idx,
    project_similarities,
    alpha,
    beta,
    gamma,
):
    fit = project_similarities.get(candidate["id"], 0.0)
    coverage = coverage_gain(team, candidate, project["required_skills"])
    diversity = avg_dissimilarity_to_team(
        candidate["id"],
        team,
        dissim_matrix,
        id_to_idx,
    )
    return (alpha * fit) + (gamma * coverage) + (beta * diversity)




def build_team_greedy_refined(
    profiles,
    project,
    candidate_dissim_matrix,
    candidate_ids,
    project_similarities,
    team_size,
    alpha=0.55,
    beta=0.25,
    gamma=0.20,
):

    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    profiles = sorted(
        profiles,
        key=lambda p: project_similarities.get(p["id"], 0.0),
        reverse=True,
    )

    team = [profiles[0]]
    remaining = profiles[1:]

    while len(team) < team_size and remaining:
        scored = [
            (
                score_candidate(
                    team,
                    cand,
                    project,
                    candidate_dissim_matrix,
                    id_to_idx,
                    project_similarities,
                    alpha,
                    beta,
                    gamma,
                ),
                cand,
            )
            for cand in remaining
        ]

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1]

        team.append(chosen)
        remaining = [c for c in remaining if c["id"] != chosen["id"]]

    return team

