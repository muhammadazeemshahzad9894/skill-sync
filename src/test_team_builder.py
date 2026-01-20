from pathlib import Path
from modules.candidate_embeddings.utils import load_profiles
from modules.candidate_embeddings.repository import load_embeddings, load_dissimilarity
from modules.project_embeddings.utils import load_projects
from modules.project_embeddings.store_embeddings import load_project_embeddings 
from modules.project_embeddings.compute_similarity import compute_project_similarities
from modules.team_builder.team_builder import build_team_greedy_refined


ROOT = Path(__file__).resolve().parent.parent


def test_team_builder():
    profiles = load_profiles(ROOT / "data/candidate_profiles_with_evidence.json")

    candidate_embeddings, candidate_ids = load_embeddings()
    candidate_dissim = load_dissimilarity()

    projects = load_projects(ROOT / "data/sample/sample_projects.json")
    project = projects[11]

    project_embeddings, project_ids = load_project_embeddings()

    proj_sim = compute_project_similarities(
        project_embeddings[11],
        candidate_embeddings,
        candidate_ids,
    )

    team = build_team_greedy_refined(
        profiles=profiles,
        project=project,
        candidate_dissim_matrix=candidate_dissim,
        candidate_ids=candidate_ids,
        project_similarities=proj_sim,
        team_size=project["team_size"],
    )

    for m in team:
        print(m["id"], m.get("technical", {}).get("skills", []))



test_team_builder()