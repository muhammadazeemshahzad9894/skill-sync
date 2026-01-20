from __future__ import annotations

from pathlib import Path
import json

from .embeddings import embed_projects
from .store_embeddings import save_project_embeddings
from .compute_similarity import compute_store_similarity


def gen_proj_emb_sim(project_requirements_path: Path):
  
    with open(project_requirements_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    projects = [project]  # wrap single project
    embeddings, texts = embed_projects(projects)

    project_ids = ["project_1"]
    save_project_embeddings(embeddings, project_ids)

    compute_store_similarity()

    print("Project embeddings + candidate-project similarity generated and saved successfully.")
