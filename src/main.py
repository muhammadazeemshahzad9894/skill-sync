from pathlib import Path
from modules.candidate_embeddings.embedding_service import gen_cand_emb_dissim
from modules.project_embeddings.generate_embeddings import gen_poj_emb_sim

PROJECT_ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    
    profiles_path = PROJECT_ROOT / "data" / "candidate_profiles.json"
    projects_path = PROJECT_ROOT / "data" / "sample" / "sample_projects.json"
    gen_cand_emb_dissim(profiles_path)
    gen_poj_emb_sim(projects_path)
    