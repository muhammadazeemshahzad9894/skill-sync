# src/modules/project_embeddings/embeddings.py
from __future__ import annotations
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_list(values):
    if not values:
        return []
    out = []
    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return sorted(out, key=lambda x: x.lower())

def project_to_text(project: dict) -> str:
    tr = project.get("technical_requirements", {}) or {}

    skills = clean_list(tr.get("skills", []))
    tools = clean_list(tr.get("tools", []))
    roles = clean_list(tr.get("roles", []))

    parts = []
    if roles:
        parts.append("Required roles: " + ", ".join(roles))
    if skills:
        parts.append("Required skills: " + ", ".join(skills))
    if tools:
        parts.append("Required tools: " + ", ".join(tools))

    return ". ".join(parts) + "." if parts else "Project requirements: Unknown."

def embed_projects(projects: list[dict]):
    texts = [project_to_text(p) for p in projects]
    embeddings = _model.encode(texts, normalize_embeddings=True)
    return embeddings, texts

