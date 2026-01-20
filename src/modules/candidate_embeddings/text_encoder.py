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

def clean_str(v):
    if v is None:
        return ""
    return str(v).strip()

def profile_to_text(profile: dict) -> str:
    tech = profile.get("technical", {}) or {}
    meta = profile.get("metadata", {}) or {}
    collab = profile.get("collaboration", {}) or {}
    pers = profile.get("personality", {}) or {}

    skills = clean_list(tech.get("skills", []))
    tools = clean_list(tech.get("tools", []))

    dev_type = clean_str(meta.get("dev_type"))
    belbin = clean_str(pers.get("Belbin_team_role"))
    comms = clean_str(collab.get("communication_style"))

    parts = []
    if dev_type:
        parts.append(f"Developer type: {dev_type}")
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    if tools:
        parts.append("Tools: " + ", ".join(tools))
    if belbin:
        parts.append(f"Belbin role: {belbin}")
    if comms:
        parts.append(f"Communication style: {comms}")

    return ". ".join(parts) + "." if parts else "Candidate profile: Unknown."

def embed_profiles(profiles: list[dict]):
    texts = [profile_to_text(p) for p in profiles]
    return _model.encode(texts, normalize_embeddings=True)
