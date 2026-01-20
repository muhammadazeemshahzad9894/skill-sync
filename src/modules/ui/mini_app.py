

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

#for deployment
file_path = os.path.abspath(__file__) 
ui_dir = os.path.dirname(file_path)
modules_dir = os.path.dirname(ui_dir)
src_dir = os.path.dirname(modules_dir)
root_dir = os.path.dirname(src_dir)
sys.path.append(root_dir)



ROOT = Path(__file__).resolve().parents[3]  # cv-sage/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# EXTRACTORS
from modules.json_extraction.proj_extractor import ProjectSchemaExtractor, ProjectExtractorConfig
from modules.json_extraction.extractor import CandidateProfileExtractor, ExtractorConfig, OPENROUTER_BASE_URL

# EMBEDDINGS
from modules.candidate_embeddings.embedding_service import gen_cand_emb_dissim
from modules.project_embeddings.generate_embeddings import gen_proj_emb_sim

# TEAM BUILDER
from modules.team_builder.builder import build_teams_round_robin, summarize_teams_for_ui
from shared.interfaces import CandidateProfile, ProjectDescription

# EVALUATORS
from modules.evaluators.profile_ex_eval import CandidateProfileEvaluator
from modules.evaluators.project_ex_eval import ProjectRequirementsEvaluator
from modules.evaluators.team_eval import TeamEvaluator
from modules.evaluators.latency_logger import log_latency  # JSONL logger

# EXPLANATION
import modules.EXPLANATION.team_report as explain_report


st.set_page_config(page_title="SkillSync - Team Formation", layout="wide")
st.title("SkillSync - Team Formation")
st.caption(
    "Project description → project constraints → extract project requirements → "
    "upload candidates → extract profiles → embeddings → similarity → build teams → evaluate → explanation report."
)

with st.sidebar:
    st.header("Model / API Settings (OpenRouter)")
    api_key = st.text_input("API Key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")
    base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", OPENROUTER_BASE_URL))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"))

tab_run, tab_explain, tab_eval, tab_raw = st.tabs(
    ["Run Pipeline", "Explanation Report", "Evaluation", "Raw JSON"]
)


artifacts_root = ROOT / "data" / "artifacts"

out_extraction_dir = artifacts_root / "extraction"
out_extraction_dir.mkdir(parents=True, exist_ok=True)

cand_profiles_clean_path = out_extraction_dir / "candidate_profiles.json"
cand_profiles_evid_path = out_extraction_dir / "candidate_profiles_with_evidence.json"
cand_inputs_path_default = out_extraction_dir / "candidate_inputs.json"

proj_dir = artifacts_root / "project"
proj_dir.mkdir(parents=True, exist_ok=True)

proj_desc_path = proj_dir / "project_description.txt"
proj_req_path = proj_dir / "project_requirements.json"

eval_dir = artifacts_root / "evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

cand_eval_path = eval_dir / "candidate_extraction_eval.json"
proj_eval_path = eval_dir / "project_extraction_eval.json"
teams_eval_path = eval_dir / "teams_evaluation_summary.json"

latency_log_path = eval_dir / "latency_log.jsonl"

# Embedding / similarity artifacts
cand_emb_path = artifacts_root / "candidate_embeddings.npy"
cand_ids_path = artifacts_root / "candidate_ids.json"
cand_dissim_path = artifacts_root / "candidate_dissimilarity.npy"
sim_json_path = artifacts_root / "candidate_project_similarity.json"

teams_out_path_default = artifacts_root / "teams_ui.json"
teams_summary_path = artifacts_root / "teams_summary.json"

explain_dir = artifacts_root / "explanation"
explain_dir.mkdir(parents=True, exist_ok=True)
explain_out_path = explain_dir / "team_report.json"



def _file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def status_line(label: str, path: Path) -> None:
    ok = _file_ok(path)
    st.write(f"{label}: {path}" + ("" if ok else " (missing)"))


def read_json_if_ok(p: Path) -> Optional[dict]:
    if not _file_ok(p):
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_availability_to_int(avail_raw: Any) -> int:
    s = str(avail_raw or "").strip()
    if not s or s.lower() == "unknown":
        return 0
    s = s.replace("–", "-")
    digits = "".join([c if c.isdigit() or c == " " else " " for c in s]).split()
    return int(digits[0]) if digits else 0


def dissimilarity_to_similarity(dissim: np.ndarray) -> np.ndarray:
    sim = 1.0 - dissim
    sim[sim < 0.0] = 0.0
    return sim


def _norm_item(s: str) -> str:
    return (s or "").strip().lower()


def candidate_requirement_coverage_percent(candidate: CandidateProfile, required_items: list[str]) -> float:
    req = {_norm_item(x) for x in (required_items or []) if str(x).strip()}
    if not req:
        return 0.0
    cand_items = {_norm_item(x) for x in (candidate.skills or []) if x}
    cand_items |= {_norm_item(x) for x in (candidate.tools or []) if x}
    covered = len(req & cand_items)
    return 100.0 * covered / len(req)


def candidate_requirement_matches(candidate: CandidateProfile, required_items: list[str]) -> list[str]:
    req_norm = {_norm_item(x) for x in (required_items or []) if str(x).strip()}
    if not req_norm:
        return []
    cand_norm = {_norm_item(x) for x in (candidate.skills or []) if x}
    cand_norm |= {_norm_item(x) for x in (candidate.tools or []) if x}
    return [x for x in (required_items or []) if _norm_item(x) in cand_norm]


def load_profiles_as_objects(profiles_path: Path) -> List[CandidateProfile]:
    raw = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles: List[CandidateProfile] = []

    for x in raw:
        if not isinstance(x, dict) or "error" in x:
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


def build_profile_lookup(profiles_json_path: Path) -> Dict[str, dict]:
    raw = json.loads(profiles_json_path.read_text(encoding="utf-8"))
    return {str(x["id"]): x for x in raw if isinstance(x, dict) and "id" in x}


def project_req_to_description_ui(project_req: dict, team_size_ui: int) -> ProjectDescription:
    tr = project_req.get("technical_requirements", {}) or {}
    required_items = list(tr.get("skills", []) or []) + list(tr.get("tools", []) or [])

    return ProjectDescription(
        id="ui_project",
        title="Extracted Project",
        description="",
        required_roles=list(tr.get("roles", []) or []),
        required_skills=required_items,
        team_size=int(team_size_ui),
        duration_weeks=0,
        priority_skills=[],
    )


def load_fit_map_from_similarity_json(sim_json_path_: Path) -> Dict[str, float]:
    payload = json.loads(sim_json_path_.read_text(encoding="utf-8"))
    candidate_ids = [str(x) for x in payload["candidate_ids"]]
    M = np.array(payload["similarity_matrix"], dtype=float)
    return {candidate_ids[i]: float(M[i, 0]) for i in range(len(candidate_ids))}


def validate_project_extraction(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")
    if "technical_requirements" not in obj:
        raise ValueError("Missing 'technical_requirements'.")
    tr = obj["technical_requirements"]
    for k in ["skills", "tools", "roles"]:
        tr[k] = list(dict.fromkeys([x.strip() for x in tr.get(k, []) if isinstance(x, str)]))
    return {"technical_requirements": tr}


@st.cache_data(show_spinner=False)
def cached_extract_project(api_key_: str, base_url_: str, model_: str, description_: str) -> dict:
    extractor_ = ProjectSchemaExtractor(
        api_key=api_key_,
        base_url=base_url_,
        config=ProjectExtractorConfig(model=model_, temperature=0.0),
    )
    return validate_project_extraction(extractor_.extract_requirements(description_))


def build_id_to_index(candidate_ids_order: List[str]) -> Dict[str, int]:
    return {str(cid): i for i, cid in enumerate(candidate_ids_order)}


def candidate_avg_dissim_to_teammates(candidate_id: str, team_ids: List[str], dissim: np.ndarray, id2idx: Dict[str, int]) -> float:
    cid = str(candidate_id)
    if cid not in id2idx:
        return 0.0
    others = [str(t) for t in team_ids if str(t) != cid and str(t) in id2idx]
    if not others:
        return 0.0
    i = id2idx[cid]
    vals = [float(dissim[i, id2idx[o]]) for o in others]
    return float(sum(vals) / len(vals)) if vals else 0.0


def avg_team_dissimilarity(team_ids: List[str], dissim: np.ndarray, id2idx: Dict[str, int]) -> float:
    idxs = [id2idx[str(c)] for c in team_ids if str(c) in id2idx]
    if len(idxs) < 2:
        return 0.0
    vals = []
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            vals.append(float(dissim[idxs[i], idxs[j]]))
    return float(sum(vals) / len(vals)) if vals else 0.0


# start pipeline here 
with tab_run:
    st.subheader("Project description")
    description = st.text_area(
        "Paste your project description (what you want to build)",
        height=180,
        placeholder="Example: Redesign an e-commerce UI, add recommendations, improve performance, etc.",
    )

    st.subheader("Project constraints")
    c1, c2 = st.columns(2)

    with c1:
        team_size = st.number_input("How many members in the project?", min_value=2, max_value=20, value=5, step=1)

    with c2:
        project_hours_per_week = st.number_input(
            "Project working hours per week",
            min_value=0,
            max_value=2000,
            value=50,
            step=5,
            help="Example: total hours/week for the whole project team.",
        )

    required_hours_per_person = float(project_hours_per_week) / float(team_size) if int(team_size) > 0 else 0.0
    st.caption(f"Auto-calculated required hours per person: **{required_hours_per_person:.1f} hrs/week**")


    st.subheader("Extract and save project requirements (LLM)")
    extract_project_btn = st.button(
        "Extract and save Project Requirements JSON",
        type="primary",
        disabled=not description.strip(),
    )

    if extract_project_btn:
        if not api_key.strip():
            st.error("Missing API key.")
        else:
            with st.spinner("Extracting project requirements..."):
                t0 = time.perf_counter()
                extracted = cached_extract_project(api_key, base_url, model, description)

                project_requirements = {
                    **extracted,
                    "constraints": {
                        "team_size": int(team_size),
                        "project_hours_per_week": float(project_hours_per_week),
                        "required_hours_per_person": float(required_hours_per_person),
                    },
                }

                proj_desc_path.write_text(description, encoding="utf-8")
                proj_req_path.write_text(
                    json.dumps(project_requirements, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                log_latency(
                    log_path=latency_log_path,
                    stage="project_extraction",
                    total_s=time.perf_counter() - t0,
                    model=model,
                    base_url=base_url,
                    description_chars=len(description),
                )

                st.success("Project requirements saved.")
                st.code(str(proj_req_path))
                st.json(project_requirements)

            # If candidate embeddings exist already, auto compute similarity too
            if _file_ok(cand_emb_path) and _file_ok(cand_ids_path):
                with st.spinner("Auto-embedding project and computing similarity..."):
                    try:
                        gen_proj_emb_sim(proj_req_path)
                        st.success("Project embeddings + candidate-project similarity generated automatically.")
                        st.code(str(sim_json_path))
                    except Exception as e:
                        st.error(f"Auto project embedding/similarity failed: {e}")
            else:
                st.info(
                    "Candidate embeddings not found yet. After extracting candidates (which generates embeddings), "
                    "click this project extraction button again to generate similarity automatically."
                )

    st.divider()

  
    # Upload CSV

    st.subheader("Upload candidate CSV")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    df_candidates = None
    if uploaded_csv is not None:
        try:
            df_candidates = pd.read_csv(uploaded_csv)
            st.success("CSV loaded successfully.")
            st.write(f"Rows: {len(df_candidates)} | Columns: {len(df_candidates.columns)}")
            st.dataframe(df_candidates.head(8), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.divider()

   
    #candidate extraction
   
    st.subheader("Extract candidate profiles (LLM)")

    if df_candidates is None:
        st.info("Upload a CSV to enable candidate extraction.")
    else:
        max_n = min(2000, len(df_candidates))
        n_candidates = st.number_input(
            "How many candidates to extract?",
            min_value=1,
            max_value=max_n,
            value=min(200, max_n),
            step=1,
        )

        pool_size_ui = st.number_input(
            "Team builder pool size (top K candidates)",
            min_value=10,
            max_value=2000,
            value=int(st.session_state.get("pool_size_ui", 200)),
            step=10,
            help="Team builder will consider the top K candidates by project fit.",
        )
        st.session_state["pool_size_ui"] = int(pool_size_ui)

        extract_candidates_btn = st.button(
            "Extract candidates",
            type="primary",
            disabled=not api_key.strip(),
        )

        if extract_candidates_btn:
            cfg = ExtractorConfig(
                model=model,
                limit=int(n_candidates),
                retries=1,
                request_timeout_s=60,
                sleep_between_calls_s=0.2,
                debug=False,
            )

            extractor = CandidateProfileExtractor(
                api_key=api_key,
                base_url=base_url,
                config=cfg,
            )

            limit = min(int(n_candidates), len(df_candidates))
            st.write(f"Extracting {limit} candidates...")

            progress = st.progress(0.0)
            status = st.empty()

            profiles: list[dict] = []
            inputs: dict[str, str] = {}

            t0 = time.perf_counter()
            for i in range(limit):
                row = df_candidates.iloc[i].to_dict()
                cid = extractor.as_str(row.get("Candidate ID"))
                status.write(f"Extracting {i + 1}/{limit} (Candidate ID={cid})")

                try:
                    text = extractor.build_profile_text(row)
                    inputs[cid] = text
                    prof = extractor.extract_one(row, i=i)
                    profiles.append(prof)
                except Exception as e:
                    profiles.append({"id": cid, "error": str(e)})

                progress.progress((i + 1) / limit)

            CandidateProfileExtractor.save_json(profiles, str(cand_profiles_evid_path))
            CandidateProfileExtractor.save_json(
                [CandidateProfileExtractor.strip_evidence(p) for p in profiles],
                str(cand_profiles_clean_path),
            )
            CandidateProfileExtractor.save_json(inputs, str(cand_inputs_path_default))

            log_latency(
                log_path=latency_log_path,
                stage="candidate_extraction",
                total_s=time.perf_counter() - t0,
                model=model,
                base_url=base_url,
                n_candidates=int(limit),
            )

            st.success("Candidate extraction finished.")
            st.code(str(cand_profiles_clean_path))
            st.code(str(cand_profiles_evid_path))
            st.code(str(cand_inputs_path_default))

            if profiles:
                st.subheader("Example: first extracted profile")
                st.json(profiles[0])

            with st.spinner("Auto-generating candidate embeddings..."):
                try:
                    gen_cand_emb_dissim(cand_profiles_clean_path)
                    st.success("Candidate embeddings generated automatically.")
                    st.code(str(cand_emb_path))
                    st.code(str(cand_ids_path))
                    st.code(str(cand_dissim_path))
                except Exception as e:
                    st.error(f"Auto candidate embeddings failed: {e}")

            # If project requirements already exist, auto compute similarity now too
            if _file_ok(proj_req_path) and _file_ok(cand_emb_path) and _file_ok(cand_ids_path):
                with st.spinner("Auto-embedding project and computing similarity..."):
                    try:
                        gen_proj_emb_sim(proj_req_path)
                        st.success("Candidate-project similarity generated automatically.")
                        st.code(str(sim_json_path))
                    except Exception as e:
                        st.error(f"Auto similarity failed: {e}")

    st.divider()

    # Team Builder

    st.subheader("Team Builder")

    st.subheader("Team Builder")
    n_teams_ui = st.number_input("How many teams to build?", min_value=1, max_value=50, value=2, step=1)


    build_team_btn = st.button(
        "Build Teams",
        type="primary",
        disabled=not (
            cand_profiles_clean_path.exists()
            and proj_req_path.exists()
            and cand_ids_path.exists()
            and cand_dissim_path.exists()
            and sim_json_path.exists()
        ),
    )

    if build_team_btn:
        with st.spinner("Building teams..."):
            try:
                project_req = json.loads(proj_req_path.read_text(encoding="utf-8"))
                project = project_req_to_description_ui(project_req, team_size_ui=int(team_size))

                profiles = load_profiles_as_objects(cand_profiles_clean_path)
                profile_lookup = build_profile_lookup(cand_profiles_clean_path)

                candidate_ids_order = json.loads(cand_ids_path.read_text(encoding="utf-8"))
                candidate_ids_order = [str(x) for x in candidate_ids_order]
                id2idx = build_id_to_index(candidate_ids_order)

                profiles_by_id = {p.id: p for p in profiles}
                profiles = [profiles_by_id[cid] for cid in candidate_ids_order if cid in profiles_by_id]

                dissim = np.load(cand_dissim_path)
                cand_sim_matrix = dissimilarity_to_similarity(dissim)

                project_similarities = load_fit_map_from_similarity_json(sim_json_path)

                pool_size_ui = int(st.session_state.get("pool_size_ui", 200))

                eligible_profiles = [p for p in profiles if p.id in project_similarities]

                need_min = int(n_teams_ui) * int(project.team_size)
                max_possible = len(eligible_profiles) // int(project.team_size)

                st.caption(
                    f"Required hours/person: {required_hours_per_person:.1f} | "
                    f"Pool size (top K): {pool_size_ui} | "
                    f"Eligible candidates (with fit): {len(eligible_profiles)} | "
                    f"Need: {need_min}"
                )

                if int(n_teams_ui) > int(max_possible):
                    st.error(
                        f"Not enough eligible candidates to build {int(n_teams_ui)} teams.\n\n"
                        f"- Requested: {int(n_teams_ui)} teams × {int(project.team_size)} people = {need_min}\n"
                        f"- Eligible candidates: {len(eligible_profiles)}\n"
                        f"- Max teams possible right now: {max_possible}\n\n"
                        "Fix ideas:\n"
                        "• Increase extracted candidates (N)\n"
                        "• Increase pool size (top K)\n"
                        "• Reduce team size\n"
                    )
                    st.stop()

                teams = build_teams_round_robin(
                    profiles=eligible_profiles,
                    project=project,
                    candidate_sim_matrix=cand_sim_matrix,
                    candidate_ids=candidate_ids_order,
                    project_similarities=project_similarities,
                    min_availability_hours=0,
                    n_teams=int(n_teams_ui),
                    pool_size=int(pool_size_ui),
                    builder_kwargs=None,
                )

                if not teams:
                    st.warning("No teams could be formed.")
                else:
                    st.success(f"Built {len(teams)} team(s).")

                    summary = summarize_teams_for_ui(
                        teams=teams,
                        project=project,
                        project_similarities=project_similarities,
                        candidate_sim_matrix=cand_sim_matrix,
                        candidate_ids_order=candidate_ids_order,
                    )
                    teams_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

                    st.subheader("Team Overview (missing items/roles and internal diversity)")
                    teams_rows = summary.get("teams", []) or []

                    overview_rows = []
                    for t in teams_rows:
                        team_num = int(t.get("team_number", 0) or 0)
                        team_ids = [str(x) for x in (t.get("member_ids", []) or [])]

                        missing_items = t.get("missing_items", []) or []
                        missing_roles = t.get("missing_roles", []) or []

                        avg_internal_sim = float(t.get("avg_internal_similarity", 0.0) or 0.0)
                        avg_internal_dissim = avg_team_dissimilarity(team_ids, dissim, id2idx)

                        avg_fit_val = t.get("avg_fit", None)
                        avg_fit_display = "N/A" if avg_fit_val is None else float(avg_fit_val)

                        overview_rows.append(
                            {
                                "Team": team_num,
                                "Avg fit": avg_fit_display,
                                "Coverage (skills and tools) %": float(t.get("coverage_percent", 0.0) or 0.0),
                                "Avg internal sim": round(avg_internal_sim, 4),
                                "Avg internal dissim": round(avg_internal_dissim, 4),
                                "Missing roles (list)": ", ".join(missing_roles) if missing_roles else "None",
                                "Missing skills and tools (list)": ", ".join(missing_items) if missing_items else "None",
                                "Members": ", ".join(team_ids),
                            }
                        )

                    overview_df = pd.DataFrame(overview_rows)
                    if not overview_df.empty:

                        def _count_list_str(s: str) -> int:
                            return 0 if s == "None" else len([x for x in s.split(",") if x.strip()])

                        overview_df["Missing roles (#)"] = overview_df["Missing roles (list)"].apply(_count_list_str)
                        overview_df["Missing items (#)"] = overview_df["Missing skills and tools (list)"].apply(_count_list_str)

                        overview_df["Avg fit (sort)"] = overview_df["Avg fit"].apply(lambda x: float(x) if x != "N/A" else -1.0)

                        overview_df = overview_df.sort_values(
                            by=["Missing roles (#)", "Missing items (#)", "Avg fit (sort)"],
                            ascending=[True, True, False],
                        ).reset_index(drop=True)

                        st.dataframe(
                            overview_df[
                                [
                                    "Team",
                                    "Avg fit",
                                    "Coverage (skills and tools) %",
                                    "Avg internal sim",
                                    "Avg internal dissim",
                                    "Missing roles (list)",
                                    "Missing skills and tools (list)",
                                    "Members",
                                ]
                            ],
                            use_container_width=True,
                        )

                    overview_path = artifacts_root / "teams_overview.json"
                    overview_path.write_text(
                        json.dumps({"rows": overview_rows}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    st.caption(f"Saved: {overview_path}")

                    out_all = []
                    for t_idx, team in enumerate(teams, start=1):
                        st.markdown(f"### Team {t_idx}")

                        team_ids = [str(mem.id) for mem in team]
                        team_avg_dissim = avg_team_dissimilarity(team_ids, dissim, id2idx)
                        st.caption(f"Internal diversity (avg pairwise dissimilarity): {team_avg_dissim:.4f}")

                        rows = []
                        for m in team:
                            rawp = profile_lookup.get(m.id, {}) or {}
                            pers = rawp.get("personality", {}) or {}
                            collab = rawp.get("collaboration", {}) or {}

                            avg_dissim_tm = candidate_avg_dissim_to_teammates(
                                candidate_id=str(m.id),
                                team_ids=team_ids,
                                dissim=dissim,
                                id2idx=id2idx,
                            )

                            req_items = project.required_skills
                            cand_cov_pct = candidate_requirement_coverage_percent(m, req_items)
                            cand_matches = candidate_requirement_matches(m, req_items)

                            fit_val = project_similarities.get(m.id, None)

                            rows.append(
                                {
                                    "Candidate ID": m.id,
                                    "Role": m.role,
                                    "Req match (skills and tools)": ", ".join(cand_matches[:10]) if cand_matches else "None",
                                    "Req coverage %": round(float(cand_cov_pct), 1),
                                    "Belbin role": pers.get("Belbin_team_role", "Unknown"),
                                    "Comm style": collab.get("communication_style", "Unknown"),
                                    "Conflict style": collab.get("conflict_style", "Unknown"),
                                    "Leadership": collab.get("leadership_preference", "Unknown"),
                                    "Deadline": collab.get("deadline_discipline", "Unknown"),
                                    "Availability": m.availability_hours,
                                    "Fit": "N/A" if fit_val is None else round(float(fit_val), 4),
                                    "Avg dissim to teammates": round(float(avg_dissim_tm), 4),
                                    "Skills (sample)": ", ".join((m.skills or [])[:8]),
                                    "Tools (sample)": ", ".join((m.tools or [])[:8]),
                                }
                            )

                        out_all.append(rows)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    teams_out_path_default.write_text(
                        json.dumps(out_all, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    st.caption(f"Saved: {teams_out_path_default}")

            except Exception as e:
                st.error(f"Team building failed: {e}")



# TAB  -> Explanation Report

with tab_explain:
    st.subheader("Team Report (LLM Explanation)")
    st.caption("Button enables only when files and API key exist.")

    status_line("Teams file", teams_out_path_default)
    status_line("Candidate inputs", cand_inputs_path_default)
    status_line("Latency log", latency_log_path)

    if proj_req_path.exists():
        status_line("Project requirements (optional)", proj_req_path)
    else:
        st.write("Project requirements (optional): Not found")

    explain_btn = st.button(
        "Generate Team Report",
        type="primary",
        disabled=not (api_key.strip() and _file_ok(teams_out_path_default) and _file_ok(cand_inputs_path_default)),
    )

    if explain_btn:
        try:
            t0 = time.perf_counter()

            payload = explain_report.run_team_report(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.0,
                teams_ui_path=teams_out_path_default,
                candidate_inputs_path=cand_inputs_path_default,
                project_req_path=proj_req_path if proj_req_path.exists() else None,
                teams_summary_path=teams_summary_path if teams_summary_path.exists() else None,
            )

            log_latency(
                log_path=latency_log_path,
                stage="explanation_report",
                total_s=time.perf_counter() - t0,
                model=model,
                base_url=base_url,
            )

            save_json(explain_out_path, payload)
            st.success(f"Saved: {explain_out_path}")

            st.markdown(payload.get("result", {}).get("report_markdown", "Not specified"))

            with st.expander("Raw JSON result"):
                st.json(payload.get("result", {}))

        except Exception as e:
            st.error(f"Explanation report failed: {e}")



# TAB 3-> Evaluation

with tab_eval:
    st.subheader("Evaluation")
    st.markdown("### Quality Dashboard")

    cand_summary = read_json_if_ok(cand_eval_path)
    proj_summary = read_json_if_ok(proj_eval_path)
    teams_summary_eval = read_json_if_ok(teams_eval_path)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Candidate extraction (LLM)")
        if cand_summary:
            schema_ok_rate = float(cand_summary.get("schema_ok_rate", 0.0))
            avg_evidence_errors = float(cand_summary.get("avg_evidence_errors", 0.0))
            avg_label_errors = float(cand_summary.get("avg_label_errors", 0.0))

            details = cand_summary.get("details", []) or []
            zero_evidence = sum(1 for r in details if len(r.get("evidence_errors", [])) == 0)
            denom = max(1, len(details))
            zero_evidence_rate = zero_evidence / denom

            st.metric("Schema valid rate", f"{schema_ok_rate*100:.1f}%")
            st.metric("Evidence OK rate", f"{zero_evidence_rate*100:.1f}%")
            st.metric("Avg evidence errors", f"{avg_evidence_errors:.2f}")
            st.metric("Avg label errors", f"{avg_label_errors:.2f}")
        else:
            st.info("Run Candidate Extraction Evaluation to populate metrics.")

    with c2:
        st.markdown("#### Project extraction (LLM)")
        if proj_summary:
            schema_ok = bool(proj_summary.get("schema_ok", False))
            skills_rate = float(proj_summary.get("skills_supported_rate", 1.0))
            tools_rate = float(proj_summary.get("tools_supported_rate", 1.0))
            uns_sk = proj_summary.get("skills_unsupported", []) or []
            uns_tl = proj_summary.get("tools_unsupported", []) or []

            st.metric("Schema OK", "OK" if schema_ok else "Not OK")
            st.metric("Skills supported rate", f"{skills_rate*100:.1f}%")
            st.metric("Tools supported rate", f"{tools_rate*100:.1f}%")
            st.metric("Unsupported items", f"{len(uns_sk)+len(uns_tl)}")
        else:
            st.info("Run Project Requirements Evaluation to populate metrics.")

    with c3:
        st.markdown("#### Team evaluation (deterministic)")
        if teams_summary_eval:
            n_teams = teams_summary_eval.get("n_teams") or teams_summary_eval.get("num_teams") or teams_summary_eval.get("n") or "Not specified"
            avg_cov = teams_summary_eval.get("avg_coverage_percent") or teams_summary_eval.get("coverage_avg")
            avg_div = teams_summary_eval.get("avg_internal_dissimilarity") or teams_summary_eval.get("avg_diversity")

            st.metric("Teams evaluated", str(n_teams))
            if avg_cov is not None:
                st.metric("Avg coverage %", f"{float(avg_cov):.1f}%")
            else:
                st.caption("Avg coverage %: Not specified (check TeamEvaluator output keys)")

            if avg_div is not None:
                st.metric("Avg internal diversity", f"{float(avg_div):.4f}")
            else:
                st.caption("Avg internal diversity: Not specified (check TeamEvaluator output keys)")
        else:
            st.info("Run Team Evaluation to populate metrics.")

    st.divider()

    st.markdown("### Latency log (LLM stages)")
    if latency_log_path.exists() and latency_log_path.stat().st_size > 0:
        lines = latency_log_path.read_text(encoding="utf-8").strip().splitlines()
        rows = []
        for ln in lines:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass

        if rows:
            df_lat = pd.DataFrame(rows)
            df_lat = df_lat[df_lat["stage"].isin(["candidate_extraction", "project_extraction", "explanation_report"])]

            if "ts_utc" in df_lat.columns:
                df_lat = df_lat.sort_values("ts_utc", ascending=False)

            st.dataframe(df_lat, use_container_width=True)
            st.caption(f"Logfile: {latency_log_path}")
        else:
            st.info("Latency log exists but no valid entries found.")
    else:
        st.info("No latency log yet. Run the LLM stages once.")

    st.divider()

    st.subheader("Candidate Profile Extraction Evaluation (grounding and schema)")
    status_line("candidate_profiles_with_evidence.json", cand_profiles_evid_path)
    status_line("candidate_inputs.json", cand_inputs_path_default)

    if st.button("Run Candidate Extraction Evaluation"):
        if not (_file_ok(cand_profiles_evid_path) and _file_ok(cand_inputs_path_default)):
            st.error("Missing files. Run candidate extraction first.")
        else:
            evaluator = CandidateProfileEvaluator()
            summary = evaluator.evaluate_files(
                profiles_path=str(cand_profiles_evid_path),
                inputs_path=str(cand_inputs_path_default),
                out_path=str(cand_eval_path),
            )
            st.success(f"Saved: {cand_eval_path}")
            st.json(summary)

    st.divider()

    st.subheader("Project Requirements Extraction Evaluation (hallucination check vs description)")
    status_line("project_requirements.json", proj_req_path)
    status_line("project_description.txt", proj_desc_path)

    if st.button("Run Project Requirements Evaluation"):
        if not (_file_ok(proj_req_path) and _file_ok(proj_desc_path)):
            st.error("Missing files. Extract project requirements first.")
        else:
            evaluator = ProjectRequirementsEvaluator()
            summary = evaluator.evaluate_files(
                requirements_path=str(proj_req_path),
                description_path=str(proj_desc_path),
                out_path=str(proj_eval_path),
            )
            st.success(f"Saved: {proj_eval_path}")
            st.json(summary)

    st.divider()
    st.subheader("Team Evaluation (deterministic, non-LLM)")

    status_line("teams_ui.json", teams_out_path_default)
    status_line("candidate_profiles.json", cand_profiles_clean_path)
    status_line("project_requirements.json", proj_req_path)

    

    if st.button("Run Team Evaluation"):
        if not (_file_ok(teams_out_path_default) and _file_ok(cand_profiles_clean_path) and _file_ok(proj_req_path)):
            st.error("Missing required files. Build teams first.")
        else:
            evaluator = TeamEvaluator()
            summary = evaluator.evaluate(
                candidate_profiles_path=cand_profiles_clean_path,
                project_requirements_path=proj_req_path,
                teams_path=teams_out_path_default,
                out_path=teams_eval_path,
                candidate_ids_path=cand_ids_path if _file_ok(cand_ids_path) else None,
                candidate_dissim_path=cand_dissim_path if _file_ok(cand_dissim_path) else None,
                required_team_hours_per_week=0.0,

            )
            st.success(f"Saved: {teams_eval_path}")
            st.json(summary)

with tab_raw:
    st.subheader("Raw JSON Outputs")

    show_files = [
        cand_profiles_clean_path,
        cand_profiles_evid_path,
        cand_inputs_path_default,
        proj_req_path,
        proj_desc_path,
        cand_eval_path,
        proj_eval_path,
        teams_out_path_default,
        teams_summary_path,
        teams_eval_path,
        sim_json_path,
        explain_out_path,
        artifacts_root / "teams_overview.json",
        latency_log_path,
    ]

    for p in show_files:
        if not p.exists():
            continue
        st.markdown(f"### {p.name}")
        try:
            if p.suffix.lower() == ".txt":
                st.code(p.read_text(encoding="utf-8"))
            elif p.suffix.lower() in [".json", ".jsonl"]:
                if p.suffix.lower() == ".jsonl":
                    lines = p.read_text(encoding="utf-8").strip().splitlines()[-50:]
                    rows = []
                    for ln in lines:
                        try:
                            rows.append(json.loads(ln))
                        except Exception:
                            pass
                    st.json(rows)
                else:
                    st.json(json.loads(p.read_text(encoding="utf-8")))
            else:
                st.caption(f"(Binary file) {p}")
        except Exception as e:
            st.warning(f"Could not display {p.name}: {e}")
