from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


ALLOWED = {
    "communication_style": {"Async", "Sync", "Mixed", "Unknown"},
    "conflict_style": {"Avoid", "Direct", "Mediation", "Unknown"},
    "leadership_preference": {"Lead", "Follow", "Shared", "Unknown"},
    "deadline_discipline": {"Strict", "Flexible", "Depends", "Unknown"},
    "learning_orientation": {"High", "Medium", "Low", "Unknown"},
    "knowledge_sharing": {"High", "Medium", "Low", "Unknown"},
    "team_role": {
        "Teamworker",
        "Shaper",
        "Coordinator",
        "Implementer",
        "Monitor-Evaluator",
        "Plant",
        "Resource-Investigator",
        "Completer-Finisher",
        "Unknown",
    },
}


def words_count(s: str) -> int:
    return len([w for w in (s or "").strip().split() if w])


def is_list_of_str(x) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def get_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def evidence_ok(evidence: str) -> bool:
    if evidence == "":
        return True
    wc = words_count(evidence)
    return 3 <= wc <= 20


def substring_in(source: str, evidence: str) -> bool:
    if evidence == "":
        return True
    return normalize(evidence) in normalize(source)


class CandidateProfileEvaluator:
    """
    Evaluates candidate extraction output from extractor.py.

    Inputs:
      - candidate_profiles_with_evidence.json  (list of profiles)
      - candidate_inputs.json                  (dict: id -> input text shown to LLM)

    Produces:
      - candidate_extraction_eval.json
    """

    REQUIRED_PATHS = [
        "id",
        "constraints.weekly_availability_hours",
        "metadata.dev_type",
        "metadata.work_experience_years",
        "metadata.years_code",
        "metadata.employment",
        "metadata.org_size",
        "metadata.industry",
        "technical.skills",
        "technical.tools",
        "personality.Belbin_team_role",
        "collaboration.communication_style",
        "collaboration.conflict_style",
        "collaboration.leadership_preference",
        "collaboration.deadline_discipline",
        "learning_behavior.learning_orientation",
        "learning_behavior.knowledge_sharing",
        "evidence.personality.team_role",
        "evidence.collaboration.communication_style",
        "evidence.collaboration.conflict_style",
        "evidence.collaboration.leadership_preference",
        "evidence.collaboration.deadline_discipline",
        "evidence.learning_behavior.learning_orientation",
        "evidence.learning_behavior.knowledge_sharing",
    ]

    def evaluate_one(self, profile: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        out = {
            "id": profile.get("id", "Unknown"),
            "schema_ok": True,
            "missing_keys": [],
            "type_errors": [],
            "label_errors": [],
            "evidence_errors": [],
            "technical_duplicates": {"skills": 0, "tools": 0},
        }

        # schema keys present
        for p in self.REQUIRED_PATHS:
            if get_path(profile, p) is None:
                out["schema_ok"] = False
                out["missing_keys"].append(p)

        # types
        skills = get_path(profile, "technical.skills") or []
        tools = get_path(profile, "technical.tools") or []
        if not is_list_of_str(skills):
            out["schema_ok"] = False
            out["type_errors"].append("technical.skills must be list[str]")
            skills = []
        if not is_list_of_str(tools):
            out["schema_ok"] = False
            out["type_errors"].append("technical.tools must be list[str]")
            tools = []

        out["technical_duplicates"]["skills"] = len(skills) - len(set(skills))
        out["technical_duplicates"]["tools"] = len(tools) - len(set(tools))

        # validate categorical labels
        team_role = get_path(profile, "personality.Belbin_team_role") or "Unknown"
        if team_role not in ALLOWED["team_role"]:
            out["label_errors"].append(f"Invalid team_role: {team_role}")

        comm = get_path(profile, "collaboration.communication_style") or "Unknown"
        conf = get_path(profile, "collaboration.conflict_style") or "Unknown"
        lead = get_path(profile, "collaboration.leadership_preference") or "Unknown"
        dead = get_path(profile, "collaboration.deadline_discipline") or "Unknown"
        lo = get_path(profile, "learning_behavior.learning_orientation") or "Unknown"
        ks = get_path(profile, "learning_behavior.knowledge_sharing") or "Unknown"

        if comm not in ALLOWED["communication_style"]:
            out["label_errors"].append(f"Invalid communication_style: {comm}")
        if conf not in ALLOWED["conflict_style"]:
            out["label_errors"].append(f"Invalid conflict_style: {conf}")
        if lead not in ALLOWED["leadership_preference"]:
            out["label_errors"].append(f"Invalid leadership_preference: {lead}")
        if dead not in ALLOWED["deadline_discipline"]:
            out["label_errors"].append(f"Invalid deadline_discipline: {dead}")
        if lo not in ALLOWED["learning_orientation"]:
            out["label_errors"].append(f"Invalid learning_orientation: {lo}")
        if ks not in ALLOWED["knowledge_sharing"]:
            out["label_errors"].append(f"Invalid knowledge_sharing: {ks}")

        
        sources: Dict[str, str] = {}
        for line in (input_text or "").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                sources[k.strip()] = v.strip()

        personality_text = sources.get("PersonalityText", "")
        ai_sources = " ".join(
            [
                sources.get("AIToolCurrentlyUsing", ""),
                sources.get("AIBen", ""),
                sources.get("AIComplex", ""),
                sources.get("AIChallenges", ""),
                sources.get("AIEthics", ""),
            ]
        )

        def check_field(label_val: str, evidence_val: str, source_text: str, field_name: str):
            if label_val == "Unknown" and evidence_val != "":
                out["evidence_errors"].append(f"{field_name}: label Unknown but evidence not empty")
            if label_val != "Unknown" and evidence_val == "":
                out["evidence_errors"].append(f"{field_name}: label set but evidence empty")
            if not evidence_ok(evidence_val):
                out["evidence_errors"].append(f"{field_name}: evidence must be 3–8 words (or empty)")
            if not substring_in(source_text, evidence_val):
                out["evidence_errors"].append(f"{field_name}: evidence not found verbatim in expected source")

        check_field(team_role, get_path(profile, "evidence.personality.team_role") or "", personality_text, "team_role")
        check_field(comm, get_path(profile, "evidence.collaboration.communication_style") or "", personality_text, "communication_style")
        check_field(conf, get_path(profile, "evidence.collaboration.conflict_style") or "", personality_text, "conflict_style")
        check_field(lead, get_path(profile, "evidence.collaboration.leadership_preference") or "", personality_text, "leadership_preference")
        check_field(dead, get_path(profile, "evidence.collaboration.deadline_discipline") or "", personality_text, "deadline_discipline")
        check_field(lo, get_path(profile, "evidence.learning_behavior.learning_orientation") or "", ai_sources, "learning_orientation")
        check_field(ks, get_path(profile, "evidence.learning_behavior.knowledge_sharing") or "", personality_text, "knowledge_sharing")

        return out

    def evaluate(self, profiles: List[Dict[str, Any]], inputs: Dict[str, str]) -> Dict[str, Any]:
        results = []
        for p in profiles:
            cid = str(p.get("id", "Unknown"))
            inp = inputs.get(cid, "")
            results.append(self.evaluate_one(p, inp))

        summary = {
            "n": len(results),
            "schema_ok_rate": sum(1 for r in results if r["schema_ok"]) / len(results) if results else 0.0,
            "avg_label_errors": sum(len(r["label_errors"]) for r in results) / len(results) if results else 0.0,
            "avg_evidence_errors": sum(len(r["evidence_errors"]) for r in results) / len(results) if results else 0.0,
            "details": results,
        }
        return summary

    @staticmethod
    def load_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(obj: Any, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def evaluate_files(self, profiles_path: str, inputs_path: str, out_path: str) -> Dict[str, Any]:
        profiles = self.load_json(profiles_path)
        inputs = self.load_json(inputs_path)
        summary = self.evaluate(profiles, inputs)
        self.save_json(summary, out_path)
        return summary
