from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


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

LIST_PATHS = ["technical.skills", "technical.tools"]

ALLOWED_VALUES = {
    "personality.Belbin_team_role": {
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
    "collaboration.communication_style": {"Async", "Sync", "Mixed", "Unknown"},
    "collaboration.conflict_style": {"Avoid", "Direct", "Mediation", "Unknown"},
    "collaboration.leadership_preference": {"Lead", "Follow", "Shared", "Unknown"},
    "collaboration.deadline_discipline": {"Strict", "Flexible", "Depends", "Unknown"},
    "learning_behavior.learning_orientation": {"High", "Medium", "Low", "Unknown"},
    "learning_behavior.knowledge_sharing": {"High", "Medium", "Low", "Unknown"},
}

EVIDENCE_RULES: Dict[str, Tuple[int, int]] = {
    "evidence.personality.team_role": (3, 8),
    "evidence.collaboration.communication_style": (3, 8),
    "evidence.collaboration.conflict_style": (3, 8),
    "evidence.collaboration.leadership_preference": (3, 8),
    "evidence.collaboration.deadline_discipline": (3, 8),
    "evidence.learning_behavior.learning_orientation": (3, 8),
    "evidence.learning_behavior.knowledge_sharing": (3, 8),
}

SKILL_SOURCE_FIELDS = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "MiscTechHaveWorkedWith",
]

TOOL_SOURCE_FIELDS = [
    "ToolsTechHaveWorkedWith",
    "NEWCollabToolsHaveWorkedWith",
]

AI_SOURCE_FIELDS = [
    "AIToolCurrentlyUsing",
    "AIBen",
    "AIComplex",
    "AIChallenges",
    "AIEthics",
]


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def word_count(s: str) -> int:
    return 0 if not s else len([w for w in s.split() if w])


def get_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def is_list_of_str(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, str) for i in x)


def parse_profile_text(text: str) -> Dict[str, str]:
    fm: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm


def tokenize_field(value: str) -> List[str]:
    if not value or value == "Unknown":
        return []
    parts = re.split(r"[;\n,]+", value)
    return [normalize_ws(p) for p in parts if normalize_ws(p)]


def evidence_ok(evidence: Any, source: str, mn: int, mx: int) -> bool:
    if evidence in ("", None):
        return True
    ev = normalize_ws(str(evidence))
    if not ev:
        return True
    src = normalize_ws(source)
    if ev not in src:
        return False
    wc = word_count(ev)
    return mn <= wc <= mx


def verbatim_rate(items: List[str], text: str) -> float:
    if not items:
        return 1.0
    src = normalize_ws(text)
    ok = sum(1 for x in items if normalize_ws(x) in src)
    return ok / len(items)


class CandidateProfileEvaluator:
    def has_all_required(self, profile: Dict[str, Any]):
        missing, type_bad = [], []
        for p in REQUIRED_PATHS:
            if get_path(profile, p) is None:
                missing.append(p)
        for p in LIST_PATHS:
            v = get_path(profile, p)
            if v is not None and not isinstance(v, list):
                type_bad.append(p)
        return missing, type_bad

    def evaluate(self, profiles: List[Dict[str, Any]], inputs: Dict[str, str]) -> Dict[str, Any]:
        report = {
            "schema_total": 0,
            "schema_ok": 0,
            "skills_text_verbatim_rates": [],
            "tools_text_verbatim_rates": [],
            "skills_source_fail": Counter(),
            "tools_source_fail": Counter(),
            "invalid_categorical": Counter(),
            "evidence_valid": Counter(),
            "evidence_present": Counter(),
            "evidence_invalid": Counter(),
            "missing_keys": Counter(),
            "type_errors": Counter(),
            "ids_missing_input_text": [],
        }

        for prof in profiles:
            report["schema_total"] += 1
            cid = str(prof.get("id", "Unknown"))
            text = inputs.get(cid, "")
            if not text:
                report["ids_missing_input_text"].append(cid)

            missing, type_bad = self.has_all_required(prof)
            if not missing and not type_bad:
                report["schema_ok"] += 1
            for m in missing:
                report["missing_keys"][m] += 1
            for t in type_bad:
                report["type_errors"][t] += 1

            for path, allowed in ALLOWED_VALUES.items():
                v = get_path(prof, path)
                if v is not None and v not in allowed:
                    report["invalid_categorical"][f"{path}:{v}"] += 1

            skills = get_path(prof, "technical.skills") or []
            tools = get_path(prof, "technical.tools") or []

            if not is_list_of_str(skills):
                skills = []
            if not is_list_of_str(tools):
                tools = []

            report["skills_text_verbatim_rates"].append(verbatim_rate(skills, text))
            report["tools_text_verbatim_rates"].append(verbatim_rate(tools, text))

            fm = parse_profile_text(text)

            skill_tokens = set()
            for k in SKILL_SOURCE_FIELDS:
                skill_tokens.update(tokenize_field(fm.get(k, "")))

            tool_tokens = set()
            for k in TOOL_SOURCE_FIELDS:
                tool_tokens.update(tokenize_field(fm.get(k, "")))

            for x in skills:
                if normalize_ws(x) and normalize_ws(x) not in skill_tokens:
                    report["skills_source_fail"][x] += 1

            for x in tools:
                if normalize_ws(x) and normalize_ws(x) not in tool_tokens:
                    report["tools_source_fail"][x] += 1

            personality_text = fm.get("PersonalityText", "")
            ai_text = " ".join([fm.get(k, "") for k in AI_SOURCE_FIELDS])

            source_map = {
                "evidence.personality.team_role": personality_text,
                "evidence.collaboration.communication_style": personality_text,
                "evidence.collaboration.conflict_style": personality_text,
                "evidence.collaboration.leadership_preference": personality_text,
                "evidence.collaboration.deadline_discipline": personality_text,
                "evidence.learning_behavior.learning_orientation": ai_text,
                "evidence.learning_behavior.knowledge_sharing": personality_text,
            }

            for ev_path, (mn, mx) in EVIDENCE_RULES.items():
                ev = get_path(prof, ev_path)
                if normalize_ws(ev or ""):
                    report["evidence_present"][ev_path] += 1
                if evidence_ok(ev, source_map.get(ev_path, ""), mn, mx):
                    report["evidence_valid"][ev_path] += 1
                else:
                    report["evidence_invalid"][ev_path] += 1

        n = report["schema_total"]
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

        return {
            "schema_valid_rate": report["schema_ok"] / n if n else 0.0,
            "skills_verbatim_in_text_avg": avg(report["skills_text_verbatim_rates"]),
            "tools_verbatim_in_text_avg": avg(report["tools_text_verbatim_rates"]),
            "skills_source_fail_total": sum(report["skills_source_fail"].values()),
            "tools_source_fail_total": sum(report["tools_source_fail"].values()),
            "top_skills_source_fail": report["skills_source_fail"].most_common(20),
            "top_tools_source_fail": report["tools_source_fail"].most_common(20),
            "invalid_categorical_total": sum(report["invalid_categorical"].values()),
            "top_invalid_categorical": report["invalid_categorical"].most_common(20),
            "evidence_valid_rates": {
                k: report["evidence_valid"][k] / n if n else 0.0 for k in EVIDENCE_RULES
            },
            "evidence_present_rates": {
                k: report["evidence_present"][k] / n if n else 0.0 for k in EVIDENCE_RULES
            },
            "evidence_invalid_counts": report["evidence_invalid"].most_common(20),
            "top_missing_keys": report["missing_keys"].most_common(20),
            "top_type_errors": report["type_errors"].most_common(20),
            "ids_missing_input_text": report["ids_missing_input_text"],
        }

    @staticmethod
    def load_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(obj: Any, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def evaluate_files(self, profiles_path: str, inputs_path: str, out_path: str):
        profiles = self.load_json(profiles_path)
        inputs = self.load_json(inputs_path)
        summary = self.evaluate(profiles, inputs)
        self.save_json(summary, out_path)
        return summary


if __name__ == "__main__":
    profiles_path = "data/artifacts/extraction/candidate_profiles_with_evidence.json"
    inputs_path = "data/artifacts/extraction/candidate_inputs.json"
    out_path = "data/artifacts/extraction/candidate_extraction_eval.json"

    evaluator = CandidateProfileEvaluator()
    summary = evaluator.evaluate_files(profiles_path, inputs_path, out_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
