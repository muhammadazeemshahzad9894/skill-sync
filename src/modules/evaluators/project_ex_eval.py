# src/modules/evaluation/project_evaluator.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


class ProjectRequirementsEvaluator:
    """
    Local evaluator for extracted project requirements JSON.

    Validates:
      - Schema completeness
      - Types (skills/tools must be lists[str])
      - Uniqueness
      - Verbatim (or alias) grounding in the project description text

    Reads:
      - project_requirements.json (dict)
      - project_description.txt (raw user project text)  [or pass as string from UI]

    Writes:
      - project_evaluation_summary.json
    """

    REQUIRED_PATHS = [
        "technical_requirements.skills",
        "technical_requirements.tools",
    ]

    # Optional alias map for canonicalization you do in prompt
    # Key: canonical label, Value: list of alias substrings that count as support
    ALIAS_SUPPORT = {
        "Amazon Web Services (AWS)": ["aws", "amazon web services"],
        "Google Cloud": ["gcp", "google cloud platform", "google cloud"],
        "Oracle Cloud Infrastructure (OCI)": ["oci", "oracle cloud"],
        "IBM Cloud Or Watson": ["ibm cloud", "watson"],
        "Digital Ocean": ["digitalocean", "digital ocean"],
        "ASP.NET CORE": ["asp.net core", ".net core"],
        ".NET (5+)": [".net 5", ".net 6", ".net 7", ".net 8", "modern .net"],
        ".NET Framework (1.0 - 4.8)": [".net framework"],
        "Torch/PyTorch": ["pytorch", "torch"],
        "Bash/Shell (all shells)": ["bash", "shell scripting", "shell"],
        "Visual Basic (.Net)": ["vb.net", "visual basic .net"],
    }

    @staticmethod
    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    @staticmethod
    def get_path(obj: Dict[str, Any], path: str):
        cur: Any = obj
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    @staticmethod
    def is_list_of_str(x) -> bool:
        return isinstance(x, list) and all(isinstance(i, str) for i in x)

    def item_supported(self, item: str, description: str) -> bool:
        """
        True if item appears in description OR an allowed alias appears.
        """
        d = self.normalize(description)
        it = self.normalize(item)

        # direct substring support
        if it and it in d:
            return True

        # alias support for canonicalized labels
        aliases = self.ALIAS_SUPPORT.get(item, [])
        for a in aliases:
            if self.normalize(a) in d:
                return True

        return False

    def evaluate(self, requirements: Dict[str, Any], description: str) -> Dict[str, Any]:
        summary = {
            "schema_ok": True,
            "missing_keys": [],
            "type_errors": [],
            "skills_count": 0,
            "tools_count": 0,
            "skills_duplicates": 0,
            "tools_duplicates": 0,
            "skills_unsupported": [],
            "tools_unsupported": [],
            "skills_supported_rate": 1.0,
            "tools_supported_rate": 1.0,
        }

        # schema
        for p in self.REQUIRED_PATHS:
            if self.get_path(requirements, p) is None:
                summary["schema_ok"] = False
                summary["missing_keys"].append(p)

        skills = self.get_path(requirements, "technical_requirements.skills") or []
        tools = self.get_path(requirements, "technical_requirements.tools") or []

        # types
        if not self.is_list_of_str(skills):
            summary["schema_ok"] = False
            summary["type_errors"].append("technical_requirements.skills must be list[str]")
            skills = []
        if not self.is_list_of_str(tools):
            summary["schema_ok"] = False
            summary["type_errors"].append("technical_requirements.tools must be list[str]")
            tools = []

        # counts + duplicates
        summary["skills_count"] = len(skills)
        summary["tools_count"] = len(tools)
        summary["skills_duplicates"] = len(skills) - len(set(skills))
        summary["tools_duplicates"] = len(tools) - len(set(tools))

        # grounding
        sup_sk = [x for x in skills if self.item_supported(x, description)]
        sup_tl = [x for x in tools if self.item_supported(x, description)]
        uns_sk = [x for x in skills if x not in sup_sk]
        uns_tl = [x for x in tools if x not in sup_tl]

        summary["skills_unsupported"] = uns_sk
        summary["tools_unsupported"] = uns_tl
        summary["skills_supported_rate"] = (len(sup_sk) / len(skills)) if skills else 1.0
        summary["tools_supported_rate"] = (len(sup_tl) / len(tools)) if tools else 1.0

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

    def evaluate_files(self, requirements_path: str, description_path: str, out_path: str) -> Dict[str, Any]:
        req = self.load_json(requirements_path)
        with open(description_path, "r", encoding="utf-8") as f:
            desc = f.read()
        summary = self.evaluate(req, desc)
        self.save_json(summary, out_path)
        return summary
