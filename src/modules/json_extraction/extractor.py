from __future__ import annotations

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"



SYSTEM_PROMPT = """You are an information extraction assistant.
Extract ONLY what is supported by the input text. If unclear or missing, use "Unknown".
Return VALID JSON ONLY (no markdown, no extra text).
Do not invent facts, skills, tools, or behaviors that are not stated or clearly implied by the text.
"""

USER_PROMPT = """Extract ONLY what is supported by the candidate profile.
If information is unclear, ambiguous, or missing, use "Unknown".
Do NOT invent facts, skills, tools, roles, or behaviors.
Return VALID JSON ONLY (no markdown, no extra text).

------------------------------------------------------------
OUTPUT FORMAT (EXACT JSON SCHEMA)
------------------------------------------------------------

{
  "id": "...",
  "constraints": {
    "weekly_availability_hours": "..."
  },
  "metadata": {
    "dev_type": "...",
    "work_experience_years": "...",
    "years_code": "...",
    "employment": "...",
    "org_size": "...",
    "industry": "..."
  },
  "technical": {
    "skills": [],
    "tools": []
  },
  "personality": {
    "Belbin_team_role": "..."
  },
  "collaboration": {
    "communication_style": "...",
    "conflict_style": "...",
    "leadership_preference": "...",
    "deadline_discipline": "..."
  },
  "learning_behavior": {
    "learning_orientation": "...",
    "knowledge_sharing": "..."
  },
  "evidence": {
    "personality": {
      "team_role": ""
    },
    "collaboration": {
      "communication_style": "",
      "conflict_style": "",
      "leadership_preference": "",
      "deadline_discipline": ""
    },
    "learning_behavior": {
      "learning_orientation": "",
      "knowledge_sharing": ""
    }
  }
}

------------------------------------------------------------
HARD RULES
------------------------------------------------------------

- Return ONLY valid JSON.
- Do NOT omit any keys.
- If a value is unknown or unsupported, use:
  - "Unknown" for string fields
  - [] for list fields
  - "" for evidence fields

------------------------------------------------------------
COPY-EXACTLY RULES (NO INFERENCE)
------------------------------------------------------------

These rules apply to identifiers, metadata, constraints, and ALL technical fields.

- id must be copied exactly from Candidate ID if present, else "Unknown".
- constraints.weekly_availability_hours must be copied exactly from WeeklyAvailabilityHours if present, else "Unknown".
- metadata.dev_type must be copied exactly from DevType if present, else "Unknown".
- metadata.work_experience_years must be copied exactly from WorkExp if present, else "Unknown".
- metadata.years_code must be copied exactly from YearsCode if present, else "Unknown".
- metadata.employment must be copied exactly from Employment if present, else "Unknown".
- metadata.org_size must be copied exactly from OrgSize if present, else "Unknown".
- metadata.industry must be copied exactly from Industry if present, else "Unknown".

IMPORTANT:
If metadata.dev_type starts with "Other",
or equals "Other (please specify):",
set metadata.dev_type to "Unknown".

------------------------------------------------------------
TECHNICAL CLASSIFICATION RULES (STRICT, VERBATIM)
------------------------------------------------------------

- technical.skills MUST include ONLY programming languages, frameworks, libraries, databases, and cloud platforms explicitly mentioned.
- technical.tools MUST include ONLY developer tools (IDEs, editors, CLIs, package managers).
- Do NOT include survey option labels or workflow phrases
  (e.g., "DevOps function", "Continuous delivery").
- Every extracted technical item MUST appear verbatim in the input text.
- Do NOT infer, normalize, or generalize technical items.

------------------------------------------------------------
ALLOWED VALUES (CATEGORICAL FIELDS)
------------------------------------------------------------

- team_role:
  Teamworker, Shaper, Coordinator, Implementer, Monitor-Evaluator,
  Plant, Resource-Investigator, Completer-Finisher.

- communication_style: Async, Sync, Mixed, Unknown
- conflict_style: Avoid, Direct, Mediation, Unknown
- leadership_preference: Lead, Follow, Shared, Unknown
- deadline_discipline: Strict, Flexible, Depends, Unknown
- learning_orientation: High, Medium, Low, Unknown
- knowledge_sharing: High, Medium, Low, Unknown

------------------------------------------------------------
SEMANTIC MAPPING RULES (FOR GENERALIZATION)
------------------------------------------------------------

These rules apply ONLY to behavioral categorical fields:
personality, collaboration, and learning_behavior.

You MAY assign an allowed value even if the exact label word
(e.g., "Shaper", "Async") does NOT appear,
AS LONG AS the text clearly and unambiguously describes
the corresponding behavior.

If signals are mixed, weak, or ambiguous, return "Unknown".

------------------------------------------------------------
BELBIN TEAM ROLE MAPPING (PERSONALITY)
------------------------------------------------------------

- Shaper: pushes progress, challenges others, results-driven, urgency
- Teamworker: supportive, harmony-focused, resolves tension, cooperative
- Coordinator: organizes people or tasks, aligns goals, delegates responsibility
- Implementer: practical execution, structured planning, turns ideas into action
- Monitor-Evaluator: analytical, cautious, evaluates pros and cons
- Plant: creative, idea generation, novel or unconventional approaches
- Resource-Investigator: explores opportunities, networking, outward-looking
- Completer-Finisher: detail-oriented, polishing work, quality control

------------------------------------------------------------
COLLABORATION STYLE MAPPING
------------------------------------------------------------

communication_style:
- Async: prefers written updates, messaging, minimal meetings
- Sync: prefers live discussion, calls, meetings
- Mixed: explicitly uses both depending on context

conflict_style:
- Avoid: avoids confrontation, de-escalates tension
- Direct: addresses issues openly and directly
- Mediation: bridges viewpoints, seeks compromise

leadership_preference:
- Lead: directional influence / ownership / organizing / motivating
- Follow: supports leader / focuses on assigned tasks / not leading
- Shared: explicitly shared decisions/ownership
- Unknown: weak or mixed signals

deadline_discipline:
- Strict: emphasizes meeting deadlines, plans ahead
- Flexible: adapts or negotiates timelines
- Depends: conditional on context or workload

------------------------------------------------------------
LEARNING BEHAVIOR (SEMANTIC RUBRIC)
------------------------------------------------------------

learning_orientation:
- High: AI used for learning, understanding, planning, reasoning
- Medium: AI used mainly for execution support or productivity
- Low: minimal or rare AI usage
- Unknown: no AI usage information present

knowledge_sharing:
- High: explicit mentoring/teaching/explaining/guiding
- Medium: explicit sharing ideas, feedback, contributing insights
- Low: mostly own tasks/execution, little mention of helping others
- Unknown: no explicit knowledge-sharing signals

CONSISTENCY CONSTRAINT (MANDATORY):
The label must be directly implied by the evidence quote.
If evidence implies a different label, change the label to match the evidence.
Do not output "Shared" unless the evidence quote explicitly indicates shared or collective decision-making.

IMPORTANT CONSTRAINT:
- Do NOT use inclusiveness, respect, feeling heard, collaboration, or communication values
  as evidence of knowledge_sharing.
- knowledge_sharing requires explicit contribution of knowledge, ideas, feedback, mentoring, or explanation.

------------------------------------------------------------
EVIDENCE RULES
------------------------------------------------------------

General:
- Evidence must be grounded in the input text.
- Evidence must be a SHORT verbatim excerpt (not a full sentence).
- Evidence must NOT include a trailing period.
- If a field value is "Unknown", its evidence MUST be "".

Per-field evidence requirements:
- evidence.personality.team_role: 3–8 words from PersonalityText (or "" if Unknown).
- evidence.collaboration.*: 3–8 words from PersonalityText (or "" if Unknown).
- evidence.learning_behavior.learning_orientation: 3–8 words copied EXACTLY from ONE of:
  AIToolCurrentlyUsing, AIBen, AIComplex, AIChallenges, AIEthics.
- evidence.learning_behavior.knowledge_sharing: 3–8 words from PersonalityText not ProfessionalTech.

Consistency rule:
- Evidence must directly support the chosen label.
- If you cannot find evidence supporting the chosen label, set the label to "Unknown" and evidence to "".
- Never provide evidence that contradicts the label.

Before returning the final JSON, verify:
- All required keys are present (including nested evidence keys).
- All categorical values belong to the allowed sets.
- Each evidence string is 3–8 words and appears verbatim in the correct source text.
- If a field is "Unknown", its evidence is "".
- Evidence supports (does not contradict) the selected label.
If any check fails, correct it before returning.

Now extract from the following candidate text:
"""


@dataclass
class ExtractorConfig:
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    limit: int = 3
    retries: int = 1
    request_timeout_s: int = 60
    sleep_between_calls_s: float = 0.2
    debug: bool = False


class CandidateProfileExtractor:
    """
    Pipeline-friendly extractor:
    - builds a candidate text from a CSV row
    - calls LLM to output a strict JSON schema
    - provides batch extraction + saving helpers
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        config: Optional[ExtractorConfig] = None,
    ):
        self.config = config or ExtractorConfig()

        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")

        # OpenRouter recommended headers (safe even if ignored)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.request_timeout_s,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "SkillSync",
            },
        )

        if self.config.debug:
            print(f"[DEBUG] base_url: {base_url}", flush=True)
            print(f"[DEBUG] model: {self.config.model}", flush=True)
            print(f"[DEBUG] timeout(s): {self.config.request_timeout_s}", flush=True)

  
    @staticmethod
    def as_str(v: Any) -> str:
        if v is None:
            return "Unknown"
        try:
            if pd.isna(v):
                return "Unknown"
        except Exception:
            pass
        s = str(v).strip()
        return s if s else "Unknown"

    @staticmethod
    def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Helps when CSV headers vary slightly.
        """
        rename_map = {}
        if "CandidateNumber" in df.columns and "Candidate ID" not in df.columns:
            rename_map["CandidateNumber"] = "Candidate ID"
        if "AIToolCurrently Using" in df.columns and "AIToolCurrentlyUsing" not in df.columns:
            rename_map["AIToolCurrently Using"] = "AIToolCurrentlyUsing"
        if "OpSysPersonal use" in df.columns and "OpSysPersonalUse" not in df.columns:
            rename_map["OpSysPersonal use"] = "OpSysPersonalUse"
        return df.rename(columns=rename_map) if rename_map else df

    @staticmethod
    def safe_json_loads(raw: str) -> Dict[str, Any]:
        """
        Extract the first {...} JSON object from model output.
        """
        raw = (raw or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found. Raw preview: {raw[:200]}")
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}. Raw preview: {candidate[:200]}")

  
    def build_profile_text(self, r: Dict[str, Any]) -> str:
        """
        Constructs exactly what the LLM sees.
        """
        fields = [
            ("Candidate ID", self.as_str(r.get("Candidate ID"))),
            ("WeeklyAvailabilityHours", self.as_str(r.get("WeeklyAvailabilityHours"))),

            ("Employment", self.as_str(r.get("Employment"))),
            ("YearsCode", self.as_str(r.get("YearsCode"))),
            ("DevType", self.as_str(r.get("DevType"))),
            ("OrgSize", self.as_str(r.get("OrgSize"))),
            ("Industry", self.as_str(r.get("Industry"))),
            ("WorkExp", self.as_str(r.get("WorkExp"))),
            ("ICorPM", self.as_str(r.get("ICorPM"))),

            ("LanguageHaveWorkedWith", self.as_str(r.get("LanguageHaveWorkedWith"))),
            ("LanguageAdmired", self.as_str(r.get("LanguageAdmired"))),
            ("DatabaseHaveWorkedWith", self.as_str(r.get("DatabaseHaveWorkedWith"))),
            ("PlatformHaveWorkedWith", self.as_str(r.get("PlatformHaveWorkedWith"))),
            ("WebframeHaveWorkedWith", self.as_str(r.get("WebframeHaveWorkedWith"))),
            ("MiscTechHaveWorkedWith", self.as_str(r.get("MiscTechHaveWorkedWith"))),
            ("ToolsTechHaveWorkedWith", self.as_str(r.get("ToolsTechHaveWorkedWith"))),
            ("NEWCollabToolsHaveWorkedWith", self.as_str(r.get("NEWCollabToolsHaveWorkedWith"))),
            ("OpSysPersonalUse", self.as_str(r.get("OpSysPersonalUse"))),

            ("AISearchDevHaveWorkedWith", self.as_str(r.get("AISearchDevHaveWorkedWith"))),
            ("AIToolCurrentlyUsing", self.as_str(r.get("AIToolCurrentlyUsing"))),
            ("AIBen", self.as_str(r.get("AIBen"))),
            ("AIComplex", self.as_str(r.get("AIComplex"))),
            ("AIEthics", self.as_str(r.get("AIEthics"))),
            ("AIChallenges", self.as_str(r.get("AIChallenges"))),

            ("Frustration", self.as_str(r.get("Frustration"))),
            ("PersonalityText", self.as_str(r.get("PersonalityText"))),

            ("ProfessionalTech", self.as_str(r.get("ProfessionalTech"))),
            ("ProfessionalCloud", self.as_str(r.get("ProfessionalCloud"))),
        ]
        return "\n".join([f"{k}: {v}" for k, v in fields])

    
    def extract_one(self, row: Dict[str, Any], i: int = -1) -> Dict[str, Any]:
        text = self.build_profile_text(row)

        if self.config.debug:
            cid = self.as_str(row.get("Candidate ID"))
            preview = text.replace("\n", " | ")
            print(f"[DEBUG] Extracting row={i} CID={cid}", flush=True)
            print(f"[DEBUG] Input preview: {preview[:200]}...", flush=True)

        resp = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + "\n\n" + text},
            ],
        )

        raw = resp.choices[0].message.content

        if self.config.debug:
            print(f"[DEBUG] Raw output preview: {(raw or '')[:200]}...", flush=True)

        profile = self.safe_json_loads(raw)

        # Ensure id is correct (copy-exactly rule)
        profile["id"] = self.as_str(row.get("Candidate ID"))

        if self.config.sleep_between_calls_s > 0:
            time.sleep(self.config.sleep_between_calls_s)

        return profile

    def extract_all(self, df: pd.DataFrame, n: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        df = self.rename_columns(df)
        limit = min(n or self.config.limit, len(df))

        profiles: List[Dict[str, Any]] = []
        inputs: Dict[str, str] = {}

        for i in range(limit):
            row = df.iloc[i].to_dict()
            cid = self.as_str(row.get("Candidate ID"))
            text = self.build_profile_text(row)
            inputs[cid] = text

            last_err: Optional[str] = None
            for attempt in range(self.config.retries + 1):
                try:
                    prof = self.extract_one(row, i=i)
                    profiles.append(prof)
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    if self.config.debug:
                        print(f"[ERROR] row={i} CID={cid} attempt={attempt+1} error={last_err}", flush=True)

            if last_err is not None:
                profiles.append({"id": cid, "error": last_err})

        return profiles, inputs

   
    @staticmethod
    def save_json(obj: Any, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def strip_evidence(profile: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(profile)
        p.pop("evidence", None)
        return p



def run_extraction(
    csv_path: str = "data/skillsync_dataset_v1_draft_with_candidate_id.csv",
    out_clean: str = "data/artifacts/extraction/candidate_profiles.json",
    out_with_evidence: Optional[str] = "data/artifacts/extraction/candidate_profiles_with_evidence.json",
    out_inputs: Optional[str] = "data/artifacts/extraction/candidate_inputs.json",
    model: str = "openai/gpt-4o-mini",
    n: int = 10,
) -> None:
    """
    Reads CSV -> extracts profiles -> saves JSON outputs.
    Requires env var:
      OPENROUTER_API_KEY
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    print(" Loading dataset:", csv_path, flush=True)
    df = pd.read_csv(csv_path)

    cfg = ExtractorConfig(model=model, limit=n, debug=True)
    extractor = CandidateProfileExtractor(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        config=cfg,
    )

    print(" Running profile extraction...", flush=True)
    profiles, inputs = extractor.extract_all(df, n=n)

    print(" Saving outputs...", flush=True)
    if out_with_evidence:
        extractor.save_json(profiles, out_with_evidence)
    extractor.save_json([extractor.strip_evidence(p) for p in profiles], out_clean)
    if out_inputs:
        extractor.save_json(inputs, out_inputs)

    print(f"Saved: {out_clean}", flush=True)
    if out_with_evidence:
        print(f" Saved: {out_with_evidence}", flush=True)
    if out_inputs:
        print(f" Saved: {out_inputs}", flush=True)


if __name__ == "__main__":
   
    run_extraction()
