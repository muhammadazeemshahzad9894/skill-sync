
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagPaths:
    project_root: Path

    def _env_or_first_existing(self, env_key: str, candidates: list[Path]) -> Path:
        override = os.getenv(env_key, "").strip()
        if override:
            return Path(override)

        for p in candidates:
            if p.exists():
                return p

        return candidates[0]

    @property
    def candidate_inputs(self) -> Path:
        return self._env_or_first_existing(
            "SKILLSYNC_CANDIDATE_INPUTS",
            [
                self.project_root / "data" / "artifacts" / "extraction" / "candidate_inputs.json",
                self.project_root / "data" / "candidate_inputs.json",
            ],
        )

    @property
    def teams_ui(self) -> Path:
        return self._env_or_first_existing(
            "SKILLSYNC_TEAMS_UI",
            [
                self.project_root / "data" / "artifacts" / "teams_ui.json",
                self.project_root / "data" / "teams_ui.json",
            ],
        )

    @property
    def teams_summary(self) -> Path:
       
        return self._env_or_first_existing(
            "SKILLSYNC_TEAMS_SUMMARY",
            [
                self.project_root / "data" / "artifacts" / "teams_summary.json",
                self.project_root / "data" / "teams_summary.json",
            ],
        )

    @property
    def project_requirements(self) -> Path:
        return self._env_or_first_existing(
            "SKILLSYNC_PROJECT_REQUIREMENTS",
            [
                self.project_root / "data" / "artifacts" / "project" / "project_requirements.json",
                self.project_root / "data" / "project_requirements.json",
            ],
        )

    @property
    def candidate_profiles(self) -> Path:
        return self._env_or_first_existing(
            "SKILLSYNC_CANDIDATE_PROFILES",
            [
                self.project_root / "data" / "artifacts" / "extraction" / "candidate_profiles_with_evidence.json",
                self.project_root / "data" / "artifacts" / "extraction" / "candidate_profiles.json",
                self.project_root / "data" / "candidate_profiles_with_evidence.json",
                self.project_root / "data" / "candidate_profiles.json",
            ],
        )

    @property
    def out_json(self) -> Path:
        return self.project_root / "data" / "artifacts" / "RAG" / "team_report.json"
