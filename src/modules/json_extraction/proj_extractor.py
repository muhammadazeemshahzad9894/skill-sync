from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from openai import OpenAI


SYSTEM_PROMPT = """You are an information extraction assistant.

Hard constraints:
- Extract ONLY what is explicitly supported by the input text.
- Return VALID JSON ONLY (no markdown, no extra text).
- Do NOT invent, assume, infer, or expand beyond the text.
- If a term is clearly a technology and is explicitly mentioned, extract it.
- If uncertain or ambiguous, omit the item.

Very important:
- The USER'S project description is the ONLY source of truth for what to extract.
- Any lists in the prompt are for NAME NORMALIZATION ONLY for technologies, not for restricting extraction,
  EXCEPT the allowed roles list, which IS a strict output whitelist for roles.
"""


USER_PROMPT = """Extract technical requirements from the project description.

Return EXACTLY this JSON schema (no extra keys):

{
  "technical_requirements": {
    "skills": [],
    "tools": [],
    "roles": []
  }
}

Rules:
- skills: programming languages, frameworks, libraries, databases, cloud platforms, protocols.
- tools: developer tools ONLY (IDEs, notebooks, CLIs, build/test/deploy tools).
- roles: required project roles (STRICT: only from the allowed role labels list below).
- Extract ONLY what is supported by the text (no guessing).
- Return lists with unique items (no duplicates).
- Return JSON only.
- If uncertain or ambiguous, omit the item.

CRITICAL INTERPRETATION RULE:
- Extract technologies from the PROJECT DESCRIPTION, even if they are NOT listed anywhere in this prompt.
- Lists in this prompt exist ONLY to normalize spelling/casing/spacing/punctuation for consistency (technologies).
- Do NOT omit a clearly mentioned technology just because it is not listed below.

CATEGORY ENFORCEMENT RULES:
- Cloud platforms (e.g., AWS, GCP, Azure, OCI) MUST ALWAYS be classified as "skills", NEVER as "tools".
- Libraries and frameworks (e.g., FAISS, NumPy, SciPy, PyTorch, Transformers) MUST be classified as "skills".
- Communication protocols explicitly mentioned (e.g., MQTT) MUST be classified as "skills".
- "tools" must ONLY include IDEs, notebooks, build systems, test frameworks, and CLIs.

Category guidance using Stack Overflow 2024 sections (for classification only):
- "skills" may include:
  1) Programming, scripting, and markup languages
  2) Web frameworks and web technologies
  3) Database environments
  4) Cloud platforms
  5) Libraries and frameworks
  6) Communication protocols

- "tools" may include:
  1) IDEs and editors
  2) Notebooks
  3) Build, test, deploy, and CLI tools

Mapping / normalization rule:
- If the text mentions a technology using a variant/typo/alias, you MAY normalize it ONLY if the target is unambiguous.
- If normalization is ambiguous, keep the original wording from the project description (or omit if unclear).
- Do NOT normalize roles.

ROLES (STRICT — LITERAL VALUES ONLY):
- You MAY ONLY output roles from the allowed list below.
- Output role labels EXACTLY as written (character-for-character).
- Do NOT reword, normalize, shorten, or expand role names.
- Do NOT invent new roles.
- Do NOT output partial matches.
- Do NOT output "Other (please specify):".
- If no role from the list is clearly supported by the project description, return an empty roles list.

ROLE PRIORITY RULE:
- If "Developer, full-stack" is added, do NOT add "Developer, front-end" or "Developer, back-end".

ROLE MAPPING RULES (ONLY TO SELECT FROM THE ALLOWED ROLE LABELS BELOW):
Apply these ONLY when the project description explicitly mentions the related technology or responsibility.

- If the text mentions a frontend framework or frontend/UI work (e.g., React, Vue, Angular, "frontend", "UI"):
  add: "Developer, front-end"

- If the text mentions backend services/server/backend work (e.g., "backend", "server", "API services",
  Python services, Node.js backend, Java backend):
  add: "Developer, back-end"

- If the text explicitly mentions "full-stack" or "full stack":
  add: "Developer, full-stack"

- If the text mentions mobile app development (e.g., React Native, iOS, Android, "mobile"):
  add: "Developer, mobile"

- If the text mentions embedded systems, IoT, firmware, or device-level programming:
  add: "Developer, embedded applications or devices"

- If the text mentions Machine Learning, NLP, or AI model training:
  add: "Data scientist or machine learning specialist"

- If the text mentions SQL/databases/data pipelines/streaming systems:
  add: "Data engineer"

- If the text mentions ANY cloud platform name (AWS, Amazon Web Services, GCP, Google Cloud, Azure, OCI):
  add: "Cloud infrastructure engineer"

- If the text explicitly mentions Docker, Kubernetes, containerization, CI/CD, or orchestration:
  add: "DevOps specialist"

- If the text explicitly mentions SRE or "site reliability":
  add: "Engineer, site reliability"

TARGETED CANONICALIZATION PATCH (NORMALIZATION ONLY — NOT A WHITELIST):

Languages / markup / shell:
- Bash/Shell (all shells)
- HTML/CSS
- Objective-C
- Visual Basic (.Net)

Database environments:
- Cloud Firestore
- Cosmos DB
- Couch DB
- Databricks SQL
- Firebase Realtime Database
- IBM DB2
- Microsoft SQL Server
- Microsoft Access

Cloud platforms:
- Alibaba Cloud
- Amazon Web Services (AWS)
- Digital Ocean
- Google Cloud
- IBM Cloud Or Watson
- Oracle Cloud Infrastructure (OCI)

Web frameworks and web technologies:
- ASP.NET CORE
- Play Framework
- Ruby on Rails

Compilers / build / test / dev tooling:
- GNU GCC
- LLVM's Clang
- Google Test
- Maven (build tool)

Other frameworks and libraries:
- .NET (5+)
- .NET Framework (1.0 - 4.8)
- Apache Kafka
- Apache Spark
- Hugging Face Transformers
- React Native
- Scikit-Learn
- Spring Framework
- Torch/PyTorch

IDEs / editors / developer environments:
- Visual Studio Solution
- Android Studio
- IntelliJ IDEA
- Jupyter Notebook/JupyterLab
- Qt Creator
- Rad Studio (Delphi, C++ Builder)

Collaborative tools:
- GitHub Discussions

Normalization rules:
1) Cloud platform normalization
- If the text mentions "AWS", output: "Amazon Web Services (AWS)"
- If the text mentions "GCP" or "Google Cloud Platform", output: "Google Cloud"
- If the text mentions "OCI" or "Oracle Cloud", output: "Oracle Cloud Infrastructure (OCI)"
- If the text mentions "IBM Cloud" or "Watson", output: "IBM Cloud Or Watson"
- If the text mentions "DigitalOcean", output: "Digital Ocean"

2) .NET normalization
- If the text mentions ".NET Core" or "ASP.NET Core", output: "ASP.NET CORE"
- If the text mentions ".NET 5+", output: ".NET (5+)"

3) Shell normalization
- If the text mentions "Bash" or "Shell scripting", output: "Bash/Shell (all shells)"

4) PyTorch normalization
- If the text mentions "PyTorch" or "Torch", output: "Torch/PyTorch"

ALLOWED ROLE LABELS (AUTHORITATIVE):
- Developer, full-stack
- Developer, back-end
- Developer, front-end
- Developer, mobile
- Developer, embedded applications or devices
- DevOps specialist
- Engineer, site reliability
- Cloud infrastructure engineer
- Data engineer
- Data scientist or machine learning specialist
- Developer, AI
- Developer, QA or test
- Developer, desktop or enterprise applications
- Product manager
- Project manager
- Security professional
- Data or business analyst
- Research & Development role
- Academic researcher
- Educator
- Student
- Developer Advocate
- Engineering manager
- Senior Executive (C-Suite, VP, etc.)

Exclusions:
- Do NOT include any Git/GitHub/version control items in tools or skills.
- Do NOT output "Other (please specify):" in roles.

If none found: return empty lists.
"""


@dataclass
class ProjectExtractorConfig:
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0


class ProjectSchemaExtractor:
    def __init__(self, api_key: str, base_url: str, config: Optional[ProjectExtractorConfig] = None):
        self.config = config or ProjectExtractorConfig()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def safe_json_loads(raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        return json.loads(raw[start : end + 1])

    def extract_requirements(self, description: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT + "\n\nProject description:\n" + (description or "")},
            ],
        )
        raw = resp.choices[0].message.content
        return self.safe_json_loads(raw)
