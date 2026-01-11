VibeCode Security Benchmark — Proof of Concept

This repository contains the proof of concept (PoC) artefact developed as part of a final-year research project investigating the evaluation of security and quality risks in LLM-generated (“vibe-coded”) software.

The PoC demonstrates the feasibility of a lightweight benchmarking framework that integrates existing static analysis tooling and presents results in a standardised, Lighthouse-inspired report format.

Project Overview

Large Language Models (LLMs) are increasingly used to generate source code from natural language prompts. While this accelerates development, it also introduces concerns around software security, reliability, and maintainability. Currently, there is no widely adopted, standardised benchmarking framework for evaluating LLM-generated code.

This proof of concept validates a core component of the proposed framework by:

analysing Python code samples using static analysis,

exporting structured results, and

visualising findings in a consistent, human-readable report.

Repository Structure

PoC/
├── sample/
│ ├── workflow_engine.py
│ ├── workflow_engine_vuln.py
│ └── semgrep_bait.py
│
├── results/
│ ├── semgrep_workflow_engine.json
│ ├── semgrep_workflow_engine_vuln.json
│ ├── semgrep_semgrep_bait.json
│ ├── semgrep_workflow_engine.txt
│ ├── semgrep_workflow_engine_vuln.txt
│ └── semgrep_semgrep_bait.txt
│
├── ui/
│ ├── index.html
│ ├── styles.css
│ └── app.js
│
└── README.md

Tooling

Language: Python
Static Analysis Tool: Semgrep
Front-End: Vanilla HTML, CSS, and JavaScript

Semgrep was selected for the PoC due to its mature rule set, strong Python support, and machine-readable JSON output, making it suitable for automated analysis pipelines.

Running the Proof of Concept

Prerequisites

Python 3.11 or later

Semgrep installed

Run Static Analysis

From the project root:

semgrep --config=auto --scan-unknown-extensions --no-git-ignore sample/workflow_engine.py --json > results/semgrep_workflow_engine.json

semgrep --config=auto --scan-unknown-extensions --no-git-ignore sample/workflow_engine_vuln.py --json > results/semgrep_workflow_engine_vuln.json

semgrep --config=auto --scan-unknown-extensions --no-git-ignore sample/semgrep_bait.py --json > results/semgrep_semgrep_bait.json

Optional human-readable output:

semgrep --config=auto --scan-unknown-extensions --no-git-ignore sample/semgrep_bait.py > results/semgrep_semgrep_bait.txt

Viewing the Report

Open ui/index.html in a web browser.

Click “Load Semgrep JSON”.

Select any file from the results directory.

The interface displays:

Lighthouse-style score indicators,

severity breakdowns,

individual findings with rule IDs and locations.

Scoring is heuristic and intended solely to demonstrate the reporting structure for this proof of concept.

Scope and Limitations

This proof of concept:

focuses exclusively on Python,

uses a single static analysis tool (Semgrep),

applies heuristic scoring,

analyses a limited number of representative code samples.

The PoC is designed to validate feasibility rather than provide a complete benchmark.

Future Work

Planned extensions for the full project include:

integration of additional tools such as SonarQube and OWASP ZAP,

multi-language benchmarking,

dynamic analysis,

CI/CD pipeline integration,

large-scale empirical evaluation of LLM-generated code.

Disclaimer

This repository represents an academic proof of concept developed for research purposes and is not intended for production use.
