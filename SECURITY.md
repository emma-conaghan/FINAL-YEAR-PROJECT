# Security Policy

## Purpose of this repository

This repository is part of a research project evaluating the security and maintainability of LLM-generated code.

Some files in this repository may intentionally contain insecure code patterns, weak practices, or vulnerable examples. These artefacts are included for testing, benchmarking, and analysis purposes only.

They must not be used in production systems.

## Supported Versions

This repository is a research prototype and does not provide production-supported releases.

## Scope

### In scope
Please report vulnerabilities that affect the research framework itself, including for example:

- pipeline orchestration code
- credential handling
- automation scripts
- CI/CD configuration
- API key management
- unsafe exposure of local services or tokens
- vulnerabilities that affect the integrity of analysis results

### Out of scope
The following are generally out of scope because they may be intentionally included as test cases:

- insecure patterns in generated example code
- deliberately vulnerable benchmark samples
- intentionally unsafe prompts or outputs used for evaluation
- archived model outputs stored for comparison purposes

## Reporting a Vulnerability

If you believe you have found a vulnerability in the framework or repository infrastructure, please report it by contacting the repository owner privately rather than opening a public issue.

Please include:

- a description of the issue
- affected file(s) or component(s)
- reproduction steps
- potential impact

## Notes

Because this repository is used to analyse insecure and LLM-generated code, the presence of vulnerable examples does not necessarily indicate a flaw in the repository itself.
