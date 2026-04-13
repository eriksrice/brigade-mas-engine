# Brigade MAS: Multi-Agent Orchestration Engine

An autonomous, locally-hosted multi-agent system (MAS) utilizing dynamic compute routing, Pydantic data validation, and semantic triage. Built as a state machine to enforce architectural first principles, standardize enterprise data operations, and autonomously red-team pipelines.

## Architecture Highlights
* **Semantic Router (The Maitre D'):** A zero-temperature LLM gatekeeper that classifies incoming unstructured text and routes it to specialized execution nodes.
* **Dynamic Compute Allocation:** Decouples intent from execution. Allocates compute tiers (Flash vs. Pro) dynamically based on payload complexity to optimize API latency and economics.
* **Stateful Tooling (The Expediter):** Integrates directly with the Gmail API via OAuth 2.0 to parse unstructured communications, utilizing a local JSON ledger to maintain idempotency and prevent duplicate processing.
* **Deterministic Guardrails:** Enforces strict enterprise semantic standards and conducts rigorous HIPAA/FERPA cybersecurity audits on drafted architectures before output.

## Tech Stack
Python 3.10+, Google Gemini API (2.5-Flash / 2.5-Pro), Google Cloud Auth (OAuth 2.0), Pydantic, TypedDict.