# Branch Development Log

This document tracks major milestones on the active development branch for **Peter Pan, Where did the Shadow Go?**—the 2D top-down MVP for shadow contour recognition and agent reasoning.

---

## Changelog

| Date | Development |
|------|-------------|
| **2026-04-01** | Initial **shadow recognition** pipeline (OpenCV, Lab luminance vs background, ROI, morphology, contour → polygon / bbox / centroid in table coordinates, JSON serialization, debug visualization). **Reasoning agent** foundation: rule-based planner (A*, edge safety, climbable books, shadow-biased paths), `environment.json` builder, closed-loop demo (`run_shadow_demo`), offline reasoning script, and example JSON artifacts. |
| **2026-04-03** | **Reasoning agent optimization**: optional **Claude (Anthropic)** backend with configurable **role prompt** (`config/role_prompt_shadow.txt`), rich environment fields (`identity`, `scene_relationships`, `semantic_context`), structured JSON output including `meta.generation_hints` for downstream generation, API key via `ANTHROPIC_API_KEY` / `.env`, and **fallback to the rule engine** when the key is missing or the API / parse fails. Shared logging (`reasoning_common`), factory (`create_reasoning_agent`), and documentation updates. |

---

## 2026-04-01 — Shadow recognition + reasoning agent (baseline)

- **Perception**: `ShadowDetector` with `table_roi` / `table_boundary`, background capture, noise filtering, `shadow_regions` output aligned with the sprint PRD.
- **Environment**: `build_environment` merging shadow output with table, agent, goal, and manual objects from `config.yaml`.
- **Reasoning**: `RuleBasedReasoningAgent` — deterministic, geometry-first decisions; outputs match the `reasoning_output.json` contract.
- **Tooling**: `scripts/run_shadow_demo.py`, `scripts/run_reasoning_offline.py`, `examples/*.json`, `config.yaml` sections for `shadow_detector` and `environment_builder`.

---

## 2026-04-03 — Reasoning agent optimization

- **Claude integration**: `ClaudeReasoningAgent` using the Messages API; system prompt = role file + strict JSON schema; user payload = full environment state.
- **Narrative + spatial**: Role-play as the source object behind the shadow; relationships and persona inform `reason` and optional `generation_hints`.
- **Operations**: `reasoning_agent.backend` (`rules` \| `claude`), `fallback_to_rules`, `.env.example`, optional `python-dotenv` loading.

---

## Related files

| Area | Location |
|------|----------|
| Shadow detection | `peter_pan/perception/shadow_detector.py` |
| Environment merge | `peter_pan/perception/environment_builder.py` |
| Rule-based reasoning | `peter_pan/agent_brain/reasoning_agent.py` |
| Claude reasoning | `peter_pan/agent_brain/claude_reasoning_agent.py` |
| Backend selection | `peter_pan/agent_brain/reasoning_factory.py` |
| Role prompt | `config/role_prompt_shadow.txt` |
| Live demo | `scripts/run_shadow_demo.py` |
| Project overview | Root `README.md` |

---

*Last updated: 2026-04-03.*
