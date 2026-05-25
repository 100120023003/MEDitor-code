# MEDitor: A Plug-and-Play Medical Multi-Agent Routing Protocol

This repository provides the code for **MEDitor**, a training-free, plug-and-play medical multi-agent routing protocol for multiple-choice medical QA.

MEDitor uses two lightweight medical expert models as interchangeable experts. It exits early when the experts agree and routes only disagreement cases to an evidence-grounded Deep Path. The Deep Path consists of an Evidence-Grounded Coordinator, an optional evidence-grounded debate module, and a candidate-constrained judge. The final decision is restricted to the expert-induced candidate set or resolved by deterministic fallback.

This repository is anonymized for peer review.

---

## Overview

MEDitor is designed for deployment settings where large proprietary APIs or monolithic 70B-scale models may be impractical because of cost, privacy, or local inference constraints. Instead of applying expensive deliberation to every query, MEDitor uses disagreement as a routing signal:

1. **Expert proposal**: two lightweight medical LLMs independently answer the same medical QA instance.
2. **Shallow Path**: if both experts agree, MEDitor returns the consensus answer.
3. **Deep Path**: if the experts disagree, MEDitor retrieves candidate-aligned evidence and performs constrained arbitration.
4. **Candidate-constrained judging**: the judge selects only between the two expert-induced candidates or abstains.
5. **Deterministic fallback**: abstentions are resolved using expert decisiveness and evidence support.

The default instantiation used in the paper is:

| Role | Model |
|---|---|
| Expert A | UltraMedical-7B |
| Expert B | Huatuo-8B |
| Coordinator | Llama-3.1-8B-Instruct |
| Judge | Prometheus-7B |

The framework also supports expert replacement. In the paper, we additionally evaluate a Qwen3-8B + Meerkat-8B expert pair.

---

## Main Features

- Training-free multi-agent medical QA protocol.
- Plug-and-play expert interface.
- Disagreement-triggered Shallow/Deep routing.
- Evidence-Grounded Coordinator for candidate-aligned evidence packs.
- Optional evidence-grounded debate.
- Candidate-constrained judge with deterministic fallback.
- Cached expert artifacts for reproducible ablations.
- Support for MedQA, PubMedQA, MedMCQA, MMLU-med, and MedXpertQA-Text.
- Deployment-cost accounting with routed and no-routing policies.

---

## Repository Structure

A typical repository layout is:

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── default_meditor.yaml
│   ├── qwen_meerkat.yaml
│   └── retrieval.yaml
├── scripts/
│   ├── run_experts.py
│   ├── run_meditor.py
│   ├── run_ablation.py
│   ├── run_updated_experts.py
│   └── evaluate.py
├── meditor/
│   ├── experts/
│   ├── routing/
│   ├── egc/
│   ├── debate/
│   ├── judge/
│   ├── retrieval/
│   └── utils/
├── data/
│   └── README.md
├── outputs/
│   └── README.md
└── examples/
    ├── example_input.jsonl
    └── example_run.sh
