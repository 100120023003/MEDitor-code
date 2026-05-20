# MEDitor Minimal Anonymous Release

This is a minimal anonymous code release for **MEDitor: A Plug-and-Play Medical Multi-Agent Routing Protocol**.

The release keeps only the main two-expert workflow:

1. Query Expert A and Expert B.
2. Return the shared answer immediately when they agree.
3. Escalate disagreement cases to an evidence-grounded coordinator.
4. Optionally run debate/critics.
5. Resolve with a No-New-Option constrained judge or deterministic fallback.

Slurm scripts, shell service launchers, stage-2 rollout adapters, four-model extensions, old router ablations, generated outputs, logs, PDFs, checkpoints, and local paths are not included.

## Package Layout

- `autogen_agent_acl/core_ab_medrag_1209.py`: main MEDitor runner.
- `autogen_agent_acl/coordinator_medrag.py`: evidence coordinator using local MedRAG or a remote RAG endpoint.
- `autogen_agent_acl/agents.py`: OpenAI-compatible HTTP client helpers.
- `autogen_agent_acl/voting.py`: self-consistency voting.
- `autogen_agent_acl/debate.py`: optional disagreement debate.
- `autogen_agent_acl/judge.py`: constrained pairwise judge.
- `autogen_agent_acl/medqa.py`, `bench_io.py`: dataset parsing.
- `autogen_agent_acl/router.py`: default agreement/disagreement router.
- `autogen_agent_acl/medrag_src/`: minimal MedRAG-compatible retrieval backend.

## Run

From the repository parent, expose this folder on `PYTHONPATH` and provide OpenAI-compatible endpoints:

```bash
export PYTHONPATH=/path/to/meditor_minimal_anonymous:${PYTHONPATH}

python -m autogen_agent_acl.core_ab_medrag_1209 \
  --data /path/to/inputs.jsonl \
  --labels /path/to/labels.jsonl \
  --run-dir /path/to/output_run \
  --a-base http://127.0.0.1:8602/v1 --a-model expert_a \
  --b-base http://127.0.0.1:8603/v1 --b-model expert_b \
  --j-base http://127.0.0.1:8605/v1 --j-model judge \
  --use-medrag-judge \
  --medrag-mode rag_textbooks \
  --medrag-llm-name /path/to/local_or_served_rag_llm \
  --medrag-corpus-dir /path/to/medrag_corpus
```

To use a complete remote RAG service instead of local MedRAG retrieval, set:

```bash
--remote-rag-base http://127.0.0.1:8604/v1 --remote-rag-model rag
```

## Outputs

The runner writes predictions, summaries, and optional per-case traces under `--run-dir`.
