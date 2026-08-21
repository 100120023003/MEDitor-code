# MEDitor: A Plug-and-Play Medical Multi-Agent Routing Protocol

[![CI](https://github.com/100120023003/MEDitor-code/actions/workflows/ci.yml/badge.svg)](https://github.com/100120023003/MEDitor-code/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

This repository contains the camera-ready research code for **MEDitor**, a
training-free routing protocol for multiple-choice medical question answering.
MEDitor queries two interchangeable medical experts, returns their shared answer
on agreement, and sends only disagreement cases through an evidence-grounded
deep path.

> [!IMPORTANT]
> MEDitor is research software, not a medical device. Its outputs must not be
> used for diagnosis, treatment, or other clinical decisions.

## Method

```text
                         experts agree
Question -> Expert A + Expert B ------------> answer
                    |
                    | experts disagree
                    v
          evidence-grounded coordinator
                    |
             optional debate
                    |
          candidate-constrained judge
                    |
          deterministic fallback
```

The judge can select only one of the expert-induced candidates or abstain. An
abstention is resolved by deterministic evidence and expert-decisiveness rules,
so the deep path cannot introduce a new answer option.

The paper configuration uses:

| Role | Model |
| --- | --- |
| Expert A | UltraMedical-7B |
| Expert B | Huatuo-8B |
| Coordinator | Llama-3.1-8B-Instruct |
| Judge | Prometheus-7B |

The implementation accepts any OpenAI-compatible chat-completions endpoint. The
paper also evaluates a Qwen3-8B and Meerkat-8B expert pair.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `meditor/runner.py` | Main agreement-aware routing and evaluation entry point |
| `meditor/coordinator_medrag.py` | Evidence-grounded coordinator |
| `meditor/agents.py` | OpenAI-compatible endpoint client |
| `meditor/voting.py` | Self-consistency voting |
| `meditor/debate.py` | Optional disagreement debate |
| `meditor/judge.py` | Candidate-constrained pairwise judge |
| `meditor/router.py` | Shallow/deep routing policy |
| `meditor/medqa.py` and `meditor/bench_io.py` | Input parsers |
| `meditor/medrag_src/` | Adapted MedRAG retrieval backend |
| `meditor/custom_rag/` | Custom corpus import and hybrid indexing utilities |
| `examples/` | Public example input and cached predictions |
| `tests/` | Dependency-light release tests |

Model weights, benchmark data, retrieval corpora, indexes, checkpoints, and
generated outputs are not redistributed. Obtain each asset from its official
source and follow its license.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/100120023003/MEDitor-code.git
cd MEDitor-code
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The base install is sufficient for remote OpenAI-compatible endpoints and the
cached-prediction smoke test. For local MedRAG retrieval:

```bash
python -m pip install -e ".[retrieval]"
```

Install the CUDA-specific PyTorch build required by your hardware before the
retrieval extra when applicable. Pyserini-based sparse retrieval also requires a
working Java installation.

### Custom Textbook Indexes

MEDitor also includes a self-contained BM25/dense indexing path for owned or
properly licensed textbook corpora. Convert source material to the
[documented corpus format](meditor/custom_rag/corpus_format.md), then build
indexes:

```bash
meditor-rag build-bm25 \
  --chunks /path/to/corpus/chunks.jsonl \
  --index_dir /path/to/corpus/indexes/bm25

meditor-rag build-dense \
  --chunks /path/to/corpus/chunks.jsonl \
  --index_dir /path/to/corpus/indexes/medcpt \
  --model_name ncbi/MedCPT-Article-Encoder \
  --query_model_name ncbi/MedCPT-Query-Encoder \
  --device cuda:0
```

Select those indexes in the main runner with
`--custom-rag-textbooks-corpus-dir /path/to/corpus`. Use only corpora whose
licenses permit this processing.

## Quick Verification

This command makes no model or network calls. Both expert predictions are read
from the example cache and take the shallow path:

```bash
python -m meditor.runner \
  --data examples/example_input.jsonl \
  --run-dir outputs/smoke \
  --a-base http://127.0.0.1:8602/v1 --a-model unused-expert-a \
  --b-base http://127.0.0.1:8603/v1 --b-model unused-expert-b \
  --a-baseline-preds examples/expert_a_predictions.jsonl \
  --b-baseline-preds examples/expert_b_predictions.jsonl \
  --disable-debate --no-llm-judge
```

On PowerShell, replace each trailing `\` with a backtick, or put the command on
one line.

## Run MEDitor

Start OpenAI-compatible servers for the two experts and judge, then run:

```bash
python -m meditor.runner \
  --data /path/to/test.jsonl \
  --run-dir outputs/experiment_name \
  --a-base http://127.0.0.1:8602/v1 --a-model expert-a \
  --b-base http://127.0.0.1:8603/v1 --b-model expert-b \
  --j-base http://127.0.0.1:8605/v1 --j-model judge \
  --use-medrag-judge \
  --remote-rag-base http://127.0.0.1:8604/v1 \
  --remote-rag-model rag-coordinator
```

`--use-medrag-judge` enables the evidence coordinator. The example above uses a
complete remote RAG service. For the bundled local retrieval backend, replace
the remote RAG arguments with:

```bash
--medrag-mode rag_textbooks \
--medrag-llm-name /path/to/coordinator_model \
--medrag-corpus-dir /path/to/medrag_corpus
```

Run `python -m meditor.runner --help` for all routing, retrieval, debate, judge,
and ablation options.

### Endpoint Authentication

Local endpoints need no key. Authenticated endpoints can use command-line
arguments or these environment variables:

| Role | Argument | Environment variable |
| --- | --- | --- |
| Expert A | `--a-api-key` | `MEDITOR_A_API_KEY` |
| Expert B | `--b-api-key` | `MEDITOR_B_API_KEY` |
| Judge | `--j-api-key` | `MEDITOR_JUDGE_API_KEY` |
| Optional solver | `--solver-api-key` | `MEDITOR_SOLVER_API_KEY` |
| RAG summarizer | `--rag-summary-api-key` | `RAG_API_KEY` |
| Remote RAG | `--remote-rag-api-key` | `REMOTE_RAG_API_KEY` |

Prefer environment variables so credentials do not appear in shell history.
Never commit `.env` files.

## Data Format

The default loader expects one JSON object per line:

```json
{"id":"case-001","question":"Question text","options":{"A":"first","B":"second","C":"third","D":"fourth"},"gold":"B"}
```

`options` may also be a list. Gold labels may be supplied as `gold_letter`,
`gold`, `label`, `answer`, `answer_idx`, or `label_idx`. To use the separate
input/label format supported by the unified benchmarks, pass
`--labels /path/to/labels.jsonl`.

Cached expert files use one record per line:

```json
{"id":"case-001","pred":"B"}
```

The cache loader also accepts `uid`/`idx` keys and common prediction field
names. Cached predictions are useful for exact routing ablations and for
avoiding repeated expert inference.

## Outputs

Each run writes:

- `preds.jsonl`: predictions, routing decisions, and per-case metadata.
- `judge_traces.jsonl`: judge inputs and outputs when judging is used.
- `run_summary.md`: aggregate accuracy and routing statistics.
- `md/`: optional readable case traces when `--md-all` is enabled.
- `medrag/`: retrieval logs when the evidence coordinator is enabled.

Do not publish raw benchmark questions, licensed corpora, or private model
traces unless their licenses and data-governance rules permit it.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks input parsing, routing decisions, endpoint configuration,
BM25 index round-tripping, and an end-to-end cached-prediction run without
heavyweight dependencies.

## Citation

Please cite the EMNLP 2026 paper **"MEDitor: A Plug-and-Play Medical
Multi-Agent Routing Protocol."** The archival ACL Anthology BibTeX entry will be
added here once the proceedings record is available.

## Acknowledgements

The retrieval backend in `meditor/medrag_src/` is adapted from
[MedRAG](https://github.com/gzxiong/MedRAG). Its United States Government Work
notice and requested attribution are retained in
[`meditor/medrag_src/LICENSE`](meditor/medrag_src/LICENSE). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

## License

MEDitor-specific code is released under the [MIT License](LICENSE). The adapted
MedRAG files retain the upstream public-domain notice described above.
