# HyFin: Hybrid Cross-Document Analysis of Financial Reports

HyFin is the reference implementation for evidence-grounded analysis of Chinese
company annual reports. It combines dense text retrieval with structured table
retrieval. For each answer, the system records the report, company, year, and
source chunk behind the supporting evidence.

The tracked files include the question-answering pipeline, frozen fact stores,
exposure-analysis artifacts, evaluation records, and an interactive demo. Model
weights and retrieval indexes are distributed separately.

## Method overview

HyFin uses two evidence channels:

- A2RAG retrieves narrative passages from annual reports.
- TabGR retrieves structured table rows and numeric evidence.
- The fusion layer keeps company, year, document, and chunk-level provenance.
- Narrative questions use a generation model. Supported metric questions are
  resolved deterministically from the frozen fact store.

Industry- and risk-exposure requests replay frozen artifacts. The
question-answering pipeline performs retrieval at inference time.

## Repository structure

```text
HyFin/
├── A2RAG/                  dense text retrieval library
├── FinGLMQA/               hybrid QA pipeline, schemas, facts, and evaluations
├── icdm_demo/              interactive demonstration interface
├── risk_exposure_method/   risk-exposure API and frozen artifacts
├── i18n/                   bilingual display resources
├── docs/                   architecture and deployment documentation
├── scripts/                environment, validation, and translation utilities
└── start.sh                service launcher
```

## Requirements

- Linux with Bash 4+, `curl`, and `setsid`
- Python 3.14 for the service layer
- Python 3.10 for the A2RAG retrieval worker
- At least 8 GB RAM
- Approximately 20 GB disk space for environments, models, indexes, and corpora

A GPU is not required for inference. The supplied runtime configuration uses
CPU inference for the embedding model.

## Installation

Clone the repository and create the two isolated Python environments:

```bash
git clone https://github.com/Jingyuan-Xue/hyfin.git
cd hyfin

python3.10 -m venv A2RAG/.venv
A2RAG/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
A2RAG/.venv/bin/pip install -r A2RAG/requirements.txt

python3.14 -m venv FinGLMQA/.venv-phase10
FinGLMQA/.venv-phase10/bin/pip install -r FinGLMQA/requirements/phase10.lock
```

See [docs/DEPLOY.md](docs/DEPLOY.md) for dependency notes and troubleshooting.

## Data preparation

The repository includes the frozen fact stores and compact evaluation artifacts.
The following large assets are not tracked by Git:

- BGE-M3 model weights
- TabGR and A2RAG retrieval indexes
- the packaged report corpus
- parsed annual-report text used by the retrieval worker

Place the released data bundle at the repository root so that the directory
layout matches the paths documented in
[docs/DEPLOY.md](docs/DEPLOY.md#2-get-the-data-bundle). The immutable input
manifest under `FinGLMQA/runs/phase_10/` records the expected hashes for the
frozen inputs.

## Configuration

Create a local environment file from the example:

```bash
cp .env.example .env
```

Set the model endpoint, model identifier, and API key required for online answer
generation. Metric queries backed by the frozen fact store can run without an
online generation model. Translation credentials are optional and affect only
the display layer.

Environment files, model weights, indexes, logs, and runtime caches are excluded
from version control.

## Running the system

After the environments, data bundle, and configuration are in place, start the
services with:

```bash
./start.sh
```

The command starts the retrieval, risk-analysis, and demonstration services,
runs the integrated readiness checks, and prints the access URL. Use the
following commands for validation and shutdown:

```bash
./selfcheck.sh
./selfcheck.sh --full-qa
./stop.sh
```

The default self-check is read-only and does not call the generation model. The
full QA check submits one grounded narrative query and may incur API usage.

## Reproducibility

Reproduction files are kept with the pipeline stage that produced them:

- `FinGLMQA/data/facts/` contains the frozen structured fact stores.
- `FinGLMQA/data/schemas/` defines the data and service contracts.
- `FinGLMQA/runs/` records input manifests, gate reports, and evaluation outputs.
- `FinGLMQA/scripts/` contains index-building, validation, and evaluation tools.
- `risk_exposure_method/output/` contains the frozen risk-exposure dataset and
  evaluation summaries.

Run the integrated check after deployment:

```bash
./selfcheck.sh
```

Individual validation programs under `FinGLMQA/scripts/` can be used to inspect
specific pipeline stages. Their corresponding manifests and expected outputs
are stored under `FinGLMQA/runs/`.

## Limitations

- Text and table candidates are combined by deterministic interleaving rather
  than a jointly calibrated cross-channel score. The table channel contributes
  at most two rows per report, so its share of the evidence set is capped rather
  than earned by relevance.
- Cross-document comparison occurs during answer consolidation; retrieval is
  performed independently for each selected report.
- Industry- and risk-exposure modules serve frozen artifacts rather than running
  their full construction pipelines online.
- The current service processes one QA request at a time.

## Citation

If you use this repository in your research, cite the project as:

```bibtex
@misc{hyfin2026,
  title        = {HyFin: Hybrid Cross-Document Analysis of Financial Reports
                  with Traceable Evidence},
  author       = {HyFin Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/Jingyuan-Xue/hyfin}}
}
```
