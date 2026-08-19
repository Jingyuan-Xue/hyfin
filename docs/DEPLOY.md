# Deployment

Complete setup for a machine that has never run HyFin. For what the system
does and how the parts fit together, see [ARCHITECTURE.md](ARCHITECTURE.md).

Budget about 40 minutes, most of it downloading the data bundle.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Bash | 4+ | `start.sh` uses `set -Eeuo pipefail` |
| `curl` | any | readiness polling |
| `setsid` | any | from `util-linux`; detaches services |
| Python (service) | 3.14 | runs the QA service, gateway and risk API |
| Python (retrieval) | 3.10 | runs the A2RAG worker — **must be separate** |
| Disk | ~20 GB | 13 GB data + ~6 GB virtualenvs |
| RAM | 8 GB+ | BGE-M3 on CPU |
| GPU | not required | the demo pins CPU; see [CPU vs GPU](#cpu-vs-gpu) |

The two Python versions are not interchangeable. The retrieval worker is a
subprocess launched from its own interpreter precisely so the heavyweight
embedding stack stays isolated from the service process.

---

## 2. Get the data bundle

Roughly 13 GB of models, indexes and corpora are **not in Git** — they are too
large and mostly not ours to redistribute. Everything below is excluded by
`.gitignore` and must be placed by hand.

| Destination | Size | Source |
|---|---:|---|
| `models/` | 4.3 GB | BGE-M3 weights, from HuggingFace |
| `FinGLMQA/data/indexes/` | 2.6 GB | TabGR + A2RAG retrieval indexes |
| `FinGLMQA/data/corpus_package/` | 135 MB | frozen evidence chunks |
| `output/` | 620 MB | MinerU markdown + embedding parquet |

### BGE-M3 weights

```bash
pip install "huggingface_hub[cli]"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download BAAI/bge-m3 --local-dir models/models--BAAI--bge-m3
```

The layout must stay in HuggingFace cache form — `query_type3_evidence.py`
probes for the literal directory name `models--BAAI--bge-m3` when deciding
whether it can run offline.

### Indexes and corpora

These are release artifacts of this project. Download the archive published
alongside the paper and unpack it at the repository root:

```bash
tar -xzf hyfin-data.tar.gz -C .
```

Verify placement before continuing:

```bash
test -f FinGLMQA/data/indexes/type3/annual_reports_170_v1/tabgr/manifest.json && echo "TabGR OK"
test -f FinGLMQA/data/indexes/a2rag_index/index_manifest.json && echo "A2RAG OK"
test -d models/models--BAAI--bge-m3 && echo "BGE-M3 OK"
```

### A note on immutability

`FinGLMQA/data/facts/financial_facts.duckdb` and its siblings are SHA-256 pinned
in `FinGLMQA/runs/phase_10/immutable_inputs_manifest.json`. The service verifies
that manifest at startup and refuses to become ready if a hash does not match.

Do not compact, re-encode or "tidy" these files. A lossless DuckDB compaction
still changes the hash and will take the service down until the manifest and
every report citing it are regenerated.

---

## 3. Create the virtualenvs

Two environments, matching the two runtimes above.

### Retrieval worker (Python 3.10)

Install the CPU build of PyTorch **first**, then the rest. Order matters: pip
resolves `torch` from whichever index it sees first, and the default PyPI wheel
is the CUDA one.

```bash
python3.10 -m venv A2RAG/.venv
A2RAG/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
A2RAG/.venv/bin/pip install -r A2RAG/requirements.txt
```

<h4 id="cpu-vs-gpu">CPU vs GPU</h4>

`scripts/env.sh` pins `A2RAG_EMBEDDING_DEVICE=cpu`, so the demo never touches
CUDA — it overrides whatever `.env` says. Installing the default PyPI `torch`
wheel therefore costs about 4.5 GB of CUDA libraries (`nvidia/*`, `triton`,
`libtorch_cuda.so`) that are downloaded, stored, and never loaded.

Skip the CPU pre-install **only** if you intend to rebuild the embedding
indexes, which does use the GPU:

```bash
A2RAG/.venv/bin/pip install -r A2RAG/requirements.txt   # pulls the CUDA build
```

### Service runtime (Python 3.14)

```bash
python3.14 -m venv FinGLMQA/.venv-phase10
FinGLMQA/.venv-phase10/bin/pip install -r FinGLMQA/requirements/phase10.lock
```

---

## 4. Configure

```bash
cp .env.example .env
```

`.env` is gitignored at any depth, which also covers `A2RAG/.env`. Never commit
it and never move these values into frontend JavaScript — the gateway reads them
server-side only.

### Minimum for a working demo

| Variable | Purpose |
|---|---|
| `A2RAG_API_KEY` | online answer generation |
| `A2RAG_CHAT_BASE_URL` | OpenAI-compatible endpoint |
| `A2RAG_CHAT_MODEL` | generation model id |

With these unset the service still starts and answers deterministically from the
frozen fact store, but narrative questions will not produce a generated answer.
`scripts/env.sh` derives `FINGLMQA_LLM_ENABLED` from their presence.

### Exposing the demo to anyone else

The keys above are metered. Before putting this on a shared URL:

```bash
python3 -c "import secrets;print(secrets.token_urlsafe(24))"   # -> DEMO_ACCESS_TOKEN
```

Set `DEMO_ACCESS_TOKEN` in `.env`. Every `/api/` route then requires it.
`DEMO_RATE_LIMIT_EXPENSIVE` (per hour) covers generation and translation;
`DEMO_RATE_LIMIT_GENERAL` (per minute) covers everything else. Set
`DEMO_TRUST_PROXY=1` **only** behind a proxy that sets `X-Forwarded-For`,
otherwise every client shares one rate-limit bucket.

### English display (optional)

Retrieval and reasoning stay in Chinese; only the display projection is
translated, server-side. Fill the `TENCENT*` block in `.env`, then:

```bash
./scripts/configure_translation.sh
./scripts/test_translation_provider.py
./scripts/pretranslate_static.py
./stop.sh && ./start.sh
DEMO_REQUIRE_TRANSLATION=1 ./selfcheck.sh
```

Dynamic content is translated on demand and cached by content hash in
`runtime/translation_cache.sqlite3`. Canonical Chinese questions, document ids,
numeric table cells and original evidence are never rewritten.

---

## 5. Start and verify

```bash
./start.sh
```

This starts the QA service (8010), the risk service (8012) and the gateway
(4173), then runs the self-check. Open <http://localhost:4173/>.

```bash
./selfcheck.sh              # read-only, no model calls
./selfcheck.sh --full-qa    # adds one real generation call (costs credits)
./stop.sh
```

A healthy default run prints 13 `[PASS]` lines. Two of them —
`TabGR index loaded` and `Hybrid evidence provider active` — assert values
measured from the running worker, so they fail closed if the TabGR index is
missing rather than reporting a hardcoded flag.

### Ports

Override without editing files:

```bash
DEMO_WEB_PORT=4273 DEMO_QA_PORT=8110 DEMO_RISK_PORT=8112 ./start.sh
```

`start.sh` refuses to start if a port is held by a process it does not own, so a
stale service can never be silently shadowed.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Missing required file: .../python` | virtualenv not created | §3 |
| `expected 170 TabGR documents, worker reported 0` | index missing or unreadable | §2, check `FinGLMQA/data/indexes/` |
| Service never becomes ready | manifest hash mismatch | a pinned file was modified; restore it |
| `Port NNNN is occupied` | previous run still up | `./stop.sh`, or override the port |
| `Online LLM generation` fails | no API key | §4 |
| `Tencent translation` warns | translation not configured | expected; English falls back to Chinese |
| Worker startup timeout | BGE-M3 loading on a cold cache | first start is slow; raise `FINGLMQA_WORKER_STARTUP_TIMEOUT_SECONDS` |

Logs are written under `runtime/logs/`. `start.sh` tails them automatically when
startup fails.

---

## 7. Relocating the package

Every path is computed from the repository root, so the directory can be moved
or renamed freely. Two caveats:

- `FinGLMQA/refs/*` are relative symlinks into sibling directories. They survive
  a move of the whole tree, not a move of individual subdirectories.
- Some scripts under `FinGLMQA/scripts/` — the offline experiment tooling, not
  the serving path — still contain absolute `/home/coder` paths.
