# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Typical end-to-end workflow

```bash
# 1. Bootstrap project (or manually create project.toml)
litreview init <seed-pdf-folder> --scope "Include if ..."

# 2. Validate criteria against seeds before committing to a full run
litreview screen --config projects/<slug>/project.toml --dry-run

# 3. Screen → review sample → tighten criteria → re-screen → traverse
litreview screen --config ...
litreview traverse --config ...

# 4. Iterate until stable (or run full automated pipeline)
litreview iterate --config ... --loop --recommend
# or: litreview run projects/<slug>/project.toml

# 5. Post-stability: extract structured fields, then synthesise
litreview extract --config ... --backend anthropic
litreview synthesize --config ...
```

The key invariant: **screen → review sample → refine criteria → re-screen → traverse**. Each traversal fans out from every included paper, so false positives compound — tighten criteria before traversing, not after.

## Setup

Requires Python ≥ 3.11. Key dependencies: `supabase` (DB), `pyalex` (OpenAlex), `semanticscholar` (S2 API), `pypdf` (PDF reference parsing), `rapidfuzz` (fuzzy dedup), `tenacity` (retry), `typer`+`rich` (CLI).

```bash
pip install -e .
# Create .env with the variables listed below
```

`--verbose` / `-v` is a **global** flag that must come before the subcommand: `litreview --verbose extract --config ...`

## Running the pipeline

```bash
# Bootstrap a new project from a folder of seed PDFs (extracts arXiv IDs from filenames)
litreview init <folder> --scope "Include if ..."

# Ingest seed papers for an existing project.toml
litreview ingest --config projects/geometry-testing-simplicial-complexes/project.toml

# Or via the standalone script (same logic, no install required)
python scripts/ingest_seeds.py --config projects/geometry-testing-simplicial-complexes/project.toml

# Ingest ad-hoc identifiers
litreview ingest "my-project" 1411.5713 2111.11316 "10.1145/3519935.3519989"
litreview ingest --config ... --no-openalex  # skip OpenAlex enrichment

# Traverse citations → adds candidates to papers table
litreview traverse --config projects/.../project.toml
litreview traverse --config ... --direction backward        # forward | backward | both
litreview traverse --config ... --from-depth 1              # traverse from depth-1 included papers

# Screen all pending papers
# Backends: gemini (default), groq, openrouter, anthropic
# Recommended: --backend groq (llama-3.3-70b-versatile default — best accuracy in tournament)
litreview screen --config projects/.../project.toml
litreview screen --config ... --backend groq                 # llama-3.3-70b-versatile default (best accuracy)
litreview screen --config ... --backend groq --model openai/gpt-oss-20b  # fastest/cheapest groq model
litreview screen --config ... --backend anthropic --model claude-haiku-4-5-20251001
litreview screen --config ... --backend openrouter           # qwen/qwen3-30b-a3b:free default
litreview screen --config ... --backend gemini,groq          # comma-separated = probe both, use fastest, auto-failover
litreview screen --config ... --dry-run                     # prints decisions, nothing written
litreview screen --config ... --limit 50                    # stop after N papers (useful for API health checks)
litreview screen --config ... --rate-sleep 0.5              # override rate limit sleep (e.g. on paid Gemini tier)
litreview screen --config ... --include-uncertain           # clears UNCERTAIN: prefix for a clean re-screen
litreview screen --config ... --non-interactive             # never prompt on failure — auto-switch or raise
litreview screen --config ... --yes                        # skip large-batch confirmation (>200 papers)

# QA check after screening: stratified sample re-checked with Claude Haiku (Anthropic)
# Stops with exit code 2 and prints disagreements if agreement < threshold
# If ANTHROPIC_API_KEY is not set, dumps sample to /tmp for in-chat QA instead
litreview screen --config ... --qa                          # default: 75-paper sample, 90% threshold
litreview screen --config ... --qa --qa-sample 50 --qa-threshold 0.85

# Fetch semantically similar papers via S2 Recommendations API (SPECTER2)
# Uses included papers as positives, excluded as negatives → upserts source='search' candidates
# Requires SEMANTIC_SCHOLAR_API_KEY
litreview recommend --config projects/.../project.toml
litreview recommend --config ... --limit 200 --dry-run

# One-shot: traverse + screen (one iteration, optionally loop until stable)
# Add --recommend to also run S2 recommendations as a parallel sourcing track
litreview iterate --config projects/.../project.toml
litreview iterate --config ... --loop --max-rounds 10 --yield-threshold 0.05
litreview iterate --config ... --recommend                  # enable S2 recommendations

# Full automated pipeline: screen → download PDFs → audit + recover → traverse → recommend → repeat
# --recommend is ON by default in `run`; disable with --no-recommend
litreview run projects/.../project.toml
litreview run projects/.../project.toml --backend anthropic --max-rounds 5
litreview run projects/.../project.toml --no-recommend      # citation traversal only

# Download open-access PDFs for included papers → projects/<slug>/included-pdfs/
litreview download-pdfs --config projects/.../project.toml
litreview download-pdfs --config ... --depth 1              # only depth-1 papers

# Audit citation coverage: compare each paper's reference list against the DB corpus
litreview audit --config projects/.../project.toml
litreview audit --config ... --included                     # audit included papers (needs PDFs)
litreview audit --config ... --recover                      # fetch missing refs from S2 and upsert
litreview audit --config ... --included --recover --dry-run

# Extract structured fields from all included papers (run post-stability)
litreview extract --config projects/.../project.toml
litreview extract --config ... --backend anthropic --model claude-sonnet-4-6
litreview extract --config ... --force     # re-extract already-extracted papers
litreview extract --config ... --dry-run

# Generate draft literature review from extractions (requires ANTHROPIC_API_KEY)
litreview synthesize --config projects/.../project.toml
litreview synthesize --config ... --model claude-opus-4-6
litreview synthesize --config ... --output /tmp/draft.md

# Reset screened papers back to pending (seeds never touched)
litreview reset-screening --config projects/.../project.toml
litreview reset-screening --config ... --depth 1 --yes

# Compare screening decisions between two models
python scripts/compare_screeners.py --config projects/.../project.toml \
    --ground-truth /tmp/claude_decisions.json --label-gt ClaudeSonnet --label-db Haiku

# Compare pipeline corpus against a list of papers from a Claude.ai chat session
# chat_papers.txt: one entry per line (arXiv ID, DOI, or free-form title)
# Produces: overlap / pipeline-only / chat-only (missed) / hallucinated counts
python scripts/compare_with_chat.py \
    --config projects/.../project.toml \
    --chat-papers chat_papers.txt \
    --out report.txt
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `SUPABASE_URL` | Yes | |
| `SUPABASE_SERVICE_KEY` | Either/or | Falls back to `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY` |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY` | Either/or | Used when no service key is set |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Without it, SPECTER2 embeddings are not fetched and `recommend` is unreliable |
| `ANTHROPIC_API_KEY` | No | Required for `synthesize`; optional for `screen`/`extract`/`iterate`/`run` when `--backend anthropic` |
| `GROQ_API_KEY` | No | Default LLM backend for screening (`--backend groq`) |
| `GEMINI_API_KEY` | No | For `--backend gemini` (Google AI Studio; free tier ~15 RPM = 4.5s/paper, paid tier sustainable ~30 RPM = 2.0s/paper) |
| `OPENROUTER_API_KEY` | No | For `--backend openrouter` (free models available) |

## Architecture

Two parallel sourcing tracks feed the same screening loop:
- **Citation traversal** (OpenAlex): follows citation graph forward/backward from included papers
- **S2 Recommendations** (SPECTER2): finds semantically similar papers from the global S2 index, using included papers as positives and excluded papers as negatives — catches parallel research communities not citation-connected to the seeds

```
projects/<name>/project.toml   # per-project config: seed IDs + inclusion criteria
litreview/
  config.py     # Settings loaded from .env; settings singleton imported everywhere
  models.py     # Paper + Author Pydantic models; db_dict() serialises for Supabase
  db.py         # Supabase CRUD: papers, citations, criteria, iterations
  ingest.py     # S2 + OpenAlex fetch, dedup, ingest_papers() entry point
  traverse.py   # Citation traversal via OpenAlex (forward/backward)
  recommend.py  # S2 Recommendations API sourcing: fetch_s2_recommendations()
  screen.py     # LLM screening (gemini/groq/openrouter/anthropic): screen_paper(), screen_project()
  extract.py    # Post-stability structured extraction: extract_paper(), extract_project()
  synthesize.py # Draft literature review generation from extractions: synthesize_project()
  download.py   # Download open-access PDFs (arXiv → S2 openAccessPdf fallback)
  audit.py      # Citation coverage audit: parse PDF refs, compare against DB corpus
  cli.py        # Typer CLI (entry point: `litreview`)
scripts/
  ingest_seeds.py       # Standalone equivalent of `litreview ingest`
  compare_screeners.py  # Compare two models' screening decisions vs a ground-truth JSON
supabase/
  schema.sql    # Canonical schema — apply via Supabase MCP or SQL editor
```

### Data model

- **projects** — one row per literature review, identified by `name`
- **papers** — deduped by `(project_id, s2_id)` and `(project_id, doi)` unique constraints (PostgREST requires full constraints, not partial indexes, for upsert conflict targeting); `embedding vector(768)` for SPECTER2; `source` is `'seed' | 'citation' | 'search'`; `depth` tracks distance from seeds. `Paper.db_dict()` drops all `None` fields before upsert AND excludes `_SCREENER_FIELDS` (`inclusion_status`, `rejection_reason`, `screening_round`, `criteria_version`) so traversal/ingest upserts never overwrite screening decisions. New rows receive the DB default (`inclusion_status = 'pending'`); existing rows keep their state.
- **criteria** — versioned inclusion/exclusion criteria; each screening decision logs `criteria_version`
- **iterations** — one row per search round; stores `yield_rate`, `overlap_rate`, `mean_embedding_distance` for stability checks
- **citations** — directed edges `(citing_paper_id, cited_paper_id)` within a project
- **extractions** — post-stability structured extraction per paper, stored as JSONB

### Criteria versioning

`project.toml` supports multiple `[criteria]` versions (`v1`, `v2`, …) — useful for tracking refinements within the file. However, `litreview ingest` only reads `v1` and stores it in the DB. To push a new criteria version to the DB, call `db.upsert_criteria(client, project_id, new_text)` directly or use the Supabase SQL editor. The screener always uses the latest criteria version from the DB (`get_current_criteria()`).

Example — adding `v2` to an existing `project.toml` for tracking:
```toml
[criteria]
v1 = "Include if ... (original)"
v2 = "Include if ... (refined)"
```
Then push `v2` manually: `db.upsert_criteria(client, project_id, cfg["criteria"]["v2"])`.

### Settings singleton (`config.py`)

`litreview/config.py` exports a module-level `settings = Settings()` that runs at import time. It immediately raises `RuntimeError` if `SUPABASE_URL` or either Supabase key is missing, so any CLI command will fail on startup without those vars — check `.env` first when debugging import errors.

### Ingestion flow (`ingest.py`)

1. Classify each identifier as `arxiv | doi | title`
2. Fetch from Semantic Scholar (primary source for metadata + SPECTER2 embeddings)
3. Enrich from OpenAlex (fills abstract, venue, authors, `openalex_id` when S2 is incomplete)
4. Deduplicate: DOI-exact first, then rapidfuzz `token_sort_ratio ≥ 92` on titles
5. Return `Paper` objects ready for `db.upsert_paper()`

**S2 API key behaviour:** `embedding` field is only requested when `SEMANTIC_SCHOLAR_API_KEY` is set — the free tier 403s on that field. Do not put placeholder text in the env var; leave it blank or comment it out.

**Retry policy:** S2 retries on 429/5xx only (not 403). Tenacity wraps exceptions in `RetryError`; catch both `httpx.HTTPStatusError` and `RetryError` at call sites.

### Traversal (`traverse.py`)

`traverse_citations()` fans out from papers at the current frontier depth via OpenAlex:
- **backward**: fetches `referenced_works` IDs, then batch-fetches full records (chunks of 50)
- **forward**: `Works().filter(cites=short_id)` — up to 200 citing papers per seed
- `--from-depth` controls which papers to traverse from (`get_frontier_depth()` returns the max depth of included papers)
- Papers are deduplicated, upserted with `source='citation'` and `depth=parent_depth+1`, and citation edges stored

OpenAlex IDs are resolved on-the-fly for papers that don't have one yet. Rate-limited with `time.sleep(0.5)` between papers.

### Screening (`screen.py`)

`screen_project()` iterates all `pending` papers, calling the configured LLM for each:
- **Gemini** (default): OpenAI-compatible endpoint; default model `gemini-2.5-flash`; 2.0 s/paper on paid tier (~30 RPM sustainable). Free tier is 4.5 s/paper but has burst-then-throttle behavior — prefer paid for large batches. Use `--rate-sleep 0.5` on high-quota paid tiers.
- **GROQ**: OpenAI-compatible API, default model `llama-3.3-70b-versatile`; free tier has a daily token quota (~100 papers/day empirically). Paid Developer plan: 30 RPM / 12K TPM for llama-3.3-70b-versatile — TPM is the binding constraint (~2.5s/paper at ~500 tokens/call). Use `--rate-sleep 0` only on higher enterprise tiers with lifted limits.
- **Anthropic**: direct API; default model `claude-haiku-4-5-20251001`; 0.5 s/paper
- **OpenRouter**: default model `qwen/qwen3-30b-a3b:free`; 6.0 s/paper (conservative; free models vary)
- System prompt embeds the current criteria; response format is `{"decision": ..., "reasoning": ...}`
- `include` → `inclusion_status = 'included'`
- `exclude` → `inclusion_status = 'excluded'`, reasoning in `rejection_reason`
- `uncertain` → stays `'pending'`, reasoning prefixed `"UNCERTAIN: "` in `rejection_reason`
- Yield rate logged to `iterations` table after each round

**Multi-backend failover**: pass a comma-separated list to `--backend`:
1. `probe_backend()` sends a test request to each; prints latency or "unavailable"
2. Backends are sorted by latency; screening starts with the fastest
3. After `_CONSECUTIVE_ERROR_LIMIT = 5` back-to-back errors, the screener pauses and prompts the user (retry / switch / quit) with a link to the backend's billing dashboard — so you can check quota before deciding. In `--non-interactive` mode (or when stdin is not a tty), it auto-switches if possible, or raises `RuntimeError`.
4. On backend switch, the new backend's rate sleep is applied immediately before the first call.
5. Progress summary printed every `_CHUNK_SIZE = 50` papers: counts, rate (s/paper), ETA, active backend

**Expected throughput** (sequential single-thread):

| Backend / Model | ~s/paper | Papers/30s | 1k papers | 10k papers | ~Cost/10k |
|---|---|---|---|---|---|
| Anthropic Haiku (default sleep) | ~1.1s | 27 | 18 min | 3.1 h | ~$8 |
| Gemini 2.5 Flash paid (default sleep) | ~2.8s | 10 | 47 min | 7.8 h | ~$0.75 |
| Groq `llama-3.3-70b-versatile` (Developer plan, default 2.0s sleep) | ~2.5s | 12 | 42 min | 7 h | $4.20 |
| Groq `llama-3.3-70b-versatile` (enterprise, `--rate-sleep 0`) | ~0.40s | 75 | 6.7 min | 67 min | $4.20 |
| Gemini free | ~5.3s | 6 | 88 min | 14.7 h | ~$0 |
| Groq free (any model) | ~1.5s | 20 | 25 min | *(quota ~100/day)* | ~$0 |
| OpenRouter free | ~9.0s | 3 | 2.5 h | 25 h | ~$0 |

Groq Developer plan limits for `llama-3.3-70b-versatile`: 30 RPM / 12K TPM / 100K TPD. TPM is the binding constraint (~500 tokens/call → ~24 RPM sustainable). Use `--rate-sleep 0` only on enterprise tiers with lifted limits. The screener prints an upfront ETA before starting and prompts for confirmation on batches >200 papers. Use `--yes` to skip, `--limit N` to cap a single run.

**Groq model priority** (set with `--model`):
1. `llama-3.3-70b-versatile` — **default**; best accuracy (~100% oracle agreement in tournament)
2. `meta-llama/llama-4-scout-17b-16e-instruct` — good quality/cost balance
3. `openai/gpt-oss-20b` — fastest + cheapest; lower accuracy (91% vs 100% in tournament)
4. `llama-3.1-8b-instant` — cheapest; test JSON reliability against your criteria before bulk use

**QA loop** (`--qa` flag on `screen`): after bulk screening, re-checks a stratified sample (default 75 papers, 30% includes / 70% excludes) with Claude Haiku (Anthropic). If agreement < threshold (default 90%), prints disagreements grouped by direction and exits with code 2. **Re-running QA without refining criteria produces the same result** — the loop must include a criteria update step, not just a re-screen.

If `ANTHROPIC_API_KEY` is not set, `--qa` writes the sample to `/tmp/qa_sample_<id>.json` and prints instructions for in-chat QA (read the file, make decisions in chat, compare manually).

**Screening strategy tiers** (use in order as batch size and throttling warrant):
1. **Probe + single backend**: `litreview screen --config ... --backend groq` — recommended default
2. **Multi-backend failover**: `litreview screen --config ... --backend groq,gemini` — automatic rotation if one backend hits limits
3. **Agent in-chat screening**: for ≤~100 remaining papers, read all titles/abstracts directly and write decisions to the DB via a Python script — bypasses all API rate limits entirely and gives instant feedback. Also the QA method when ANTHROPIC_API_KEY is not available. Example script pattern:
   ```python
   from litreview.db import get_client, get_papers, update_paper_screening
   client = get_client()
   # ... read papers, apply decisions, call update_paper_screening()
   ```

### Recommendations (`recommend.py`)

`fetch_s2_recommendations()` calls `POST /recommendations/v1/papers` with:
- **Positives**: seed papers + included papers, sorted by depth descending (frontier-first), truncated to the API max of 100
- **Negatives**: excluded papers, truncated to 100
- Returns up to 500 candidates upserted as `source='search'` at `frontier_depth + 1`

The signal sharpens over rounds as more papers are included/excluded. Papers from parallel research communities unreachable via citation links are the primary target. Requires `SEMANTIC_SCHOLAR_API_KEY` — the free tier is heavily rate-limited and may fail for large corpora.

`enrich_s2_ids(client, papers)` — called automatically before `fetch_s2_recommendations()` in `recommend`, `iterate`, and `run`. Resolves missing S2 IDs for papers that have a DOI but no `s2_id` (common for OpenAlex-sourced traversal papers), via S2 batch lookup. Patches the paper list in-place and updates the DB. Papers without S2 IDs are silently excluded from positives/negatives if this step is skipped.

### Audit (`audit.py`)

`audit_traversal()` / `audit_included()` compare a paper's reference list against the DB corpus:
- **PDF-first**: tries local file in project dir, then arXiv, then S2 `openAccessPdf`
- **S2 API fallback**: if no PDF available, fetches from `/paper/{id}/references`
- Matching uses DOI-exact → S2 ID → rapidfuzz `partial_ratio ≥ 88` on title vs raw entry text
- `recover_missing_refs()` fetches unmatched refs from S2 by title and upserts them as citation candidates

**S2 references API quirk**: the `data` field in the response can be `None` (not just `[]`) when a publisher elides the reference list. Always guard with `data.get("data") or []`, never `data.get("data", [])`.

### Full automated run (`litreview run`)

The `run` command executes: screen → download PDFs → audit included + recover → traverse → recommend, repeating until stability is declared (`is_stable()` checks yield rate over consecutive rounds and new paper count). Recommendations are on by default; disable with `--no-recommend`.

### Extraction (`extract.py`)

`extract_project()` runs post-stability over all included papers, calling the LLM once per paper (title + abstract) to populate the `extractions` table.

Universal fields (always extracted): `contribution_type`, `problem_statement`, `setting`, `methods`, `main_results`, `limitations`, `related_work_positioning`, `open_questions`.

Project-specific fields are defined in `project.toml` under `[[extraction.extra_fields]]` with `name` and `description` keys — appended to the universal schema in the prompt. See `projects/geometry-testing-simplicial-complexes/project.toml` for an example (adds `theorems`, `proof_techniques`, `statistical_regime`, `complexity_bounds`).

Skips papers without abstracts and already-extracted papers (override with `--force`). GROQ rate-limited at 1.2 s/paper.

### Synthesis (`synthesize.py`)

`synthesize_project()` loads all extractions, joins with paper metadata, and calls Claude to produce a markdown literature review written to `projects/<slug>/literature-review.md`.

- **≤150 papers**: single-pass synthesis
- **>150 papers**: map-reduce — chunks of 50 papers each produce thematic JSON summaries (map), which are then synthesised into a final narrative (reduce)
- Always uses Anthropic (requires `ANTHROPIC_API_KEY`); defaults to `claude-sonnet-4-6`
- Papers are formatted as structured blocks with citation keys `(AuthorEtAl Year)` for the LLM to use inline
- **Post-processing:** `_apply_numbered_bibliography()` automatically converts all `(Key)` / `Author (Year)` citations to `[N]` and appends a `## References` section — no manual cleanup needed

## Supabase schema notes

The schema in `supabase/schema.sql` is the source of truth. To apply it, use the Supabase MCP (`apply_migration`) or the Supabase SQL editor. `CREATE TYPE IF NOT EXISTS` is invalid in PostgreSQL — use a `DO $$ BEGIN ... EXCEPTION WHEN duplicate_object THEN null; END $$;` block for enum creation.

## Adding a new project

**Option A — from a folder of seed PDFs** (filenames must contain arXiv IDs like `2305.04802v2.pdf`):
```bash
litreview init <folder> --scope "Include if ..."
```
This writes `project.toml` into the folder, then runs ingest + traversal automatically.

**Option B — manual**:
1. Create `projects/<slug>/project.toml`:
   ```toml
   [project]
   name = "<slug>"

   [seeds]
   identifiers = ["<arxiv-id>", "<doi>", ...]  # arXiv IDs, DOIs, or title strings

   [criteria]
   v1 = """
   Include if ...
   Exclude if ...
   """
   ```
2. Run `litreview ingest --config projects/<slug>/project.toml`
3. No code changes needed — the schema is multi-project by design

## Iterative criteria refinement

Criteria can be tightened at any point — after every screen, traversal, or recommendation round:

1. Update criteria in the DB: `db.upsert_criteria(client, project_id, new_text)` or via Supabase SQL editor. This creates a new version; the screener always uses the latest.
2. Reset non-seed papers to pending: `litreview reset-screening --config ...` (seeds are never touched)
3. Re-screen: `litreview screen --config ...`

New papers from traversal/recommend are always `pending` and pick up the latest criteria automatically — only previously-screened papers need resetting.

**Before starting a new project:** validate your criteria against all seed papers first (`litreview screen --dry-run` on seeds only, or manually). If any seed fails its own criteria, the criteria are wrong. Fix before the first full screen run to avoid a costly re-screen round.

**Before each traversal (especially round 2+):** sample ~20 included and ~20 excluded papers from the previous screen round and review them manually. Check for false positives (included papers that don't belong) and tighten criteria accordingly before traversing further. Traversal fans out from every included paper, so false positives compound quickly — each extra included paper at depth N adds potentially hundreds of new candidates at depth N+1. The right sequence is: screen → review sample → tighten criteria if needed → re-screen → traverse.

**After each traversal:** check the `included` count before and after to verify the traversal didn't accidentally reset screening decisions. If the count drops unexpectedly, a bug may have clobbered `inclusion_status` via upsert — this was fixed in `models.py` by excluding `_SCREENER_FIELDS` from `db_dict()`, but verify if you see unexpected resets.

**Seed ID validation:** title-based S2 lookup can pull the wrong paper when the title is ambiguous or when the paper is very new. Always verify ingested seeds against the DB immediately after ingest — check title, year, and authors. If wrong papers appear, delete them via `client.table("papers").delete().eq("paper_id", id).execute()` and re-ingest with the correct identifier (prefer arXiv ID or DOI over title strings).

## Tests

No test suite exists yet. Test manually by running the CLI against a real project and verifying Supabase rows.