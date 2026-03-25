# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python ≥ 3.11.

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
# Backends: groq (default), gemini, openrouter, anthropic
litreview screen --config projects/.../project.toml
litreview screen --config ... --backend anthropic --model claude-haiku-4-5-20251001
litreview screen --config ... --backend gemini               # gemini-2.5-flash default
litreview screen --config ... --backend openrouter           # qwen/qwen3-30b-a3b:free default
litreview screen --config ... --dry-run                     # prints decisions, nothing written

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
| `GEMINI_API_KEY` | No | For `--backend gemini` (Google AI Studio; 15 RPM free tier) |
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
  screen.py     # LLM screening (GROQ or Anthropic): screen_paper(), screen_project()
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
- **papers** — deduped by `(project_id, s2_id)` and `(project_id, doi)` unique constraints (PostgREST requires full constraints, not partial indexes, for upsert conflict targeting); `embedding vector(768)` for SPECTER2; `source` is `'seed' | 'citation' | 'search'`; `depth` tracks distance from seeds
- **criteria** — versioned inclusion/exclusion criteria; each screening decision logs `criteria_version`
- **iterations** — one row per search round; stores `yield_rate`, `overlap_rate`, `mean_embedding_distance` for stability checks
- **citations** — directed edges `(citing_paper_id, cited_paper_id)` within a project
- **extractions** — post-stability structured extraction per paper, stored as JSONB

### Criteria versioning

`project.toml` supports multiple `[criteria]` versions (`v1`, `v2`, …), but `litreview ingest` only reads `v1` and stores it in the DB. To update live criteria, upsert directly via `db.upsert_criteria()` or the Supabase SQL editor. The screener always uses the latest criteria version from the DB (`get_current_criteria()`).

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
- **GROQ** (default): OpenAI-compatible API, default model `llama-3.3-70b-versatile`; rate-limited at 1.2 s/paper (30 RPM free tier)
- **Anthropic**: uses `anthropic` SDK; default model `claude-haiku-4-5-20251001`
- System prompt embeds the current criteria; response format is `{"decision": ..., "reasoning": ...}`
- `include` → `inclusion_status = 'included'`
- `exclude` → `inclusion_status = 'excluded'`, reasoning in `rejection_reason`
- `uncertain` → stays `'pending'`, reasoning prefixed `"UNCERTAIN: "` in `rejection_reason`
- Yield rate logged to `iterations` table after each round

### Recommendations (`recommend.py`)

`fetch_s2_recommendations()` calls `POST /recommendations/v1/papers` with:
- **Positives**: seed papers + included papers, sorted by depth descending (frontier-first), truncated to the API max of 100
- **Negatives**: excluded papers, truncated to 100
- Returns up to 500 candidates upserted as `source='search'` at `frontier_depth + 1`

The signal sharpens over rounds as more papers are included/excluded. Papers from parallel research communities unreachable via citation links are the primary target. Requires `SEMANTIC_SCHOLAR_API_KEY` — the free tier is heavily rate-limited and may fail for large corpora.

### Audit (`audit.py`)

`audit_traversal()` / `audit_included()` compare a paper's reference list against the DB corpus:
- **PDF-first**: tries local file in project dir, then arXiv, then S2 `openAccessPdf`
- **S2 API fallback**: if no PDF available, fetches from `/paper/{id}/references`
- Matching uses DOI-exact → S2 ID → rapidfuzz `partial_ratio ≥ 88` on title vs raw entry text
- `recover_missing_refs()` fetches unmatched refs from S2 by title and upserts them as citation candidates

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

## Tests

No test suite exists yet. Test manually by running the CLI against a real project and verifying Supabase rows.