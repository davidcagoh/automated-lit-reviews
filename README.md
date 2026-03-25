# litreview

Automated systematic literature review pipeline. Give it seed papers and inclusion criteria; it traverses citations, fetches semantically similar papers, screens candidates with an LLM, extracts structured data from included papers, and writes a numbered, cited literature review.

---

## Quick start

```bash
# 1. Install
pip install -e .
cp .env.example .env   # fill in keys (see Environment variables below)

# 2. Apply DB schema (once, in Supabase SQL editor or via MCP)
#    paste contents of supabase/schema.sql

# 3. New project from a folder of seed PDFs
litreview init my-papers/ --scope "Include if the paper addresses X. Exclude if Y."

# 4. Run to stability (traverse + screen + recommend, repeat)
litreview run projects/my-papers/project.toml

# 5. Extract structured fields, then synthesise
litreview extract --config projects/my-papers/project.toml
litreview synthesize --config projects/my-papers/project.toml
```

Output: `projects/my-papers/literature-review.md` — a full narrative review with numbered citations and a `## References` section.

Human time required: writing the scope statement, adjudicating a handful of `uncertain` papers per round, and optionally tightening criteria mid-run.

---

## Full guide

### Prerequisites

- Python ≥ 3.11
- A [Supabase](https://supabase.com) project (free tier works)
- API keys — see table below

### Installation

```bash
git clone <repo>
cd automated-lit-reviews
pip install -e .
```

### Environment variables

Create `.env` in the repo root:

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_KEY=<service-role key>      # or use the publishable key below
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=<anon/publishable key>

SEMANTIC_SCHOLAR_API_KEY=<key>   # needed for SPECTER2 embeddings + recommend
ANTHROPIC_API_KEY=<key>          # needed for synthesize; optional for screen/extract

GROQ_API_KEY=<key>               # default screening backend (free, fast)
GEMINI_API_KEY=<key>             # alternative screening backend
OPENROUTER_API_KEY=<key>         # alternative screening backend
```

`SUPABASE_SERVICE_KEY` is preferred (bypasses RLS). Falls back to `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY` if absent.

Only one LLM backend key is needed for screening. Groq is the default and has a generous free tier. Gemini free tier is 15 RPM. Anthropic requires a paid API key.

`SEMANTIC_SCHOLAR_API_KEY` is technically optional but without it SPECTER2 embeddings are not fetched and `recommend` is unreliable.

`ANTHROPIC_API_KEY` is required only for `synthesize`. All other steps work without it.

### Database setup

Apply `supabase/schema.sql` once to your Supabase project. Use the Supabase SQL editor (Dashboard → SQL Editor → paste and run) or the Supabase MCP tool. The schema is idempotent — safe to re-apply.

Tables created: `projects`, `papers`, `citations`, `criteria`, `iterations`, `extractions`.

---

### Starting a new project

**Option A — from a folder of seed PDFs** (recommended)

PDF filenames must contain the arXiv ID, e.g. `2305.04802v2.pdf`. The `init` command extracts IDs from filenames automatically.

```bash
litreview init path/to/seed-pdfs/ --scope "Include if ... Exclude if ..."
```

This creates `path/to/seed-pdfs/project.toml`, ingests the seeds, and runs one traversal round.

**Option B — manual project.toml**

```toml
[project]
name = "my-project-slug"

[seeds]
identifiers = ["2305.04802", "10.1145/3519935.3519989", "some paper title"]

[criteria]
v1 = """
Include if the paper addresses X, Y, or Z.
Exclude if the paper only covers A without B.
"""
```

Then: `litreview ingest --config projects/my-project-slug/project.toml`

Identifiers can be arXiv IDs, DOIs, or free-text titles — the ingestion step classifies and resolves them.

**Writing good criteria**

Criteria are the only thing that can't be automated — they encode your domain judgment. A few rules:

- Be specific about the core phenomenon, not just the topic area. "Papers about random graphs" is too broad; "papers that prove or conjecture a TV-distance or detection threshold for a geometric random graph model" is actionable.
- Include a clear exclude clause for the adjacent literature that will show up in citation traversal but doesn't belong.
- **Validate against your seed papers first.** Run `litreview screen --dry-run` (or mentally apply the criteria) to every seed. If any seed fails, the criteria are too tight. Fix before running the full pipeline — a bad criteria definition discovered after 3 screening rounds costs a full re-screen.

---

### Pipeline commands

#### Ingest

```bash
litreview ingest --config projects/.../project.toml
litreview ingest --config ... --no-openalex   # skip OpenAlex enrichment
litreview ingest "my-project" arxiv:2305.04802 doi:10.1145/... "Paper Title"
```

Fetches metadata from Semantic Scholar (primary) and OpenAlex (enrichment), deduplicates by DOI then by title similarity (rapidfuzz token_sort_ratio ≥ 92), and upserts into `papers`.

#### Traverse

```bash
litreview traverse --config projects/.../project.toml
litreview traverse --config ... --direction backward   # forward | backward | both (default: both)
litreview traverse --config ... --from-depth 1         # traverse from depth-1 included papers
```

Follows the citation graph via OpenAlex. Backward = references of included papers. Forward = papers that cite included papers (up to 200 per seed). New candidates are inserted as `pending` at `depth = parent_depth + 1`.

#### Recommend

```bash
litreview recommend --config projects/.../project.toml
litreview recommend --config ... --limit 200 --dry-run
```

Calls the Semantic Scholar Recommendations API (SPECTER2) with included papers as positives and excluded papers as negatives. Returns semantically similar papers from outside the citation graph — useful for finding parallel research communities. Signal improves over rounds as more papers are included/excluded. Requires `SEMANTIC_SCHOLAR_API_KEY`.

#### Screen

```bash
litreview screen --config projects/.../project.toml
litreview screen --config ... --backend gemini          # gemini-2.5-flash
litreview screen --config ... --backend anthropic --model claude-haiku-4-5-20251001
litreview screen --config ... --backend openrouter      # qwen/qwen3-30b-a3b:free default
litreview screen --config ... --dry-run                 # print decisions, write nothing
```

Screens all `pending` papers using the latest criteria version from the DB. Each paper gets `included`, `excluded`, or `uncertain`. Uncertain papers stay `pending` with reasoning in `rejection_reason` — review them manually and set status via Supabase or a follow-up screen pass.

Default backend is Groq (llama-3.3-70b-versatile, 30 RPM free tier → ~1.2 s/paper).

#### Iterate / Run

```bash
# One traverse + screen cycle, optionally looping until stable
litreview iterate --config projects/.../project.toml
litreview iterate --config ... --loop --max-rounds 10 --yield-threshold 0.05
litreview iterate --config ... --recommend   # add S2 recommendations track

# Full automated pipeline (screen → download → audit → traverse → recommend → repeat)
litreview run projects/.../project.toml
litreview run projects/.../project.toml --backend gemini --max-rounds 5
litreview run projects/.../project.toml --no-recommend   # citation traversal only
```

`run` is the main entrypoint for a new project after ingest. It loops until stable (yield rate falls below threshold over consecutive rounds). `--recommend` is on by default in `run`.

Stability is declared when both the yield rate (% new candidates included) and new paper count fall below the threshold for two consecutive rounds.

#### Extract

```bash
litreview extract --config projects/.../project.toml
litreview extract --config ... --backend gemini
litreview extract --config ... --force     # re-extract already-extracted papers
litreview extract --config ... --dry-run
```

Run post-stability. Calls the LLM once per included paper (title + abstract) to populate the `extractions` table with structured fields. Universal fields: `contribution_type`, `problem_statement`, `setting`, `methods`, `main_results`, `limitations`, `related_work_positioning`, `open_questions`. Project-specific fields are defined in `project.toml` under `[[extraction.extra_fields]]`.

Skips papers without abstracts (they're in the DB but can't be extracted). Use `--force` to re-extract after changing the field schema.

#### Synthesize

```bash
litreview synthesize --config projects/.../project.toml
litreview synthesize --config ... --model claude-opus-4-6
litreview synthesize --config ... --output /tmp/draft.md
```

Requires `ANTHROPIC_API_KEY`. Loads all extractions, calls Claude to write a thematic narrative review in markdown, then automatically post-processes the output to replace `(Author Year)` citations with numbered references `[N]` and append a `## References` section.

For ≤150 papers: single-pass. For >150: map-reduce (50-paper chunks → thematic summaries → final synthesis).

Output: `projects/<slug>/literature-review.md`.

#### Utilities

```bash
# Download open-access PDFs for included papers
litreview download-pdfs --config projects/.../project.toml
litreview download-pdfs --config ... --depth 1   # only depth-1 papers

# Audit citation coverage (compares reference lists against DB corpus)
litreview audit --config projects/.../project.toml
litreview audit --config ... --included          # audit from PDFs (needs download-pdfs first)
litreview audit --config ... --recover           # fetch unmatched refs from S2 and upsert
litreview audit --config ... --included --recover --dry-run

# Reset screened papers back to pending (seeds never touched)
litreview reset-screening --config projects/.../project.toml
litreview reset-screening --config ... --depth 1 --yes

# Verbose logging (global flag, before subcommand)
litreview --verbose run projects/.../project.toml
```

#### Analysis scripts

```bash
# Compare two models' screening decisions against a ground-truth JSON
python scripts/compare_screeners.py --config projects/.../project.toml \
    --ground-truth /tmp/claude_decisions.json --label-gt ClaudeSonnet --label-db Haiku

# Compare pipeline corpus against a manually compiled paper list
# chat_papers.txt: one entry per line (arXiv ID, DOI, or title)
python scripts/compare_with_chat.py \
    --config projects/.../project.toml \
    --chat-papers chat_papers.txt \
    --out report.txt
```

---

### Iterative criteria refinement

Criteria can be tightened at any point — after any screen, traversal, or recommendation round:

1. **Update criteria** in the DB. Either call `db.upsert_criteria(client, project_id, new_text)` in a script, or run SQL directly in the Supabase editor:
   ```sql
   insert into criteria (project_id, version, content, trigger)
   values ('<project_id>', <next_version>, '<new criteria text>', 'reason for change');
   ```
   The screener always uses the latest version.

2. **Reset** previously-screened papers back to pending:
   ```bash
   litreview reset-screening --config projects/.../project.toml --yes
   ```
   Seeds are never reset regardless of flags.

3. **Re-screen**:
   ```bash
   litreview screen --config projects/.../project.toml
   ```

New papers added by traversal or recommend are always `pending` and automatically use the latest criteria — only previously-screened papers need resetting.

---

### Architecture

Two parallel sourcing tracks feed the same screening loop:

```
Seeds (arXiv / DOI / title)
    │
    ▼
 ingest.py  ──── Semantic Scholar (metadata + SPECTER2)
                 OpenAlex (abstract / venue / authors enrichment)
    │
    ├──── traverse.py  ─── OpenAlex citation graph (forward + backward)
    │                      → new candidates at depth+1
    │
    └──── recommend.py ─── S2 Recommendations API (SPECTER2 similarity)
                           positives: included papers
                           negatives: excluded papers
                           → semantically similar candidates outside citation graph
    │
    ▼
 screen.py  ──── LLM (Groq / Gemini / Anthropic / OpenRouter)
                 include | exclude | uncertain
    │
    ▼ (loop until stable)
    │
 extract.py ──── LLM structured extraction per included paper
    │
    ▼
 synthesize.py ─ Claude (Anthropic) → narrative review → numbered bibliography
```

**Why two tracks?** Citation traversal is precise but misses parallel research communities not citation-linked to your seeds. SPECTER2 recommendations catch these, at the cost of lower precision. Together they give better recall than either alone.

---

### Data model

| Table | Key columns | Notes |
|---|---|---|
| `projects` | `project_id`, `name` | One row per review |
| `papers` | `project_id`, `s2_id`, `doi`, `inclusion_status`, `source`, `depth` | Deduped by `(project_id, s2_id)` and `(project_id, doi)`; `source` ∈ `seed\|citation\|search`; SPECTER2 embedding at `vector(768)` |
| `criteria` | `project_id`, `version`, `content` | Versioned; screener uses `max(version)` |
| `iterations` | `round`, `yield_rate`, `overlap_rate`, `mean_embedding_distance` | One row per screening round; used for stability checks |
| `citations` | `citing_paper_id`, `cited_paper_id` | Directed edges within a project |
| `extractions` | `paper_id`, `project_id`, `data` (JSONB) | One row per included paper post-stability |

---

### Project-specific extraction fields

Add custom extraction fields in `project.toml`:

```toml
[[extraction.extra_fields]]
name = "theorems"
description = "List the main theorems or formal results stated."

[[extraction.extra_fields]]
name = "proof_techniques"
description = "Key mathematical techniques used in the proofs."
```

These are appended to the universal schema (`contribution_type`, `problem_statement`, `setting`, `methods`, `main_results`, `limitations`, `related_work_positioning`, `open_questions`) in the extraction prompt.

---

### Known gotchas

**Supabase 1000-row default limit.** The PostgREST API returns at most 1000 rows by default. All DB reads in this codebase use explicit pagination — don't bypass them with raw `.execute()` calls on large tables.

**S2 API and embeddings.** The `embedding` field requires `SEMANTIC_SCHOLAR_API_KEY`. Without it the field is omitted from S2 requests (the free tier returns 403 on that field). Do not set the key to a placeholder string — leave it blank or comment it out, otherwise the 403 is treated as an error rather than gracefully skipped.

**S2 retry policy.** Tenacity wraps S2 calls and retries on 429/5xx only, not 403. Catch both `httpx.HTTPStatusError` and `tenacity.RetryError` at call sites.

**LLM citation drift in synthesis.** `synthesize` instructs the LLM to use the exact citation key from each paper's block header. The post-processor matches both parenthetical `(Author Year)` and inline `Author (Year)` forms, and handles `&` vs `and` variants. If the LLM invents an author string not matching any key, that citation is left as-is in the output with a warning logged — the bibliography won't be wrong, just incomplete for that citation.

**OpenAlex rate limiting.** Traversal sleeps 0.5 s between papers. For large corpora at depth 2 this can take hours. Run overnight or use `--from-depth` to limit scope.

**`uncertain` papers.** The screener leaves uncertain papers as `pending` with `"UNCERTAIN: ..."` in `rejection_reason`. They are not re-screened automatically. Review them manually (Supabase table view or a custom query) and set `inclusion_status` directly, or adjust criteria and re-screen.

**Synthesis requires Anthropic.** `litreview synthesize` only calls the Anthropic API — there is no `--backend` flag here. All other steps work without an Anthropic key.
