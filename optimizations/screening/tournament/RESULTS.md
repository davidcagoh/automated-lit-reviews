# Screening Tournament Results

**Project tested against:** `language-robotic-control`
**Ground truth:** 120 lrc papers screened in-chat by Claude Sonnet 4.6 using v3 criteria + `literature_review_cosmo.md` context
**Test set composition:** 42 include / 72 exclude (35% positive rate — oversampled to make accuracy discriminating; natural rate is ~13%)
**Metrics:** agreement % (primary), false-negative rate (secondary — missing relevant papers is the costly error)

---

## Round Summary

| Rnd | Contestant | Prompt | Agree | FNR | FPR | Lat/call | Cost/1k |
|-----|-----------|--------|------:|----:|----:|----------|---------|
| R1 | groq-llama70b | v1_baseline | 63% | 100% | 0% | 0.67s | $0.67 |
| R1 | groq-llama8b | v1_baseline | 63% | 100% | 0% | 0.44s | $0.04 |
| R1 | groq-llama4scout | v1_baseline | 63% | 100% | 0% | 0.51s | $0.08 |
| R1 | gemini-flash | v1_baseline | 63% | 100% | 0% | 3.30s | $0.13 |
| R2 | groq-llama70b-v2 | v2_include_by_default | 76% | 0% | 38% | 0.58s | $0.59 |
| R2 | groq-llama4scout-v2 | v2_include_by_default | 72% | 0% | 44% | 0.45s | $0.12 |
| R2 | gemini-flash-v2 | v2_include_by_default | 70% | 2% | 46% | 6.22s | $0.18 |
| **R3** | **groq-llama70b-v3** | **v3_language_primary_gate** | **89%** | **2%** | **15%** | **0.62s** | **$0.67** |
| **R3** | **groq-llama4scout-v3** | **v3_language_primary_gate** | **86%** | **0%** | **22%** | **0.46s** | **$0.14** |
| R3 | gemini-flash-v3 | v3_language_primary_gate | 77% | 7% | 32% | 4.96s | $0.20 |

*FPR = false positive rate on excluded papers. Cost/1k corrected to USD/1M token pricing.*
*Lat/call is API latency only — rate sleep (3.0s llama70b, 2.0s llama4scout, 0.2s gemini) adds to wall time.*

---

## Prompt Evolution

### v1_baseline — always-exclude collapse
Verbatim `_SYSTEM_PROMPT` from `screen.py`. Models interpreted the exclusion list
as a strict gate and never included anything: 100% FNR, 63% agreement (just the
true-negative rate from the 65% exclude majority).

**Root cause:** No default direction. Exclusion list over-triggers on technically-framed HCI papers.

### v2_include_by_default — over-inclusion
Added `DEFAULT RULE: include when uncertain` and a broad tiebreaker:
"language + robots = include." FNR dropped to 0-2% but FPR ballooned to 38-46%.

**Root cause:** Tiebreaker was too permissive. Models included HRI-without-language (gaze,
trust studies), NLP papers with implied-not-demonstrated robotic use, autonomous vehicles.

**False positive taxonomy (from llama70b-v2, 27 FPs):**
1. HRI papers about gaze/gesture/trust without language interface (7 papers)
2. NLP/language papers where "implications for robotics" were inferred, not demonstrated (8 papers)
3. Autonomous vehicles without language (2 papers)
4. Social robot appearance/sound with no language component (4 papers)
5. Vision/perception robotics where language was incidental (6 papers)

### v3_language_primary_gate — winner (86-89% agree)
Added STEP 1 gating question before criteria: "Is language/speech the PRIMARY interface?"
with explicit exclusion clarifications for each FP category. Kept DEFAULT RULE for genuine
borderlines.

**Result:** llama70b 89%/2%FNR, llama4scout 86%/0%FNR — well-calibrated. Gemini lagged at 77%/7%.

---

## Recommendation for lrc Bulk Screening (8,100 papers)

Priority order: **fastest → most accurate → cheapest**

### Option A: groq-llama4scout + v3 ← recommended
```bash
litreview screen --config projects/language-robotic-control/project.toml \
    --backend groq --model meta-llama/llama-4-scout-17b-16e-instruct
```
- **Speed:** 2.0s/paper → ~4.5 hours for 8,100 papers (single Groq Developer plan session, no RPD cap)
- **Accuracy:** 86% agree / **0% FNR** / 22% FPR
- **Cost:** ~$1.13 total
- **Why over llama70b:** FNR=0% (vs 2% — never misses relevant papers), 33% faster, 4.8x cheaper. The 3% agreement delta doesn't justify the extra cost/time.
- **Why over gemini:** Groq has no RPD cap; Gemini Tier 1 ~1,500 RPD → 5-6 days for 8,100 papers despite 0.2s/call

### Option B: groq-llama70b + v3 (highest accuracy)
```bash
litreview screen --config projects/language-robotic-control/project.toml \
    --backend groq --model llama-3.3-70b-versatile
```
- **Speed:** 3.0s/paper → ~6.75 hours
- **Accuracy:** 89% agree / 2% FNR / 15% FPR
- **Cost:** ~$5.43 total
- Choose this if accuracy matters more than speed and the 1 in 50 missed papers would be unacceptable.

### Option C: gemini-flash + v3 (fastest per-call, RPD-limited in practice)
```bash
litreview screen --config projects/language-robotic-control/project.toml \
    --backend gemini --rate-sleep 0.2
```
- **Speed:** 0.2s/call but ~1,500 RPD on Tier 1 → 5-6 days for 8,100 papers
- **Accuracy:** 77% agree / 7% FNR / 32% FPR — consistently worse than Groq models
- Only viable on Tier 2 (10,000 RPD → ~1 day, still with 7% FNR)
- Not recommended unless you need < 30-min test runs on a small batch

### Note on lrc-specific prompt
The v3 prompt is now the default `_SYSTEM_PROMPT` in `litreview/screen.py`. The system
prompt is generic enough to work on other projects (the gating question is universal for
language-HRI work), but `v1_baseline` is preserved in `prompts/` if you need to revert for
a different domain.

---

## Files

```
optimizations/screening/tournament/
├── RESULTS.md                        ← this file
├── ground_truth.json                 ← 120 lrc papers, Claude Sonnet labels
├── candidate_test_set.json           ← same 120 papers with abstracts for LLM calls
├── sampler.py                        ← stratified test set generator
├── wrapper.py                        ← instrumented screen call (token counts + latency)
├── metrics.py                        ← agreement + FNR computation
├── run_tournament.py                 ← tournament runner CLI
├── prompts/
│   ├── v1_baseline.txt               ← original system prompt (always-exclude on lrc)
│   ├── v2_include_by_default.txt     ← fixed FNR but caused over-inclusion
│   └── v3_language_primary_gate.txt  ← winner: gate + default-include + FP clarifications
├── rounds/
│   ├── round_1.toml                  ← 4 models × v1 (llama70b, llama8b, llama4scout, gemini)
│   ├── round_2.toml                  ← 3 models × v2 (dropped llama8b)
│   └── round_3.toml                  ← 3 models × v3
└── results/
    ├── leaderboard.json              ← cross-round summary
    ├── round_1_results.json          ← round 1 aggregated
    ├── round_2_results.json          ← round 2 aggregated
    └── round_3_results.json          ← round 3 aggregated
```
