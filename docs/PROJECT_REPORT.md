# WikiPulse — Design Notes for Project Report

This document summarizes the enrichment, training, and verification improvements
implemented for the WikiPulse streaming pipeline. It is written so sections can
be copied into a course or team project report.

## 1. Problem statement

WikiPulse detects unusual Wikipedia edit activity in near real time, then checks
whether those anomalies correspond to concurrent real-world news. The pipeline
has three coupled stages:

1. **Ingest** Wikimedia recent-change events and GDELT news articles into Kafka.
2. **Detect** per-page edit bursts with an Isolation Forest model over 15-minute
   windows (`spark/ml_inference_stream.py`).
3. **Verify** anomalies against recent news via sentence-transformer embeddings
   and cosine similarity search in Qdrant (`vectordb/match_events.py`).

Two gaps limited the quality of stage 3 and the validity of stage 2:

- Embeddings were effectively **title-only**, so semantic matching had little
  topical context from either Wikipedia edits or news bodies.
- The anomaly model was trained on **lifetime per-page aggregates** from
  historical edits, but scored live against **15-minute windows** — a
  train/serve granularity mismatch that over-flagged ordinary edit bursts.

## 2. Richer embeddings

### 2.1 News articles: title + body snippets

GDELT’s Web NGrams feed publishes a companion `.ngrams.txt.gz` file alongside
each minute’s table-of-contents (TOC) file. The GDELT producer now:

1. Fetches that companion ngrams file for each processed timestamp.
2. Parses tab-delimited quadgrams (4-word phrases) keyed by DOCID.
3. Builds a short `snippet` per article (top-count, deduplicated phrases).
4. Publishes `snippet` on `news-topic` (empty string when unavailable).

Embedding text (`vectordb.embeddings.gdelt_article_text`) becomes
`title` + `snippet` rather than title alone. Snippets are not full article
text; they are overlapping fragments that still add topical signal for
similarity search.

### 2.2 Wikipedia anomalies: title + cleaned edit summaries

The Wikimedia SSE stream includes a per-edit `comment` (edit summary). The
pipeline now:

1. Captures `comment` in `producer/wikipedia_producer.py`.
2. Carries it through Spark schemas (`spark/wiki_stream.py`,
   `spark/ml_inference_stream.py`).
3. Filters and cleans comments with `vectordb.embeddings.substantive_comment_text`:
   - strips MediaWiki revert/rollback boilerplate,
   - unwraps `/* section */` markers,
   - converts wikilinks to display text,
   - strips template markup,
   - rejects jargon-only / maintenance-only summaries (e.g. `ce`, `rv`,
     `created article`, `/* See also */`).
4. Aggregates up to five surviving summaries into `recent_comments` on
   `wikipedia-anomalies`.

Embedding text (`wikipedia_anomaly_text`) becomes
`page_title` + `recent_comments`.

### 2.3 Why this matters for verification

Anomaly verification remains cosine similarity over local
`sentence-transformers` embeddings (`all-MiniLM-L6-v2` by default). Adding
body/edit context reduces false negatives from title phrasing differences
(e.g. “wildfire” vs “blaze”) while keeping the design API-key-free and local.

## 3. Windowed anomaly training

### 3.1 Previous approach (lifetime aggregation)

Historical training (`models/historical_training.py`) used to group all edits
for a page across the entire downloaded history (~30-day MediaWiki
`recentchanges` retention) into one feature row:

- `edit_count`, `unique_editors`, `total_byte_changes`, ratios, and derived
  features (`relative_growth`, `human_bot_friction`, `editor_concentration`).

Live inference aggregates the **same features** over a **15-minute sliding
window**. Because Isolation Forest has no notion of elapsed time, a page that
received 9 edits / ~44 KB in 15 minutes looked like “a huge fraction of a
lifetime page total,” even when that burst would be unremarkable if spread
over weeks.

Empirical check on the 150k-edit training file before the fix:

- ~76.8% of pages had **exactly one lifetime edit**.
- Live flags such as large page creations were often driven by extreme
  `relative_growth` (bytes per edit) relative to a lifetime baseline.
- Re-scoring confirmed byte volume as the main driver: identical edit/editor
  counts with typical byte volumes scored as normal.

### 3.2 New approach (15-minute tumbling windows)

Training now buckets edits into non-overlapping **15-minute tumbling windows**
per page (`WINDOW_SECONDS = 900`), matching the live window duration used by
Spark (`window(..., "15 minutes", "5 minutes")`). Sliding vs tumbling need not
be identical; the important property is that feature magnitudes are computed
over a comparable time span.

After re-aggregation on the same 150k-edit file:

| Metric | Lifetime (old) | Windowed (new) |
|---|---:|---:|
| Training rows | ~95k pages | ~123k (page, window) samples |
| Rows with `edit_count == 1` | ~77% | ~86% |
| Model self-flag rate at `contamination=0.01` | calibrated | **1.00%** of windows |

Top anomalous windows in the retrained training set look qualitatively more
plausible (multi-editor / high-byte bursts on recognizable topics such as
“Donald Trump”, “Andy Beshear”, coordinated game-wiki pages) than isolated
maintenance-style single-edit page creations alone.

### 3.3 More historical data

Default download limit raised from **20,000 → 150,000** edits in
`scripts/download_wikimedia_history.py`, closer to MediaWiki’s ~30-day
`recentchanges` retention. More data alone does **not** fix the lifetime vs
window mismatch; windowed aggregation is the structural fix. Larger history
still helps by giving more (page, window) samples for RobustScaler /
Isolation Forest.

### 3.4 What “valid anomaly” means

An Isolation Forest score `< 0` means “statistically unusual relative to the
training distribution,” not “confirmed newsworthy.” News verification is a
separate stage. After windowed retraining:

- Large single-edit page dumps can still be rare and correctly flagged.
- Multi-edit bursts are judged against other 15-minute windows, so
  `edit_count` carries real signal.
- Validity for the product goal (news-relevant Wikipedia spikes) still depends
  on the embedding match stage, not the anomaly detector alone.

## 4. Dashboard and operational improvements

Dashboard (`dashboard/app.py`) changes for operator usability:

- Show **Recent edit summaries** (`recent_comments`) on anomaly tables.
- Recency filter (default last 60 minutes) and dedupe latest window per page.
- **Anomalies verified against real news** shows confirmed matches only.
- Wikipedia page links, “Detected at” timestamps, newest-first sorting.
- Wider text column for edit summaries.

Operational notes discovered during end-to-end runs:

- Spark loads `models/anomaly_detector.joblib` once at startup; retrain requires
  restarting `spark/ml_inference_stream.py`.
- Set `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` to the project venv Python when
  the shell’s default interpreter differs (worker/driver minor-version mismatch).
- Kafka archival stream uses `failOnDataLoss=false` so topic resets do not hard
  fail the query; stale Spark checkpoints may still need clearing after a
  full topic recreate.

## 5. Tests and reproducibility

New / updated tests:

- `tests/test_embeddings.py` — snippet/title text builders, comment cleaning,
  jargon/maintenance filtering.
- `tests/test_gdelt_producer.py` — ngrams fetch, snippet build, end-to-end
  snippet attachment.
- `tests/test_historical_training.py` — edits in different windows stay
  separate; edits in the same window merge.

Reproducible training path:

```bash
python scripts/download_wikimedia_history.py   # or reuse existing JSONL
python models/train_model.py
# then restart spark/ml_inference_stream.py
```

Artifacts:

- Historical edits: `data/historical/en.wikipedia.org.recentchanges.jsonl`
  (default 150k edits).
- Trained model: `models/anomaly_detector.joblib` (regenerable; small binary).

## 6. Pipeline architecture (current)

```text
Wikimedia SSE ──► wikipedia_producer ──► kafka:wikipedia-edits
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                 wiki_stream (archive)   ml_inference_stream   (consumers)
                                              │
                                              ▼
                                   kafka:wikipedia-anomalies
                                              │
GDELT TOC+ngrams ──► gdelt_producer ──► kafka:news-topic
                                              │
                                              ▼
                                   match_events (+ Qdrant)
                                              │
                                              ▼
                                   kafka:anomaly-news-verdicts
                                              │
                                              ▼
                                      Streamlit dashboard
```

Feature vector (training + inference, 8 columns):

1. `edit_count`
2. `unique_editors`
3. `total_byte_changes`
4. `bot_ratio`
5. `minor_edit_ratio`
6. `relative_growth` = `total_byte_changes / edit_count`
7. `human_bot_friction` = `minor_edit_ratio - bot_ratio`
8. `editor_concentration` = `unique_editors / edit_count`

Model: `RobustScaler` → `IsolationForest(contamination=0.01)`.

## 7. Results observed during validation

Qualitative / quantitative checks from the windowed retrain session:

- **71** unit tests passing after the changes.
- Retrained model self-flags **1.00%** of training windows (matches
  contamination setting).
- Live pipeline continued publishing to `wikipedia-anomalies` and
  `anomaly-news-verdicts` after restart with the new model.
- Example score movement under the new model (illustrative of
  `relative_growth` / burst sensitivity, not a claim of news confirmation):
  - large page-creation style vectors remained clearly anomalous;
  - multi-edit high-byte windows scored more strongly anomalous once
    compared to other 15-minute windows rather than lifetime totals.

News confirmation (`matched: true` on `anomaly-news-verdicts`) remains sparse
by design: it requires concurrent topical overlap between a statistical edit
burst and recent GDELT articles above `SIMILARITY_THRESHOLD` (default 0.7).
(The topic was renamed from `verified-events` to reflect that it stores all
verdicts, including `matched: false`.)

### 7.1 Finding: sparse overlap between Wikipedia edits and GDELT news

A consistent live-run observation is that the two feeds mostly **do not
line up**:

1. **Most Wikimedia edits we capture are routine activity on already-
   established pages** (maintenance, formatting, sports/box-score style
   updates, quiet article growth). Those edits usually do **not** match a
   concurrent GDELT article above the similarity threshold — i.e. they are
   not behaving like “breaking news just hit the page.”
2. **Most GDELT articles never appear as a verified anomaly.** The news
   firehose is large; only a small fraction coincide with a statistically
   anomalous Wikipedia edit burst on a topically similar page.

So the dashboard’s “Recent GDELT news” table being much fuller than
“Anomalies verified against real news” is expected, not a pipeline failure.
Verified rows are the rare intersection: a sharp wiki burst *and* a close
news match (e.g. `2026 Colombia earthquake` ↔ a quake headline).

**What we can and cannot claim**

| Fair claim from this system | Overclaim to avoid |
|---|---|
| Under our anomaly + embedding match definition, wiki↔news correlation is sparse. | “Wikipedia pages were never updated for that story” (we only see *anomalous* bursts, not every quiet edit). |
| Most captured edits do not look like news-driven spikes. | “Most edits are to very old pages” as a measured result (we do not score page age). |
| Most news items never clear verification. | “Those stories have no Wikipedia page at all.” |

**Which comes first — Wikipedia or mainstream news?**

This pipeline cannot settle a societal “chicken or egg” by itself: the two
streams are ingested **independently**, and verification only checks
**near-simultaneous topical overlap**, not causal order. Qualitatively,
though, the sparse intersection fits a common pattern for breaking events:

- mainstream outlets often publish first;
- Wikipedia editors then update (or create) articles, sometimes in a burst
  that our anomaly detector can see;
- the reverse also happens (wiki maintenance with no news; news with no
  wiki spike).

In that sense WikiPulse is less “which medium leads society?” and more
“when do both move together loudly enough to detect?” — and the empirical
answer from our runs is: **rarely, and those rare cases are the interesting
ones.**

### 7.2 Example: sliding windows vs. one news article

Live Spark scoring uses a **15-minute window that slides every 5 minutes**.
While a breaking-news page keeps receiving edits, the scorer re-emits that
page for each overlapping window (e.g. 10:10–10:25, then 10:15–10:30, then
10:20–10:35). The news matcher evaluates **each emission independently**,
so the same GDELT article can appear multiple times with slightly different
cosine similarities as the window’s edit count / `recent_comments` (and thus
the anomaly embedding) change.

**Observed example (2026-08-11, live run):**

| Wikipedia page | Matched news (GDELT) | Example windows (UTC) | Similarity range |
|---|---|---|---|
| `2026 Colombia earthquake` | “A 7.4-magnitude earthquake shakes western Colombia, leaving 2 dead” | 10:10–10:25, 10:15–10:30, 10:20–10:35 (re-emitted as edit_count grew from ~5 → ~12) | ~0.80–0.85 |

Kafka correctly retains every evaluation. For the dashboard’s “Anomalies
verified against real news” table we **dedupe to one row per
`page_title`, keeping the latest `window_end`**, so operators see the
current match rather than a stack of near-duplicate sliding-window
updates for the same story.

## 8. Limitations and future work

1. **Per-page baselines** — compare a window to that page’s own history rather
   than the global population (best next step for “quiet page suddenly busy”).
2. **Rate features** — bytes/minute, edits/minute to further stabilize against
   window-size choices.
3. **Snippet quality** — GDELT quadgrams are short and overlapping; fuller
   article text (if licensed/available) would improve embeddings further.
4. **Comment recall** — aggressive jargon/maintenance filtering reduces noise
   but can drop borderline-useful summaries.
5. **Sliding vs tumbling training windows** — training uses tumbling buckets;
   live Spark uses a 15-minute window with 5-minute slide. Close enough for
   magnitude calibration; exact slide replication is optional polish.

## 9. Files touched (implementation inventory)

| Area | Files |
|---|---|
| News snippets | `producer/gdelt_producer.py`, `tests/test_gdelt_producer.py` |
| Edit comments | `producer/wikipedia_producer.py`, `spark/wiki_stream.py`, `spark/ml_inference_stream.py` |
| Embedding text / filters | `vectordb/embeddings.py`, `tests/test_embeddings.py` |
| Match forwarding | `vectordb/match_events.py` |
| Windowed training | `models/historical_training.py`, `models/train_model.py`, `tests/test_historical_training.py` |
| Larger history | `scripts/download_wikimedia_history.py`, `data/historical/...jsonl` |
| Dashboard UX | `dashboard/app.py` |
| Docs | `README.md`, `docs/PROJECT_REPORT.md` |
