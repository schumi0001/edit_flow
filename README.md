# WikiPulse

Streaming Wikipedia edit-anomaly detection with GDELT news verification
(Kafka + Spark Structured Streaming + Isolation Forest + Qdrant embeddings).

For a report-oriented write-up of the embedding enrichment, windowed training
fix, and validation analysis, see [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

## Run

Requirements: Python 3.11+ (3.12 also fine), Java 17, and Docker Desktop.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd wikipulse

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m streamlit run dashboard/app.py
```

Open http://localhost:8501 and click Start Pipeline.

To stop, click Stop Pipeline, then press Control + C in the terminal.

## News ingestion (GDELT)

The GDELT producer polls GDELT's public Web NGrams "table of contents" files
(no API key required) and publishes discovered news articles to the
`news-topic` Kafka topic, for downstream embedding and similarity search
against Wikipedia anomalies.

Kafka must be running before you start the producer (or before you try to
read from `news-topic`):

```bash
docker compose up -d kafka
python producer/gdelt_producer.py
```

Optional configuration (defaults shown):

```bash
export KAFKA_SERVER="localhost:9092"
export GDELT_KAFKA_TOPIC="news-topic"
export GDELT_POLL_INTERVAL_SECONDS="60"
export GDELT_LOOKBACK_MINUTES="45"
export GDELT_SAFETY_DELAY_MINUTES="5"
export GDELT_STATE_FILE=".runtime/gdelt_producer_state.json"
export GDELT_LANGUAGES=""      # e.g. "en" or "en,es" — empty means no language filter
```

`GDELT_LANGUAGES` keeps only articles whose GDELT-reported language code
(e.g. `en`, `es`, `zh`) is in this comma-separated list. This is a scope
filter, not a relevance filter: it exists so articles line up with whatever
Wikipedia language and embedding model the downstream similarity search
uses, not to guess which articles are topically related to a given
anomaly. Deciding relevance is intentionally left to the embedding /
cosine-similarity stage, which can catch semantic matches (e.g. "wildfire"
vs. "blaze") that a keyword filter at ingestion time would miss.

The producer polls a rolling window of recent minute-resolution timestamps.
GDELT only publishes files during periodic heartbeat windows (a handful of
consecutive minutes, then a gap until the next quarter-hour), so most
timestamps return HTTP 404 and are simply skipped; already-processed
timestamps are persisted to `GDELT_STATE_FILE` so a restart does not
republish the same articles. Stop it with Control + C.

Because of those heartbeat windows, don't be alarmed if nothing shows up for
several minutes right after you first start the producer -- it may simply be
waiting for the next window. Once it publishes its first batch, new articles
typically keep arriving every 1-15 minutes.

Each published event's `gdelt_seen_at` field is when GDELT's crawler
observed/re-crawled the page during that TOC batch -- every article in a
single poll's TOC file shares the exact same `gdelt_seen_at` value. It is
**not** the article's original publication date; GDELT does not reliably
expose that. A story can legitimately show a `gdelt_seen_at` of "just now"
while its own URL or byline indicates it was originally published days or
weeks earlier. Use `observed_at` (when this producer processed the record)
for pipeline-latency purposes, and treat `gdelt_seen_at` only as "GDELT's
crawler touched this URL around this time," not as a publish date.

### Reading news articles

The easiest way to see what the GDELT producer is publishing is
`view_news.py`, which tails `news-topic` and pretty-prints each article as a
readable card (language, timestamp, title, URL) instead of raw JSON:

```bash
python view_news.py
```

By default this waits up to 90 seconds for *new* articles only
(`--offset latest`) -- keep `producer/gdelt_producer.py` running in another
terminal so there's something to see. To replay everything still retained on
the topic instead (useful right after starting the producer, before it's had
time to publish anything new), use `--offset earliest`:

```bash
python view_news.py --offset earliest
```

Other flags:

```bash
python view_news.py --limit 50        # show up to 50 articles (default: 25)
python view_news.py --timeout 180     # wait longer before giving up (default: 90s)
```

If you just want the raw JSON (e.g. for debugging or piping into another
tool), use a plain consumer instead:

```bash
python -c "
from kafka import KafkaConsumer
c = KafkaConsumer(
    'news-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda value: value.decode('utf-8'),
)
for message in c:
    print(message.value)
"
```

#### Event schema

Each message on `news-topic` is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `event_id` | Stable SHA-256 hash of the article URL; used as the Kafka key and for dedup |
| `title` | Article headline |
| `snippet` | A handful of real quadgrams (4-word phrases) pulled from the article's body text via GDELT's companion `.ngrams.txt.gz` file for this batch, cross-referenced by DOCID -- `""` if that file was missing or had nothing for this article. Used alongside `title` for embedding (see `vectordb/embeddings.py`); not full article text, only short possibly-overlapping fragments. |
| `url` | Article URL |
| `language` | GDELT-reported language code (e.g. `en`, `es`, `zh`) |
| `image_url` | Article thumbnail image, if GDELT reported one (else `null`) |
| `gdelt_seen_at` | When GDELT's crawler observed/re-crawled the page during this poll -- **not** the article's original publish date. Every article in one poll's batch shares the same value; see the note above. |
| `observed_at` | When *this producer* processed the record -- use this for pipeline-latency purposes |
| `toc_id` | GDELT's internal ID for the record within its source TOC file |
| `toc_timestamp` | The minute-resolution TOC file timestamp this record came from (`YYYYMMDDHHMM00`) |
| `source` | Always `"gdelt_web_ngrams"` -- identifies which producer emitted the event |
| `event_type` | Always `"article"` |

**Anomaly Detection**
Run the following commands from `/workspaces/edit_flow`.

**1. Start Kafka**
```bash
docker compose up -d
```

**2. Train the offline scikit-learn model** (one-time, or whenever you want to refresh the baseline)

First, fetch real historical edits from the MediaWiki `recentchanges` API (one-time, or to refresh):
```bash
python scripts/download_wikimedia_history.py
```
This writes `data/historical/en.wikipedia.org.recentchanges.jsonl` and trains from it immediately. By default it pages through up to 150,000 edits (roughly 5-10 minutes, a few hundred paginated requests) to get much closer to the full ~30-day window MediaWiki's `recentchanges` table retains, rather than stopping well short of it. Pass `--limit` to fetch fewer (faster) or more.

To train again later without re-downloading:
```bash
python models/train_model.py
```
This defaults to `data/historical/en.wikipedia.org.recentchanges.jsonl`. Pass `--history-jsonl /path/to/other.jsonl` to train from a different file.

There is also a `--feature-lake` flag that trains from whatever has accumulated in the local `data/lake/features` parquet files (populated by `spark/ml_feature_stream.py` from the live `wikipedia-edits` Kafka topic). This is **not recommended** as a default: that topic can accumulate test/synthetic events published during development, and there's no guarantee the live window covers a representative slice of normal activity.

This trains an Isolation Forest pipeline offline and saves it to `models/anomaly_detector.joblib`.

Training features are computed per (page, 15-minute window) rather than per page over the whole historical file, since that's the same granularity `spark/ml_inference_stream.py` scores live (see step 3). Aggregating lifetime totals instead would teach the model that "normal" edit volume looks like weeks' worth of activity, which makes almost any real-time burst look anomalous purely from the mismatch in time scale, not because it's actually unusual. Re-running `download_wikimedia_history.py` or `train_model.py` after this change requires restarting `spark/ml_inference_stream.py` (it loads the model into memory once at startup and won't pick up a newer file on disk).

**3. Start the offline scikit-learn scorer** — start this *before* step 4, since it reads Kafka from `latest` and will not see edits published before it starts.

If your shell’s default `python` is a different minor version than the project
venv (common when Homebrew Python 3.13 coexists with a 3.11 venv), point Spark
workers at the venv first:

```bash
export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"
export PYSPARK_DRIVER_PYTHON="$(pwd)/.venv/bin/python"
python spark/ml_inference_stream.py
```

Leave this running. It reads `wikipedia-edits` from Kafka, aggregates events into 15-minute windows, scores each page with the saved scikit-learn model, and writes flagged anomalies to the `wikipedia-anomalies` topic:

| Field | Meaning |
|---|---|
| `page_title` | The flagged Wikipedia page |
| `window_start` / `window_end` | The anomaly's 15-minute detection window |
| `edit_count` / `unique_editors` / `total_byte_changes` | Aggregated edit-activity features for this page and window |
| `anomaly_score` | Isolation Forest decision-function score (negative = flagged as anomalous) |
| `recent_comments` | Up to 5 real, substantive edit summaries (`comment` from the Wikimedia stream) sampled from this window, joined with `" \| "`. Blank, editing-jargon-only (e.g. `"ce"`, `"rv"`), and generic maintenance/boilerplate-only summaries (e.g. `"created article"`, `"/* See also */"`) are filtered out first; MediaWiki markup (wikilinks, templates, auto-generated revert/rollback prefixes) is stripped from the ones that remain, so this carries clean topical text rather than raw wiki syntax (see `vectordb/embeddings.substantive_comment_text`). `""` if none of this window's comments were substantive. Used alongside `page_title` for embedding (see `vectordb/embeddings.py`). |

**4. Feed it live data** — this is the step that actually publishes to the `wikipedia-edits` Kafka topic that step 3 reads from:
```bash
python producer/wikipedia_producer.py
```
Optionally, also run `python spark/wiki_stream.py` alongside it. That script only *reads* `wikipedia-edits` and archives it to Parquet under `data/raw_wikipedia_edits` -- it is not required for anomaly detection and does not feed anything itself, despite the similar name.

**5. Check for flagged anomalies**
```bash
python -c "
from kafka import KafkaConsumer
c = KafkaConsumer('wikipedia-anomalies', bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=8000)
for msg in c:
    print(msg.value.decode())
"
```
The `wikipedia-anomalies` Kafka topic is the pipeline's actual output — there is no separate results file.

**6. Dashboard UI**
```bash
python -m streamlit run dashboard/app.py
```

## Verifying anomalies against real news (vector search)

An anomalous edit spike isn't by itself evidence of anything — it could be
a real news event driving edits, or vandalism/an edit war/bot activity.
`vectordb/match_events.py` embeds titles from both `wikipedia-anomalies`
and `news-topic` (using a local `sentence-transformers` model — no API key),
storing news embeddings in [Qdrant](https://qdrant.tech/), and for each
anomaly searches that index for the closest recent match. A cosine
similarity of at least `SIMILARITY_THRESHOLD` (default `0.7`) is treated as
confirmation that the anomaly corresponds to a real, concurrent news event.

Both sides also carry a bit of real body text beyond the bare title —
`news-topic`'s `snippet` (real quadgrams from the article body) and
`wikipedia-anomalies`' `recent_comments` (real edit summaries) — see the
schema tables above and `vectordb/embeddings.py`. Even so, this is still
largely title-driven semantic matching — e.g. Wikipedia page
`2026_California_wildfires` vs. a GDELT article titled "Wildfires force
evacuations in LA County".

Start Qdrant (it's part of the same Compose file as Kafka):
```bash
docker compose up -d qdrant
```

Then run the verifier alongside `producer/gdelt_producer.py` and the
anomaly-detection pipeline above (steps 1–4):
```bash
python vectordb/match_events.py
```
The first run downloads the embedding model (~80MB, needs network); it's
cached locally after that. Qdrant's own data is persisted in a Docker
volume, so the news index survives restarts of `match_events.py` itself.

Every evaluated anomaly — matched or not — is published to the
`verified-events` Kafka topic, keyed by `page_title`:

| Field | Meaning |
|---|---|
| `page_title` | The flagged Wikipedia page |
| `window_start` / `window_end` | The anomaly's 15-minute detection window |
| `anomaly_score` | Score from `wikipedia-anomalies` |
| `edit_count` | Edit count from `wikipedia-anomalies` |
| `unique_editors` | Unique editor count from `wikipedia-anomalies` |
| `total_byte_changes` | Total absolute byte change from `wikipedia-anomalies` |
| `matched` | `true` if `similarity_score >= SIMILARITY_THRESHOLD` |
| `similarity_score` | Best cosine similarity found against recent news, or `null` if the news index was empty |
| `recent_comments` | Forwarded from `wikipedia-anomalies` for dashboard display / debugging |
| `similarity_threshold` | The `SIMILARITY_THRESHOLD` this verdict was computed with |
| `matched_article` | `{title, url, language, event_id, gdelt_seen_at, observed_at}` of the best-matching article — present only when `matched` is `true`, else `null` |
| `closest_article` | Same shape as `matched_article`, but always present when the news index had any candidate — even below the threshold — so unverified anomalies stay explainable |
| `evaluated_at` | When this script evaluated the anomaly |

Optional configuration (defaults shown):
```bash
export KAFKA_SERVER="localhost:9092"
export GDELT_KAFKA_TOPIC="news-topic"
export ANOMALY_KAFKA_TOPIC="wikipedia-anomalies"
export VERIFIED_KAFKA_TOPIC="verified-events"
export SIMILARITY_THRESHOLD="0.7"
export NEWS_RETENTION_HOURS="24"   # how far back to keep news embeddings for matching
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export QDRANT_URL="http://localhost:6333"
```

To check the output:
```bash
python -c "
from kafka import KafkaConsumer
c = KafkaConsumer('verified-events', bootstrap_servers='localhost:9092', auto_offset_reset='earliest', consumer_timeout_ms=8000)
for msg in c:
    print(msg.value.decode())
"
```