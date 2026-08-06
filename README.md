# WikiPulse

## Run

Requirements: Python 3.12, Java, and Docker Desktop.

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

**2. Train the offline scikit-learn model** (one-time, or whenever the feature lake changes)

From the existing local feature lake:
```bash
python models/train_model.py
```

From a historical Wikimedia-style JSONL file (useful when you do not want to leave the live stream running):
```bash
python models/train_model.py --history-jsonl /path/to/history.jsonl
```

This trains an Isolation Forest pipeline offline and saves it to `models/anomaly_detector.joblib`.

**3. Start the offline scikit-learn scorer** — start this *before* step 4, since it reads Kafka from `latest` and will not see edits published before it starts:
```bash
python spark/ml_inference_stream.py
```
Leave this running. It reads `wikipedia-edits` from Kafka, aggregates events into 15-minute windows, scores each page with the saved scikit-learn model, and writes flagged anomalies to the `wikipedia-anomalies` topic.

**4. Feed it live data**
```bash
python spark/wiki_stream.py
```

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