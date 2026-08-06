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