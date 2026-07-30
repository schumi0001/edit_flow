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
Run the following commands from `/workspaces/edit_flow` 

**1. Start Kafka**
```bash
docker compose up -d
```

**2. Train the model** (one-time, or whenever `data/lake/features` changes)
```bash
python models/train_model_spark.py
```
Saves the pipeline to `models/anomaly_detector_spark/` and its calibrated threshold to `models/anomaly_detector_spark_threshold.json`.

**3. Start the streaming scorer** — must be running *before* new data arrives, since it reads Kafka from `latest`:
```bash
python spark/ml_inference_stream_spark.py
```
Leave this running. It reads `wikipedia-edits`, aggregates into 15-minute windows, scores each page, and writes flagged anomalies to the `wikipedia-anomalies` topic.

**4. Feed it data**
a. synthetic test scenario (5 normal pages + 1 injected spike page)
```bash
python tests/test_injector.py
```

b. live Wikimedia edit feed
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