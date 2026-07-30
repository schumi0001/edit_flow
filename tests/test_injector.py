import time
import json
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

# Initialize a lightweight local Spark session to handle the data injection
spark = (
    SparkSession.builder
    .appName("WikiPulseTestInjector")
    .master("local[1]")  # Only needs a single local thread to publish test batches
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .getOrCreate()
)

# 1. Construct the list of mock payloads in standard Python
current_time = int(time.time())
mock_records = []
PAGES = ["Python_(programming_language)", "Data_engineering", "Coffee", "Cat", "Earth"]
SPIKE_PAGE = "Breaking_News_Event_2026"

# Generate 100 normal background edits spread across pages
for i in range(100):
    mock_records.append(json.dumps({
        "event_id": f"normal_{i}_{random.randint(1000,9999)}",
        "recent_change_id": 100000 + i,
        "revision_id": 200000 + i,
        "timestamp": current_time,
        "page_title": random.choice(PAGES),
        "namespace": 0,
        "user": f"Editor_{random.randint(1,10)}",
        "bot": random.choice([True, False, False, False]),
        "minor": random.choice([True, False]),
        "old_length": 5000,
        "new_length": 5000 + random.randint(-50, 100),
        "byte_change": random.randint(0, 100),
        "server_name": "en.wikipedia.org",
        "event_type": "edit"
    }))

# Generate 150 immediate, dense edits to simulate a breaking news event anomaly
for i in range(150):
    mock_records.append(json.dumps({
        "event_id": f"spike_{i}",
        "recent_change_id": 300000 + i,
        "revision_id": 400000 + i,
        "timestamp": current_time,
        "page_title": SPIKE_PAGE,
        "namespace": 0,
        "user": f"BreakingEditor_{i}",
        "bot": False,
        "minor": False,
        "old_length": 1200,
        "new_length": 1200 + random.randint(500, 2000),
        "byte_change": random.randint(500, 2000),
        "server_name": "en.wikipedia.org",
        "event_type": "edit"
    }))

# 2. Turn the Python list into a Spark DataFrame
# Kafka requires a string or binary column explicitly named "value"
df = spark.createDataFrame(mock_records, StringType()).toDF("value")

# 3. Publish everything to Kafka using Spark's robust network layer
# This works natively on 'localhost:9092' across Macs, Windows, and Linux Dev Containers
(
    df.write
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("topic", "wikipedia-edits")
    .save()
)

spark.stop()
