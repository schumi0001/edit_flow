import os
import sys
import json
import time
import random

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, approx_count_distinct, sum, abs, when
from pyspark.sql.types import StringType

OUTPUT_DIR = "data/lake/features_clean_test"

spark = (
    SparkSession.builder
    .appName("WikiPulseCleanTestFeatures")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Same mock payloads as tests/test_injector.py, generated directly instead of via Kafka.
random.seed(42)
current_time = int(time.time())
mock_records = []
PAGES = ["Python_(programming_language)", "Data_engineering", "Coffee", "Cat", "Earth"]
SPIKE_PAGE = "Breaking_News_Event_2026"

for i in range(100):
    mock_records.append(json.dumps({
        "event_id": f"normal_{i}_{random.randint(1000,9999)}",
        "page_title": random.choice(PAGES),
        "user": f"Editor_{random.randint(1,10)}",
        "bot": random.choice([True, False, False, False]),
        "minor": random.choice([True, False]),
        "byte_change": random.randint(0, 100),
    }))

for i in range(150):
    mock_records.append(json.dumps({
        "event_id": f"spike_{i}",
        "page_title": SPIKE_PAGE,
        "user": f"BreakingEditor_{i}",
        "bot": False,
        "minor": False,
        "byte_change": random.randint(500, 2000),
    }))

raw = spark.createDataFrame(mock_records, StringType()).toDF("value")

from pyspark.sql.functions import from_json
from pyspark.sql.types import BooleanType, StructField, StructType, LongType

event_schema = StructType([
    StructField("event_id", StringType()),
    StructField("page_title", StringType()),
    StructField("user", StringType()),
    StructField("bot", BooleanType()),
    StructField("minor", BooleanType()),
    StructField("byte_change", LongType()),
])

features = (
    raw
    .select(from_json(col("value"), event_schema).alias("event"))
    .select("event.*")
    .groupBy("page_title")
    .agg(
        count("event_id").alias("edit_count"),
        approx_count_distinct("user").alias("unique_editors"),
        sum(abs(col("byte_change"))).alias("total_byte_changes"),
        (sum(when(col("bot") == True, 1).otherwise(0)) / count("event_id")).alias("bot_ratio"),
        (sum(when(col("minor") == True, 1).otherwise(0)) / count("event_id")).alias("minor_edit_ratio")
    )
    .withColumn("relative_growth", col("total_byte_changes") / col("edit_count"))
    .withColumn("human_bot_friction", col("minor_edit_ratio") - col("bot_ratio"))
    .withColumn("editor_concentration", col("unique_editors") / col("edit_count"))
)

features.write.mode("overwrite").parquet(OUTPUT_DIR)
print(f"Wrote {features.count()} rows to {OUTPUT_DIR}")
features.show(truncate=False)

spark.stop()
