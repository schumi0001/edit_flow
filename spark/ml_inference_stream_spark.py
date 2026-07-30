import os
import sys
import json

# Ensure Spark's Python workers use this same interpreter (with numpy/pyspark installed)
# instead of falling back to the bare system python3 on PATH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count, approx_count_distinct, sum, abs, when, udf
from pyspark.sql.types import (
    BooleanType, LongType, StringType, StructField, StructType, TimestampType, DoubleType,
)
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors

KAFKA_SERVER = "localhost:9092"
KAFKA_INPUT_TOPIC = "wikipedia-edits"
KAFKA_OUTPUT_TOPIC = "wikipedia-anomalies"
MODEL_PATH = "models/anomaly_detector_spark"
THRESHOLD_PATH = "models/anomaly_detector_spark_threshold.json"

# 1. Load the pre-trained spark.ml pipeline and its calibrated distance threshold
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model_spark.py first.")
if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(f"Anomaly threshold not found at {THRESHOLD_PATH}. Run train_model_spark.py first.")

with open(THRESHOLD_PATH) as f:
    _threshold_meta = json.load(f)
    ANOMALY_THRESHOLD = _threshold_meta["threshold"]
    NORMAL_CENTER = Vectors.dense(_threshold_meta["normal_center"])

spark = (
    SparkSession.builder
    .appName("WikiPulseMLInferenceStreamSpark")
    .master("local[2]")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

pipeline_model = PipelineModel.load(MODEL_PATH)


# Distance to the training-time "normal" cluster centroid, not to whichever cluster a
# row happens to be assigned to -- see train_model_spark.py for why that distinction
# matters (an assigned-cluster distance can never flag the point that defines it).
@udf(DoubleType())
def distance_to_normal_udf(features):
    return float(features.squared_distance(NORMAL_CENTER) ** 0.5)


# 2. Standard Ingestion and Cleaning Layout
event_schema = StructType([
    StructField("event_id", StringType()),
    StructField("recent_change_id", LongType()),
    StructField("revision_id", LongType()),
    StructField("timestamp", LongType()),
    StructField("page_title", StringType()),
    StructField("namespace", LongType()),
    StructField("user", StringType()),
    StructField("bot", BooleanType()),
    StructField("minor", BooleanType()),
    StructField("old_length", LongType()),
    StructField("new_length", LongType()),
    StructField("byte_change", LongType()),
    StructField("server_name", StringType()),
    StructField("event_type", StringType()),
])

kafka_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", KAFKA_INPUT_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

cleaned_events = (
    kafka_stream
    .select(from_json(col("value").cast("string"), event_schema).alias("event"))
    .select("event.*")
    .filter(col("event_id").isNotNull())
    .withColumn("event_datetime", col("timestamp").cast(TimestampType()))
)

# 3. Compute dynamic aggregations, then re-derive the same engineered features used at training time
features_aggregated = (
    cleaned_events
    .withWatermark("event_datetime", "10 minutes")
    .groupBy(
        window(col("event_datetime"), "15 minutes", "5 minutes"),
        col("page_title")
    )
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

# 4. Apply the spark.ml pipeline (assemble -> scale) and score by distance to "normal"
scored_stream = pipeline_model.transform(features_aggregated).withColumn(
    "anomaly_score", distance_to_normal_udf(col("scaled_features"))
)

# 5. Filter for true anomalies: distance at or above the training-time threshold
final_alerts = (
    scored_stream
    .filter(col("anomaly_score") >= ANOMALY_THRESHOLD)
    .select(
        col("page_title"),
        col("window.start").cast("string").alias("window_start"),
        col("window.end").cast("string").alias("window_end"),
        col("edit_count"),
        col("unique_editors"),
        col("total_byte_changes"),
        col("anomaly_score")
    )
)

# 6. Sink: broadcast alerts back into Kafka
query = (
    final_alerts
    .selectExpr("CAST(page_title AS STRING) AS key", "to_json(struct(*)) AS value")
    .writeStream
    .outputMode("update")
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("topic", KAFKA_OUTPUT_TOPIC)
    .option("checkpointLocation", "checkpoints/ml_inference_spark")
    .start()
)

try:
    query.awaitTermination()
except KeyboardInterrupt:
    pass
finally:
    query.stop()
    spark.stop()
