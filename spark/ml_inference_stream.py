import os
import sys
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count, approx_count_distinct, sum, abs, when, pandas_udf
from pyspark.sql.types import (
    BooleanType, LongType, StringType, StructField, StructType, TimestampType, DoubleType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_utils import load_model

KAFKA_SERVER = "localhost:9092"
KAFKA_INPUT_TOPIC = "wikipedia-edits"
KAFKA_OUTPUT_TOPIC = "wikipedia-anomalies"
MODEL_PATH = "models/anomaly_detector.joblib"

# 1. Load the pre-trained model so it is cached in memory
trained_model = load_model(MODEL_PATH)

spark = (
    SparkSession.builder
    .appName("WikiPulseMLInferenceStream")
    .master("local[2]")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 2. Corrected Vectorized Pandas UDF with all 8 engineered features
@pandas_udf(DoubleType())
def predict_anomaly_udf(edit_count: pd.Series, unique_editors: pd.Series, total_byte_changes: pd.Series, bot_ratio: pd.Series, minor_edit_ratio: pd.Series) -> pd.Series:
    # Pack base aggregations into a Pandas DataFrame
    features_df = pd.DataFrame({
        "edit_count": edit_count,
        "unique_editors": unique_editors,
        "total_byte_changes": total_byte_changes,
        "bot_ratio": bot_ratio,
        "minor_edit_ratio": minor_edit_ratio
    }).fillna(0.0)
    
    # Re-engineer the exact mathematical features used during training
    features_df["relative_growth"] = features_df["total_byte_changes"] / features_df["edit_count"]
    features_df["human_bot_friction"] = features_df["minor_edit_ratio"] - features_df["bot_ratio"]
    features_df["editor_concentration"] = features_df["unique_editors"] / features_df["edit_count"]
    
    # Fill any divide-by-zero NaNs that could result from empty fields
    features_df = features_df.fillna(0.0)
    
    # Enforce strict column order matching the model training expectations
    feature_cols = [
        "edit_count", "unique_editors", "total_byte_changes", "bot_ratio", 
        "minor_edit_ratio", "relative_growth", "human_bot_friction", "editor_concentration"
    ]
    X = features_df[feature_cols]
    
    # Generate continuous anomaly scores
    anomaly_scores = trained_model.decision_function(X)
    return pd.Series(anomaly_scores)

# 3. Standard Ingestion and Cleaning Layout
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
    .option("kafka.group.id", "wikipulse-sklearn-demo")
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

cleaned_events = (
    kafka_stream
    .select(from_json(col("value").cast("string"), event_schema).alias("event"))
    .select("event.*")
    .filter(col("event_id").isNotNull())
    .withColumn("event_datetime", col("timestamp").cast(TimestampType()))
)

# 4. Compute Dynamic Aggregations
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
)

# 5. Apply the ML Model to generate live scores
scored_stream = features_aggregated.withColumn(
    "anomaly_score",
    predict_anomaly_udf(
        col("edit_count"),
        col("unique_editors"),
        col("total_byte_changes"),
        col("bot_ratio"),
        col("minor_edit_ratio")
    )
)

# 6. Filter for True Anomalies (Scores strictly less than 0)
final_alerts = (
    scored_stream
    .filter(col("anomaly_score") < 0.0)
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

# 7. Sink: Broadcast alerts back into Kafka
query = (
    final_alerts
    .selectExpr("CAST(page_title AS STRING) AS key", "to_json(struct(*)) AS value")
    .writeStream
    .outputMode("update")
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("topic", KAFKA_OUTPUT_TOPIC)
    .option("checkpointLocation", "checkpoints/ml_inference")
    .start()
)

try:
    query.awaitTermination()
except KeyboardInterrupt:
    pass
finally:
    query.stop()
    spark.stop()