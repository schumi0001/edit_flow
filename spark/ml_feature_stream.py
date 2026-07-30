import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, count, approx_count_distinct, sum, abs, when
from pyspark.sql.types import BooleanType, LongType, StringType, StructField, StructType, TimestampType

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "wikipedia-edits"
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"

spark = (
    SparkSession.builder
    .appName("WikiPulseMLFeatureStream")
    .master("local[2]")
    .config("spark.jars.packages", KAFKA_PACKAGE)
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

os.makedirs("data/lake/features", exist_ok=True)
os.makedirs("checkpoints/lake_features", exist_ok=True)

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
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)

def write_features(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    (
        batch_df
        .select(from_json(col("value").cast("string"), event_schema).alias("event"))
        .select("event.*")
        .filter(col("event_id").isNotNull())
        .withColumn("event_datetime", col("timestamp").cast(TimestampType()))
        .groupBy("page_title")
        .agg(
            count("event_id").alias("edit_count"),
            approx_count_distinct("user").alias("unique_editors"),
            sum(abs(col("byte_change"))).alias("total_byte_changes"),
            (sum(when(col("bot") == True, 1).otherwise(0)) / count("event_id")).alias("bot_ratio"),
            (sum(when(col("minor") == True, 1).otherwise(0)) / count("event_id")).alias("minor_edit_ratio")
        )
        #Isolation Forest Feature Engineering
        .withColumn(
            "relative_growth", 
            col("total_byte_changes") / col("edit_count")
        )
        .withColumn(
            "human_bot_friction", 
            col("minor_edit_ratio") - col("bot_ratio")
        )
        .withColumn(
            "editor_concentration", 
            col("unique_editors") / col("edit_count")
        )
        .write
        .mode("append")
        .parquet("data/lake/features")
    )

query = (
    kafka_stream.writeStream
    .foreachBatch(write_features)
    .option("checkpointLocation", "checkpoints/lake_features")
    .trigger(once=True)
    .start()
)

try:
    query.awaitTermination()
except KeyboardInterrupt:
    pass
finally:
    query.stop()
    spark.stop()
