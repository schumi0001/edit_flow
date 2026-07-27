from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    BooleanType,
    LongType,
    StringType,
    StructField,
    StructType,
)


KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "wikipedia-edits"


spark = (
    SparkSession.builder
    .appName("WikiPulseStream")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


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
    .option("startingOffsets", "latest")
    .load()
)


events = (
    kafka_stream
    .select(
        from_json(
            col("value").cast("string"),
            event_schema
        ).alias("event"),
        col("partition"),
        col("offset"),
        col("timestamp").alias("kafka_timestamp"),
    )
    .select(
        "event.*",
        "partition",
        "offset",
        "kafka_timestamp",
    )
    .filter(col("event_id").isNotNull())
)


query = (
    events.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", "data/raw_wikipedia_edits")
    .option(
        "checkpointLocation",
        "checkpoints/raw_wikipedia_edits"
    )
    .trigger(processingTime="10 seconds")
    .start()
)


try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nStopping WikiPulse Spark stream...")
finally:
    query.stop()
    spark.stop()
    print("WikiPulse Spark stream stopped.")
