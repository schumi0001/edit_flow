"""Verify flagged Wikipedia edit anomalies against real-world GDELT news.

An anomalous edit spike on its own is not evidence of anything -- it could
be a real news event driving edits, or it could be vandalism, an edit war,
or bot activity. This script embeds both `wikipedia-anomalies` and
`news-topic` titles (see vectordb/embeddings.py) and, for each anomaly,
searches a rolling window of recent news-article embeddings for the closest
match. A cosine similarity >= SIMILARITY_THRESHOLD is treated as
confirmation that the anomaly corresponds to a real, concurrent news event.

Every evaluated anomaly (matched or not) is published to the
`verified-events` Kafka topic, not just the ones that pass the threshold --
that keeps the negative signal (an anomaly with no news match) available
too, e.g. for a downstream "% of anomalies confirmed as real" metric.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from kafka import KafkaConsumer, KafkaProducer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vectordb.embeddings import embed_text, gdelt_article_text, wikipedia_anomaly_text
from vectordb.news_index import NewsEmbeddingIndex, DEFAULT_RETENTION_HOURS

DEFAULT_KAFKA_SERVER = "localhost:9092"
DEFAULT_NEWS_TOPIC = "news-topic"
DEFAULT_ANOMALY_TOPIC = "wikipedia-anomalies"
DEFAULT_VERIFIED_TOPIC = "verified-events"
DEFAULT_SIMILARITY_THRESHOLD = 0.7

PUBLISH_TIMEOUT_SECONDS = 10


class EventVerifier:
    """Embeds incoming news articles and anomalies, and matches the two."""

    def __init__(
        self,
        kafka_producer: Any,
        output_topic: str = DEFAULT_VERIFIED_TOPIC,
        index: NewsEmbeddingIndex | None = None,
        embed_fn: Callable[[str], Any] = embed_text,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.kafka = kafka_producer
        self.output_topic = output_topic
        self.index = index if index is not None else NewsEmbeddingIndex()
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold

    def handle_news_article(self, article: dict) -> None:
        text = gdelt_article_text(article)
        if not text:
            return

        embedding = self.embed_fn(text)
        self.index.add(article, embedding)
        self.index.prune()

    def handle_anomaly(self, anomaly: dict) -> dict:
        """Evaluate one anomaly against the news index and publish the verdict."""
        text = wikipedia_anomaly_text(anomaly)
        similarity_score: float | None = None
        matched_article: dict | None = None

        if text:
            embedding = self.embed_fn(text)
            match = self.index.best_match(embedding)
            if match is not None:
                similarity_score, article = match
                matched_article = {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "language": article.get("language"),
                    "event_id": article.get("event_id"),
                }

        matched = similarity_score is not None and similarity_score >= self.similarity_threshold

        record = {
            "page_title": anomaly.get("page_title"),
            "window_start": anomaly.get("window_start"),
            "window_end": anomaly.get("window_end"),
            "anomaly_score": anomaly.get("anomaly_score"),
            "edit_count": anomaly.get("edit_count"),
            "recent_comments": anomaly.get("recent_comments"),
            "matched": matched,
            "similarity_score": similarity_score,
            "matched_article": matched_article if matched else None,
            "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        self.publish(record)
        self.print_verdict(record)
        return record

    def publish(self, record: dict) -> None:
        future = self.kafka.send(
            self.output_topic,
            key=record.get("page_title"),
            value=record,
        )
        future.get(timeout=PUBLISH_TIMEOUT_SECONDS)

    def print_verdict(self, record: dict) -> None:
        page_title = record["page_title"]
        if record["matched"]:
            article_title = (record["matched_article"] or {}).get("title", "?")
            print(
                f"MATCH ({record['similarity_score']:.2f}): '{page_title}' <-> "
                f"'{article_title}'"
            )
        elif record["similarity_score"] is not None:
            print(
                f"no match ({record['similarity_score']:.2f} < "
                f"{self.similarity_threshold:.2f}): '{page_title}'"
            )
        else:
            print(f"no match (news index empty): '{page_title}'")

    def close(self) -> None:
        self.kafka.flush()
        self.kafka.close()


def build_kafka_producer(kafka_server: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=kafka_server,
        key_serializer=lambda key: (key or "").encode("utf-8"),
        value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
        acks="all",
    )


def build_kafka_consumer(kafka_server: str, news_topic: str, anomaly_topic: str) -> KafkaConsumer:
    return KafkaConsumer(
        news_topic,
        anomaly_topic,
        bootstrap_servers=kafka_server,
        group_id="wikipulse-event-verifier",
        auto_offset_reset="latest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )


def run() -> None:
    kafka_server = os.environ.get("KAFKA_SERVER", DEFAULT_KAFKA_SERVER)
    news_topic = os.environ.get("GDELT_KAFKA_TOPIC", DEFAULT_NEWS_TOPIC)
    anomaly_topic = os.environ.get("ANOMALY_KAFKA_TOPIC", DEFAULT_ANOMALY_TOPIC)
    output_topic = os.environ.get("VERIFIED_KAFKA_TOPIC", DEFAULT_VERIFIED_TOPIC)
    similarity_threshold = float(
        os.environ.get("SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD)
    )
    retention_hours = float(
        os.environ.get("NEWS_RETENTION_HOURS", DEFAULT_RETENTION_HOURS)
    )

    verifier = EventVerifier(
        kafka_producer=build_kafka_producer(kafka_server),
        output_topic=output_topic,
        index=NewsEmbeddingIndex(retention=timedelta(hours=retention_hours)),
        similarity_threshold=similarity_threshold,
    )
    consumer = build_kafka_consumer(kafka_server, news_topic, anomaly_topic)

    print(
        f"Event verifier started. Watching '{news_topic}' and '{anomaly_topic}', "
        f"publishing verdicts to '{output_topic}' (threshold={similarity_threshold})."
    )
    print("Loading embedding model (first run may download it)...")
    embed_text("warm up")  # Force the model to load before the first message arrives.
    print("Embedding model ready.")

    try:
        for message in consumer:
            if message.topic == news_topic:
                verifier.handle_news_article(message.value)
            elif message.topic == anomaly_topic:
                verifier.handle_anomaly(message.value)
    except KeyboardInterrupt:
        print("\nStopping event verifier...")
    finally:
        consumer.close()
        verifier.close()
        print("Event verifier stopped.")


if __name__ == "__main__":
    run()
