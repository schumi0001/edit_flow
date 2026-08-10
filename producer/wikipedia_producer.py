import json
import time

import requests
from kafka import KafkaProducer
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    RequestException,
    Timeout,
)
from sseclient import SSEClient


STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
KAFKA_TOPIC = "wikipedia-edits"
KAFKA_SERVER = "localhost:9092"

INITIAL_RECONNECT_DELAY = 2
MAX_RECONNECT_DELAY = 60

HEADERS = {
    "Accept": "text/event-stream",
    "User-Agent": "WikiPulse/1.0 (student big-data project)",
}


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    key_serializer=lambda key: key.encode("utf-8"),
    value_serializer=lambda value: json.dumps(
        value,
        ensure_ascii=False,
    ).encode("utf-8"),
    acks="all",
)


def process_event(event):
    """Parse, filter, clean, and send one Wikimedia event to Kafka."""

    if not event.data:
        return

    try:
        data = json.loads(event.data)
    except json.JSONDecodeError:
        return

    # Keep only English Wikipedia article edits and new articles.
    if not (
        data.get("server_name") == "en.wikipedia.org"
        and data.get("type") in ("edit", "new")
        and data.get("namespace") == 0
    ):
        return

    length = data.get("length") or {}
    revision = data.get("revision") or {}
    meta = data.get("meta") or {}

    old_length = length.get("old") or 0
    new_length = length.get("new") or 0

    clean_event = {
        "event_id": meta.get("id"),
        "recent_change_id": data.get("id"),
        "revision_id": revision.get("new"),
        "timestamp": data.get("timestamp"),
        "page_title": data.get("title"),
        "namespace": data.get("namespace"),
        "user": data.get("user"),
        "bot": data.get("bot", False),
        "minor": data.get("minor", False),
        "old_length": old_length,
        "new_length": new_length,
        "byte_change": new_length - old_length,
        # The editor's edit summary (rc_comment) -- real, if often terse,
        # text describing what changed. Sampled downstream (see
        # spark/ml_inference_stream.py) into anomaly records for embedding.
        "comment": data.get("comment") or "",
        "server_name": data.get("server_name"),
        "event_type": data.get("type"),
    }

    producer.send(
        KAFKA_TOPIC,
        key=clean_event["page_title"],
        value=clean_event,
    )

    print(
        f"Sent: {clean_event['page_title']}"
        f"({clean_event['byte_change']:+d} bytes)"
    )


def run():
    reconnect_delay = INITIAL_RECONNECT_DELAY

    print("WikiPulse producer started. Sending events to Kafka...")

    try:
        while True:
            response = None

            try:
                print("Connecting to Wikimedia stream...")

                response = requests.get(
                    STREAM_URL,
                    headers=HEADERS,
                    stream=True,
                    timeout=(10, 90),
                )
                response.raise_for_status()

                client = SSEClient(response)

                print("Connected to Wikimedia stream.")
                reconnect_delay = INITIAL_RECONNECT_DELAY

                for event in client.events():
                    process_event(event)

                # Reconnect if the event iterator ends without an exception.
                print(
                    "Wikimedia stream ended. "
                    f"Reconnecting in {reconnect_delay} seconds..."
                )

            except (
                ChunkedEncodingError,
                ConnectionError,
                Timeout,
            ) as error:
                print(
                    f"Wikimedia stream disconnected: {error}\n"
                    f"Reconnecting in {reconnect_delay} seconds..."
                )

            except RequestException as error:
                print(
                    f"Wikimedia request failed: {error}\n"
                    f"Reconnecting in {reconnect_delay} seconds..."
                )

            except Exception as error:
                print(
                    f"Unexpected stream error: {error}\n"
                    f"Reconnecting in {reconnect_delay} seconds..."
                )

            finally:
                if response is not None:
                    response.close()

            time.sleep(reconnect_delay)
            reconnect_delay = min(
                reconnect_delay * 2,
                MAX_RECONNECT_DELAY,
            )

    except KeyboardInterrupt:
        print("\nStopping WikiPulse producer...")

    finally:
        producer.flush()
        producer.close()
        print("WikiPulse producer stopped.")


if __name__ == "__main__":
    run()