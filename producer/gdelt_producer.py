"""Stream GDELT Web NGrams article metadata into Kafka.

GDELT publishes a gzipped JSON-lines "table of contents" (TOC) file most
minutes at:

    https://storage.googleapis.com/data.gdeltproject.org/gdeltv5/weblegacy/ngrams/YYYYMMDDHHMM00.toc.json.gz

Files only appear during GDELT's 15-minute heartbeat windows (a handful of
consecutive minutes, then a gap until the next quarter-hour), so most
timestamps legitimately return HTTP 404. This producer polls a rolling
window of recent minute timestamps, skips ones that are not yet published,
and republishes each discovered article as a normalized JSON event on a
Kafka topic.

Deciding whether an article is *relevant* to a given Wikipedia anomaly is
intentionally out of scope here. That judgment belongs to the downstream
embedding / cosine-similarity stage, which can match semantically related
articles that a lexical keyword filter would miss. The only filtering
performed at this layer is an optional language restriction (GDELT_LANGUAGES),
which exists to keep articles in scope for whatever embedding model and
Wikipedia-language pairing downstream consumers use -- not to judge topical
relevance.
"""

import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError
from requests.exceptions import RequestException


BASE_URL = (
    "https://storage.googleapis.com/data.gdeltproject.org/"
    "gdeltv5/weblegacy/ngrams/"
)
TIMESTAMP_FORMAT = "%Y%m%d%H%M00"

DEFAULT_KAFKA_SERVER = "localhost:9092"
DEFAULT_KAFKA_TOPIC = "news-topic"
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_LOOKBACK_MINUTES = 45
DEFAULT_SAFETY_DELAY_MINUTES = 5
DEFAULT_STATE_FILE = ".runtime/gdelt_producer_state.json"
DEFAULT_LANGUAGES = ""

REQUEST_TIMEOUT_SECONDS = 30
MAX_FETCH_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 30
MAX_PUBLISH_ATTEMPTS = 3
PUBLISH_TIMEOUT_SECONDS = 10
RECENT_URL_CAPACITY = 20_000
COMPLETED_TIMESTAMP_RETENTION_MINUTES = 180

HEADERS = {
    "User-Agent": "WikiPulse/1.0 (student big-data project)",
}


def candidate_timestamps(
    now_utc: datetime,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    safety_delay_minutes: int = DEFAULT_SAFETY_DELAY_MINUTES,
) -> Iterator[str]:
    """Yield minute-resolution TOC timestamps, oldest first.

    GDELT recommends staying a few minutes behind "now" since the most
    recent minute may not have finished publishing yet.
    """
    end = now_utc.replace(second=0, microsecond=0) - timedelta(
        minutes=safety_delay_minutes
    )
    start = end - timedelta(minutes=lookback_minutes)

    minute = start
    while minute <= end:
        yield minute.strftime(TIMESTAMP_FORMAT)
        minute += timedelta(minutes=1)


def toc_url(timestamp: str) -> str:
    return f"{BASE_URL}{timestamp}.toc.json.gz"


def event_id_for_url(url: str) -> str:
    """Derive a stable, deduplicating event ID from an article URL."""
    normalized = url.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_record(
    record: Mapping[str, Any], timestamp: str
) -> dict[str, Any] | None:
    """Convert one raw TOC record into the stable news-topic event schema."""
    url = (record.get("url") or "").strip()
    title = (record.get("title") or "").strip()
    if not url or not title:
        return None

    return {
        "event_id": event_id_for_url(url),
        "toc_id": record.get("ID"),
        "toc_timestamp": timestamp,
        # GDELT's "date" is when its crawler observed/re-crawled the page
        # during this TOC batch, not the article's original publish date --
        # every record in a given TOC file shares this exact value. Do not
        # treat it as a publication timestamp.
        "gdelt_seen_at": record.get("date"),
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": title,
        "url": url,
        "language": record.get("lang"),
        "image_url": record.get("img") or None,
        "source": "gdelt_web_ngrams",
        "event_type": "article",
    }


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated env var into a lowercase, de-duplicated tuple.

    An empty or unset value yields an empty tuple, which callers should
    treat as "no filter" (i.e. everything passes).
    """
    if not value:
        return ()
    seen: dict[str, None] = {}
    for part in value.split(","):
        normalized = part.strip().lower()
        if normalized:
            seen[normalized] = None
    return tuple(seen.keys())


def matches_language(
    event: Mapping[str, Any],
    languages: tuple[str, ...] = (),
) -> bool:
    """Return True if a normalized event's language is in scope.

    This is a scope/compatibility filter, not a relevance filter: it exists
    because downstream anomaly detection and embedding models are scoped to
    specific languages, not because language predicts topical relevance.
    Deciding *which* articles are actually related to a Wikipedia anomaly
    is the job of the similarity search stage (embeddings + cosine
    similarity), not this producer. An empty `languages` tuple disables the
    filter entirely (everything passes).
    """
    if not languages:
        return True

    language = (event.get("language") or "").strip().lower()
    return language in languages


class RecentUrls:
    """Bounded in-memory set used to suppress duplicate article URLs."""

    def __init__(self, capacity: int = RECENT_URL_CAPACITY):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._queue: list[str] = []
        self._ids: set[str] = set()

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._ids

    def add(self, event_id: str) -> None:
        if event_id in self._ids:
            return
        if len(self._queue) >= self.capacity:
            oldest = self._queue.pop(0)
            self._ids.discard(oldest)
        self._queue.append(event_id)
        self._ids.add(event_id)


class CompletedTimestamps:
    """Persists which TOC timestamps have already been fully processed."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self._timestamps: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text())
            self._timestamps = set(payload.get("completed_timestamps", []))
        except (json.JSONDecodeError, OSError):
            self._timestamps = set()

    def __contains__(self, timestamp: str) -> bool:
        return timestamp in self._timestamps

    def mark_done(self, timestamp: str) -> None:
        self._timestamps.add(timestamp)

    def prune(self, now_utc: datetime, retention_minutes: int) -> None:
        cutoff = now_utc - timedelta(minutes=retention_minutes)
        cutoff_str = cutoff.strftime(TIMESTAMP_FORMAT)
        self._timestamps = {ts for ts in self._timestamps if ts >= cutoff_str}

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"completed_timestamps": sorted(self._timestamps)}
        self.state_file.write_text(json.dumps(payload))


def fetch_toc_records(timestamp: str) -> list[dict[str, Any]] | None:
    """Fetch and parse one TOC file, or return None if not yet published."""
    url = toc_url(timestamp)
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except RequestException as error:
            if attempt == MAX_FETCH_ATTEMPTS:
                print(f"Giving up on {timestamp} after {attempt} attempts: {error}")
                return None
            print(
                f"Fetch error for {timestamp} (attempt {attempt}/"
                f"{MAX_FETCH_ATTEMPTS}): {error}. Retrying in {delay}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)
            continue

        if response.status_code == 404:
            return None

        if response.status_code != 200:
            if attempt == MAX_FETCH_ATTEMPTS:
                print(
                    f"Giving up on {timestamp}: HTTP {response.status_code}"
                )
                return None
            print(
                f"HTTP {response.status_code} for {timestamp} "
                f"(attempt {attempt}/{MAX_FETCH_ATTEMPTS}). Retrying in "
                f"{delay}s..."
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)
            continue

        try:
            decompressed = gzip.decompress(response.content).decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Failed to decompress TOC for {timestamp}: {error}")
            return None

        records = []
        for line in decompressed.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    return None


class GdeltIngestionProducer:
    """Poll GDELT Web NGrams TOC files and publish articles to Kafka."""

    def __init__(
        self,
        kafka_producer: Any,
        kafka_topic: str = DEFAULT_KAFKA_TOPIC,
        completed_timestamps: CompletedTimestamps | None = None,
        recent_urls: RecentUrls | None = None,
        languages: tuple[str, ...] = (),
    ):
        self.kafka = kafka_producer
        self.kafka_topic = kafka_topic
        self.completed = completed_timestamps
        self.recent_urls = recent_urls or RecentUrls()
        self.languages = languages

    def publish_article(self, event: dict[str, Any]) -> bool:
        """Publish one normalized article event, keyed by its event ID."""
        event_id = event["event_id"]
        if event_id in self.recent_urls:
            return False

        delay = INITIAL_RETRY_DELAY
        for attempt in range(1, MAX_PUBLISH_ATTEMPTS + 1):
            try:
                future = self.kafka.send(
                    self.kafka_topic,
                    key=event_id,
                    value=event,
                )
                future.get(timeout=PUBLISH_TIMEOUT_SECONDS)
                self.recent_urls.add(event_id)
                print(f"Sent: {event['title'][:80]} ({event['url']})")
                return True
            except KafkaError as error:
                if attempt == MAX_PUBLISH_ATTEMPTS:
                    raise
                print(
                    f"Kafka publish failed (attempt {attempt}/"
                    f"{MAX_PUBLISH_ATTEMPTS}): {error}. Retrying in "
                    f"{delay}s..."
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)

        return False

    def process_timestamp(self, timestamp: str) -> int:
        """Fetch, normalize, scope-filter, and publish one TOC timestamp.

        Returns the number of articles actually published. Relevance to any
        particular Wikipedia anomaly is intentionally not decided here; that
        judgment belongs to the downstream embedding / similarity search
        stage, which can compare semantically related articles that would
        never match a lexical keyword filter.
        """
        if self.completed is not None and timestamp in self.completed:
            return 0

        records = fetch_toc_records(timestamp)
        if records is None:
            return 0

        sent = 0
        for record in records:
            event = normalize_record(record, timestamp)
            if event is None:
                continue
            if not matches_language(event, self.languages):
                continue
            if self.publish_article(event):
                sent += 1

        if self.completed is not None:
            self.completed.mark_done(timestamp)

        if sent:
            print(f"{timestamp}: published {sent}/{len(records)} article(s)")

        return sent

    def poll_once(
        self,
        lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
        safety_delay_minutes: int = DEFAULT_SAFETY_DELAY_MINUTES,
    ) -> int:
        now_utc = datetime.now(timezone.utc)
        total_sent = 0
        for timestamp in candidate_timestamps(
            now_utc, lookback_minutes, safety_delay_minutes
        ):
            total_sent += self.process_timestamp(timestamp)

        if self.completed is not None:
            self.completed.prune(now_utc, COMPLETED_TIMESTAMP_RETENTION_MINUTES)
            self.completed.save()

        return total_sent

    def run(
        self,
        poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
        lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
        safety_delay_minutes: int = DEFAULT_SAFETY_DELAY_MINUTES,
    ) -> None:
        print(
            "GDELT producer started. Polling every "
            f"{poll_interval_seconds}s for new TOC files..."
        )
        while True:
            try:
                self.poll_once(lookback_minutes, safety_delay_minutes)
            except Exception as error:  # noqa: BLE001 - keep the poller alive
                print(f"Unexpected error during poll cycle: {error}")
            time.sleep(poll_interval_seconds)

    def close(self) -> None:
        self.kafka.flush()
        self.kafka.close()


def build_kafka_producer(kafka_server: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=kafka_server,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(
            value,
            ensure_ascii=False,
        ).encode("utf-8"),
        acks="all",
    )


def run() -> None:
    kafka_server = os.environ.get("KAFKA_SERVER", DEFAULT_KAFKA_SERVER)
    kafka_topic = os.environ.get("GDELT_KAFKA_TOPIC", DEFAULT_KAFKA_TOPIC)
    poll_interval_seconds = int(
        os.environ.get("GDELT_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS)
    )
    lookback_minutes = int(
        os.environ.get("GDELT_LOOKBACK_MINUTES", DEFAULT_LOOKBACK_MINUTES)
    )
    safety_delay_minutes = int(
        os.environ.get(
            "GDELT_SAFETY_DELAY_MINUTES", DEFAULT_SAFETY_DELAY_MINUTES
        )
    )
    state_file = os.environ.get("GDELT_STATE_FILE", DEFAULT_STATE_FILE)
    languages = parse_csv_list(
        os.environ.get("GDELT_LANGUAGES", DEFAULT_LANGUAGES)
    )

    if languages:
        print(f"Restricting to languages: {', '.join(languages)}")

    ingestion = GdeltIngestionProducer(
        kafka_producer=build_kafka_producer(kafka_server),
        kafka_topic=kafka_topic,
        completed_timestamps=CompletedTimestamps(state_file),
        languages=languages,
    )

    try:
        ingestion.run(
            poll_interval_seconds=poll_interval_seconds,
            lookback_minutes=lookback_minutes,
            safety_delay_minutes=safety_delay_minutes,
        )
    except KeyboardInterrupt:
        print("\nStopping GDELT producer...")
    finally:
        ingestion.close()
        print("GDELT producer stopped.")


if __name__ == "__main__":
    run()
