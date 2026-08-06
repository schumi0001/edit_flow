"""Unit tests for the GDELT-to-Kafka news producer."""

import gzip
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from producer.gdelt_producer import (
    CompletedTimestamps,
    GdeltIngestionProducer,
    RecentUrls,
    candidate_timestamps,
    event_id_for_url,
    fetch_toc_records,
    matches_language,
    normalize_record,
    parse_csv_list,
    toc_url,
)


class CompletedFuture:
    def get(self, timeout):
        return {"timeout": timeout}


class FakeKafkaProducer:
    def __init__(self):
        self.messages = []
        self.flushed = False
        self.closed = False

    def send(self, topic, key, value):
        self.messages.append({"topic": topic, "key": key, "value": value})
        return CompletedFuture()

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


def gzip_lines(records):
    body = "\n".join(json.dumps(record) for record in records)
    return gzip.compress(body.encode("utf-8"))


def make_record(**overrides):
    values = {
        "ID": 1,
        "date": "2026-08-06T02:17:00.000Z",
        "img": "",
        "lang": "en",
        "title": "A developing story",
        "url": "https://example.com/a-developing-story",
    }
    values.update(overrides)
    return values


class CandidateTimestampsTests(unittest.TestCase):
    def test_applies_safety_delay_and_lookback_window(self):
        now_utc = datetime(2026, 8, 6, 2, 30, 0, tzinfo=timezone.utc)

        timestamps = list(
            candidate_timestamps(
                now_utc, lookback_minutes=10, safety_delay_minutes=5
            )
        )

        self.assertEqual(timestamps[0], "20260806021500")
        self.assertEqual(timestamps[-1], "20260806022500")
        self.assertEqual(len(timestamps), 11)

    def test_toc_url_uses_minute_resolution_timestamp(self):
        url = toc_url("20260806021700")

        self.assertTrue(url.endswith("20260806021700.toc.json.gz"))


class EventIdTests(unittest.TestCase):
    def test_same_url_produces_same_id(self):
        first = event_id_for_url("https://example.com/story")
        second = event_id_for_url("https://example.com/story")

        self.assertEqual(first, second)

    def test_different_urls_produce_different_ids(self):
        first = event_id_for_url("https://example.com/story-one")
        second = event_id_for_url("https://example.com/story-two")

        self.assertNotEqual(first, second)


class NormalizationTests(unittest.TestCase):
    def test_normalizes_record_to_event_contract(self):
        event = normalize_record(make_record(), "20260806021700")

        self.assertEqual(event["toc_id"], 1)
        self.assertEqual(event["toc_timestamp"], "20260806021700")
        self.assertEqual(event["gdelt_seen_at"], "2026-08-06T02:17:00.000Z")
        self.assertEqual(event["title"], "A developing story")
        self.assertEqual(event["url"], "https://example.com/a-developing-story")
        self.assertEqual(event["language"], "en")
        self.assertIsNone(event["image_url"])
        self.assertEqual(event["source"], "gdelt_web_ngrams")
        self.assertEqual(event["event_type"], "article")
        self.assertEqual(
            event["event_id"], event_id_for_url(event["url"])
        )

    def test_rejects_records_missing_title_or_url(self):
        self.assertIsNone(normalize_record(make_record(title=""), "ts"))
        self.assertIsNone(normalize_record(make_record(url=""), "ts"))


class FetchTocRecordsTests(unittest.TestCase):
    def test_returns_none_on_404(self):
        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=FakeResponse(404),
        ):
            self.assertIsNone(fetch_toc_records("20260806021500"))

    def test_parses_gzipped_json_lines_on_success(self):
        records = [make_record(ID=1), make_record(ID=2, url="https://example.com/2")]
        response = FakeResponse(200, gzip_lines(records))

        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=response,
        ):
            parsed = fetch_toc_records("20260806021700")

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["ID"], 1)
        self.assertEqual(parsed[1]["ID"], 2)

    def test_skips_malformed_lines_without_failing(self):
        valid_line = json.dumps(make_record()).encode("utf-8")
        payload = gzip.compress(b"not-json\n" + valid_line)
        response = FakeResponse(200, payload)

        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=response,
        ):
            parsed = fetch_toc_records("20260806021700")

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "A developing story")


class ParseCsvListTests(unittest.TestCase):
    def test_returns_empty_tuple_for_none_or_blank(self):
        self.assertEqual(parse_csv_list(None), ())
        self.assertEqual(parse_csv_list(""), ())
        self.assertEqual(parse_csv_list("   "), ())

    def test_normalizes_case_whitespace_and_duplicates(self):
        result = parse_csv_list(" EN, es , en,FR ")

        self.assertEqual(result, ("en", "es", "fr"))


class MatchesLanguageTests(unittest.TestCase):
    def test_no_languages_configured_always_matches(self):
        event = {"language": "en", "title": "Anything at all"}

        self.assertTrue(matches_language(event))

    def test_accepts_matching_language(self):
        event = {"language": "en", "title": "Story"}

        self.assertTrue(matches_language(event, languages=("en", "es")))

    def test_rejects_other_language(self):
        event = {"language": "fr", "title": "Histoire"}

        self.assertFalse(matches_language(event, languages=("en", "es")))

    def test_is_case_insensitive(self):
        event = {"language": "EN", "title": "Story"}

        self.assertTrue(matches_language(event, languages=("en",)))


class RecentUrlsTests(unittest.TestCase):
    def test_evicts_oldest_entry_at_capacity(self):
        recent = RecentUrls(capacity=2)
        recent.add("a")
        recent.add("b")
        recent.add("c")

        self.assertNotIn("a", recent)
        self.assertIn("b", recent)
        self.assertIn("c", recent)


class CompletedTimestampsTests(unittest.TestCase):
    def test_round_trips_through_disk(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"

            first = CompletedTimestamps(state_file)
            first.mark_done("20260806021700")
            first.save()

            second = CompletedTimestamps(state_file)
            self.assertIn("20260806021700", second)

    def test_prune_removes_old_timestamps(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            completed = CompletedTimestamps(state_file)
            completed.mark_done("20260101000000")
            completed.mark_done("20260806021700")

            completed.prune(
                datetime(2026, 8, 6, 2, 30, tzinfo=timezone.utc),
                retention_minutes=180,
            )

            self.assertNotIn("20260101000000", completed)
            self.assertIn("20260806021700", completed)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.kafka = FakeKafkaProducer()
        self.producer = GdeltIngestionProducer(
            kafka_producer=self.kafka,
            kafka_topic="news-topic",
            recent_urls=RecentUrls(capacity=10),
        )

    def test_publishes_with_event_id_as_kafka_key(self):
        event = normalize_record(make_record(), "20260806021700")

        published = self.producer.publish_article(event)

        self.assertTrue(published)
        self.assertEqual(len(self.kafka.messages), 1)
        message = self.kafka.messages[0]
        self.assertEqual(message["topic"], "news-topic")
        self.assertEqual(message["key"], event["event_id"])
        self.assertEqual(message["value"]["url"], event["url"])

    def test_suppresses_duplicate_urls(self):
        event = normalize_record(make_record(), "20260806021700")

        self.assertTrue(self.producer.publish_article(event))
        self.assertFalse(self.producer.publish_article(event))
        self.assertEqual(len(self.kafka.messages), 1)

    def test_close_flushes_and_closes_kafka(self):
        self.producer.close()

        self.assertTrue(self.kafka.flushed)
        self.assertTrue(self.kafka.closed)


class ProcessTimestampTests(unittest.TestCase):
    def test_marks_timestamp_completed_after_processing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            kafka = FakeKafkaProducer()
            completed = CompletedTimestamps(state_file)
            producer = GdeltIngestionProducer(
                kafka_producer=kafka,
                completed_timestamps=completed,
            )
            response = FakeResponse(200, gzip_lines([make_record()]))

            with patch(
                "producer.gdelt_producer.requests.get",
                return_value=response,
            ):
                sent = producer.process_timestamp("20260806021700")

            self.assertEqual(sent, 1)
            self.assertIn("20260806021700", completed)

    def test_language_filter_is_applied_before_publishing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            kafka = FakeKafkaProducer()
            records = [
                make_record(
                    ID=1,
                    url="https://example.com/1",
                    title="An English-language story",
                    lang="en",
                ),
                make_record(
                    ID=2,
                    url="https://example.com/2",
                    title="Une histoire en français",
                    lang="fr",
                ),
            ]
            producer = GdeltIngestionProducer(
                kafka_producer=kafka,
                completed_timestamps=CompletedTimestamps(state_file),
                languages=("en",),
            )
            response = FakeResponse(200, gzip_lines(records))

            with patch(
                "producer.gdelt_producer.requests.get",
                return_value=response,
            ):
                sent = producer.process_timestamp("20260806021700")

            self.assertEqual(sent, 1)
            self.assertEqual(len(kafka.messages), 1)
            self.assertEqual(
                kafka.messages[0]["value"]["url"], "https://example.com/1"
            )

    def test_skips_already_completed_timestamps_without_fetching(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            kafka = FakeKafkaProducer()
            completed = CompletedTimestamps(state_file)
            completed.mark_done("20260806021700")
            producer = GdeltIngestionProducer(
                kafka_producer=kafka,
                completed_timestamps=completed,
            )

            with patch(
                "producer.gdelt_producer.requests.get",
                side_effect=AssertionError("should not fetch"),
            ):
                sent = producer.process_timestamp("20260806021700")

            self.assertEqual(sent, 0)


if __name__ == "__main__":
    unittest.main()
