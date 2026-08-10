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
    build_snippets,
    candidate_timestamps,
    event_id_for_url,
    fetch_ngrams_records,
    fetch_toc_records,
    matches_language,
    ngrams_url,
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


def gzip_text(text):
    return gzip.compress(text.encode("utf-8"))


def make_ngrams_text(rows):
    """Build raw (undecompressed) ngrams.txt content from (docid, quadgram, count) rows."""
    return "\n".join(f"{docid}\t{quadgram}\t{count}" for docid, quadgram, count in rows)


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

    def test_ngrams_url_uses_minute_resolution_timestamp_and_same_base(self):
        toc = toc_url("20260806021700")
        ngrams = ngrams_url("20260806021700")

        self.assertTrue(ngrams.endswith("20260806021700.ngrams.txt.gz"))
        # Companion file lives at the exact same base path as the TOC file.
        self.assertEqual(
            toc.rsplit("/", 1)[0], ngrams.rsplit("/", 1)[0]
        )


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
        self.assertEqual(event["snippet"], "")
        self.assertEqual(
            event["event_id"], event_id_for_url(event["url"])
        )

    def test_rejects_records_missing_title_or_url(self):
        self.assertIsNone(normalize_record(make_record(title=""), "ts"))
        self.assertIsNone(normalize_record(make_record(url=""), "ts"))

    def test_snippet_defaults_to_empty_string_and_can_be_provided(self):
        without_snippet = normalize_record(make_record(), "20260806021700")
        self.assertEqual(without_snippet["snippet"], "")

        with_snippet = normalize_record(
            make_record(), "20260806021700", snippet="disease outbreak spreads"
        )
        self.assertEqual(with_snippet["snippet"], "disease outbreak spreads")


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


class FetchNgramsRecordsTests(unittest.TestCase):
    def test_returns_none_on_404(self):
        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=FakeResponse(404),
        ):
            self.assertIsNone(fetch_ngrams_records("20260806021500"))

    def test_parses_tab_delimited_lines_on_success(self):
        text = make_ngrams_text(
            [(56, "diseases such as measles,", 3), (56, "disease later in life.", 1)]
        )
        response = FakeResponse(200, gzip_text(text))

        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=response,
        ):
            parsed = fetch_ngrams_records("20260806021700")

        self.assertEqual(
            parsed,
            [
                (56, "diseases such as measles,", 3),
                (56, "disease later in life.", 1),
            ],
        )

    def test_skips_malformed_lines_without_failing(self):
        text = "\n".join(
            [
                "not-a-valid-line",  # wrong column count
                "abc\tsome quadgram\t5",  # non-integer DOCID
                "56\tsome quadgram\tnotanumber",  # non-integer COUNT
                "56\t\t3",  # empty quadgram
                "56\treal quadgram here\t3",  # valid
            ]
        )
        response = FakeResponse(200, gzip_text(text))

        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=response,
        ):
            parsed = fetch_ngrams_records("20260806021700")

        self.assertEqual(parsed, [(56, "real quadgram here", 3)])

    def test_returns_empty_list_when_file_has_no_matching_lines(self):
        response = FakeResponse(200, gzip_text(""))

        with patch(
            "producer.gdelt_producer.requests.get",
            return_value=response,
        ):
            parsed = fetch_ngrams_records("20260806021700")

        self.assertEqual(parsed, [])


class BuildSnippetsTests(unittest.TestCase):
    def test_returns_empty_dict_for_none_or_empty_input(self):
        self.assertEqual(build_snippets(None), {})
        self.assertEqual(build_snippets([]), {})

    def test_groups_by_docid_and_orders_by_count_descending(self):
        records = [
            (1, "alpha quadgram here", 1),
            (1, "beta quadgram here", 5),
            (2, "gamma quadgram here", 2),
        ]

        snippets = build_snippets(records)

        self.assertEqual(snippets[1], "beta quadgram here alpha quadgram here")
        self.assertEqual(snippets[2], "gamma quadgram here")

    def test_caps_quadgrams_per_snippet(self):
        records = [(1, f"quadgram number {i}", i) for i in range(10)]

        snippets = build_snippets(records, max_quadgrams=3)

        self.assertEqual(snippets[1].count("quadgram number"), 3)
        # Keeps the highest-COUNT entries (9, 8, 7), not just the first 3 seen.
        self.assertIn("quadgram number 9", snippets[1])
        self.assertIn("quadgram number 8", snippets[1])
        self.assertIn("quadgram number 7", snippets[1])
        self.assertNotIn("quadgram number 0", snippets[1])

    def test_cleans_and_deduplicates_punctuation_variants(self):
        # These clean to the same text ("diseases such as measles") and
        # should be merged (counts summed) rather than kept as separate
        # entries.
        records = [
            (56, "diseases such as measles,", 3),
            (56, "diseases such as measles.", 2),
        ]

        snippets = build_snippets(records)

        self.assertEqual(snippets[56], "diseases such as measles")

    def test_skips_quadgrams_that_are_punctuation_only(self):
        records = [(1, "...", 5), (1, "real content here", 1)]

        snippets = build_snippets(records)

        self.assertEqual(snippets[1], "real content here")


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

    def test_attaches_snippet_from_companion_ngrams_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            kafka = FakeKafkaProducer()
            producer = GdeltIngestionProducer(
                kafka_producer=kafka,
                completed_timestamps=CompletedTimestamps(state_file),
            )
            records = [
                make_record(ID=1, url="https://example.com/1"),
                make_record(ID=2, url="https://example.com/2"),
            ]
            toc_response = FakeResponse(200, gzip_lines(records))
            ngrams_text = make_ngrams_text(
                [
                    (1, "wildfire evacuation orders issued", 4),
                    (2, "unrelated other story text", 2),
                ]
            )
            ngrams_response = FakeResponse(200, gzip_text(ngrams_text))

            def fake_get(url, **kwargs):
                if url.endswith(".toc.json.gz"):
                    return toc_response
                if url.endswith(".ngrams.txt.gz"):
                    return ngrams_response
                raise AssertionError(f"unexpected URL: {url}")

            with patch(
                "producer.gdelt_producer.requests.get", side_effect=fake_get
            ):
                sent = producer.process_timestamp("20260806021700")

            self.assertEqual(sent, 2)
            by_url = {
                message["value"]["url"]: message["value"]
                for message in kafka.messages
            }
            self.assertEqual(
                by_url["https://example.com/1"]["snippet"],
                "wildfire evacuation orders issued",
            )
            self.assertEqual(
                by_url["https://example.com/2"]["snippet"],
                "unrelated other story text",
            )

    def test_falls_back_to_empty_snippet_when_ngrams_file_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = f"{tmp_dir}/state.json"
            kafka = FakeKafkaProducer()
            producer = GdeltIngestionProducer(
                kafka_producer=kafka,
                completed_timestamps=CompletedTimestamps(state_file),
            )
            toc_response = FakeResponse(200, gzip_lines([make_record()]))

            def fake_get(url, **kwargs):
                if url.endswith(".toc.json.gz"):
                    return toc_response
                if url.endswith(".ngrams.txt.gz"):
                    return FakeResponse(404)
                raise AssertionError(f"unexpected URL: {url}")

            with patch(
                "producer.gdelt_producer.requests.get", side_effect=fake_get
            ):
                sent = producer.process_timestamp("20260806021700")

            self.assertEqual(sent, 1)
            self.assertEqual(kafka.messages[0]["value"]["snippet"], "")


if __name__ == "__main__":
    unittest.main()
