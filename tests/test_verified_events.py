import unittest

import numpy as np

from dashboard.anomaly_utils import (
    VERIFIED_STATUS_BELOW_THRESHOLD,
    VERIFIED_STATUS_MATCHED,
    VERIFIED_STATUS_NO_CANDIDATE,
    flatten_verified_event,
)
from vectordb.match_events import EventVerifier


ANOMALY = {
    "page_title": "2026_California_wildfires",
    "window_start": "2026-08-10 14:30:00",
    "window_end": "2026-08-10 14:45:00",
    "edit_count": 42,
    "unique_editors": 17,
    "total_byte_changes": 90210,
    "anomaly_score": -0.31,
}

ARTICLE = {
    "event_id": "abc123",
    "title": "Wildfires force evacuations in LA County",
    "url": "https://news.example.com/wildfires-la",
    "language": "en",
    "gdelt_seen_at": "2026-08-10T14:20:00Z",
    "observed_at": "2026-08-10T14:22:31Z",
    "toc_id": 7,
}


class FakeFuture:
    def get(self, timeout=None):
        return None


class FakeKafkaProducer:
    def __init__(self):
        self.sent = []

    def send(self, topic, key=None, value=None):
        self.sent.append((topic, key, value))
        return FakeFuture()

    def flush(self):
        pass

    def close(self):
        pass


class FakeIndex:
    """Stands in for NewsEmbeddingIndex without any Qdrant connection."""

    def __init__(self, match=None):
        self.match = match

    def add(self, article, embedding):
        pass

    def prune(self):
        pass

    def best_match(self, embedding):
        return self.match


def fake_embed(text):
    return np.zeros(3)


def build_verifier(match, threshold=0.65):
    return EventVerifier(
        kafka_producer=FakeKafkaProducer(),
        index=FakeIndex(match=match),
        embed_fn=fake_embed,
        similarity_threshold=threshold,
    )


class VerifiedEventPayloadTests(unittest.TestCase):
    def test_matched_anomaly_carries_complete_paired_result(self):
        verifier = build_verifier(match=(0.83, dict(ARTICLE)))

        record = verifier.handle_anomaly(dict(ANOMALY))

        self.assertTrue(record["matched"])
        self.assertEqual(record["similarity_score"], 0.83)
        self.assertEqual(record["similarity_threshold"], 0.65)
        self.assertEqual(record["edit_count"], 42)
        self.assertEqual(record["unique_editors"], 17)
        self.assertEqual(record["total_byte_changes"], 90210)
        self.assertEqual(
            record["matched_article"]["title"], ARTICLE["title"]
        )
        self.assertEqual(
            record["matched_article"]["gdelt_seen_at"],
            ARTICLE["gdelt_seen_at"],
        )
        self.assertEqual(record["closest_article"], record["matched_article"])

    def test_below_threshold_keeps_closest_candidate(self):
        verifier = build_verifier(match=(0.41, dict(ARTICLE)))

        record = verifier.handle_anomaly(dict(ANOMALY))

        self.assertFalse(record["matched"])
        self.assertEqual(record["similarity_score"], 0.41)
        # Documented meaning of matched_article is unchanged...
        self.assertIsNone(record["matched_article"])
        # ...but the best candidate is still published for explainability.
        self.assertEqual(
            record["closest_article"]["url"], ARTICLE["url"]
        )

    def test_empty_index_publishes_null_similarity(self):
        verifier = build_verifier(match=None)

        record = verifier.handle_anomaly(dict(ANOMALY))

        self.assertFalse(record["matched"])
        self.assertIsNone(record["similarity_score"])
        self.assertIsNone(record["matched_article"])
        self.assertIsNone(record["closest_article"])

    def test_record_is_published_to_kafka(self):
        verifier = build_verifier(match=(0.83, dict(ARTICLE)))

        record = verifier.handle_anomaly(dict(ANOMALY))

        topic, key, value = verifier.kafka.sent[-1]
        self.assertEqual(topic, "anomaly-news-verdicts")
        self.assertEqual(key, ANOMALY["page_title"])
        self.assertEqual(value, record)


class FlattenVerifiedEventTests(unittest.TestCase):
    def test_flattens_new_format_matched_record(self):
        record = {
            **ANOMALY,
            "matched": True,
            "similarity_score": 0.83,
            "similarity_threshold": 0.65,
            "matched_article": dict(ARTICLE),
            "closest_article": dict(ARTICLE),
            "evaluated_at": "2026-08-10T14:46:00Z",
        }

        row = flatten_verified_event(record)

        self.assertEqual(row["status"], VERIFIED_STATUS_MATCHED)
        self.assertEqual(row["news_title"], ARTICLE["title"])
        self.assertEqual(row["news_domain"], "news.example.com")
        self.assertEqual(row["news_language"], "en")
        self.assertEqual(row["news_seen_at"], ARTICLE["gdelt_seen_at"])
        self.assertEqual(row["similarity_threshold"], 0.65)
        self.assertEqual(row["unique_editors"], 17)
        self.assertEqual(row["total_byte_changes"], 90210)

    def test_below_threshold_row_uses_closest_article(self):
        record = {
            **ANOMALY,
            "matched": False,
            "similarity_score": 0.41,
            "similarity_threshold": 0.65,
            "matched_article": None,
            "closest_article": dict(ARTICLE),
        }

        row = flatten_verified_event(record)

        self.assertEqual(row["status"], VERIFIED_STATUS_BELOW_THRESHOLD)
        self.assertEqual(row["news_title"], ARTICLE["title"])
        self.assertEqual(row["similarity_score"], 0.41)

    def test_old_format_message_without_new_fields(self):
        # Pre-upgrade payload: no unique_editors/total_byte_changes/
        # threshold/closest_article; matched_article only when matched.
        record = {
            "page_title": "Some_Page",
            "window_start": "2026-08-09 21:00:00",
            "window_end": "2026-08-09 21:15:00",
            "anomaly_score": -0.2,
            "edit_count": 9,
            "matched": True,
            "similarity_score": 0.75,
            "matched_article": {
                "title": "Old style article",
                "url": "https://old.example.com/story",
                "language": "en",
                "event_id": "def456",
            },
            "evaluated_at": "2026-08-09T21:16:00Z",
        }

        row = flatten_verified_event(record)

        self.assertEqual(row["status"], VERIFIED_STATUS_MATCHED)
        self.assertEqual(row["news_title"], "Old style article")
        self.assertEqual(row["news_domain"], "old.example.com")
        self.assertIsNone(row["news_seen_at"])
        self.assertIsNone(row["similarity_threshold"])
        self.assertIsNone(row["unique_editors"])
        self.assertIsNone(row["total_byte_changes"])

    def test_no_candidate_status(self):
        record = {
            "page_title": "Lonely_Page",
            "matched": False,
            "similarity_score": None,
            "matched_article": None,
        }

        row = flatten_verified_event(record)

        self.assertEqual(row["status"], VERIFIED_STATUS_NO_CANDIDATE)
        self.assertIsNone(row["news_title"])
        self.assertIsNone(row["news_url"])


if __name__ == "__main__":
    unittest.main()
