import unittest

from dashboard.anomaly_utils import parse_anomaly_message


class AnomalyParsingTests(unittest.TestCase):
    def test_parse_anomaly_message_accepts_json_bytes(self):
        payload = b'{"page_title":"Breaking_News_Event_2026","anomaly_score":-0.42,"edit_count":150}'

        parsed = parse_anomaly_message(payload)

        self.assertEqual(parsed["page_title"], "Breaking_News_Event_2026")
        self.assertEqual(parsed["edit_count"], 150)
        self.assertEqual(parsed["anomaly_score"], -0.42)

    def test_parse_anomaly_message_returns_none_for_invalid_payload(self):
        self.assertIsNone(parse_anomaly_message("not-json"))


if __name__ == "__main__":
    unittest.main()
