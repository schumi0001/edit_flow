import json


def parse_anomaly_message(payload):
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="ignore")

    if not isinstance(payload, str):
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    return data
