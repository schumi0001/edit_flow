import json
from urllib.parse import urlparse

VERIFIED_STATUS_MATCHED = "✅ Verified"
VERIFIED_STATUS_BELOW_THRESHOLD = "⚠️ Below threshold"
VERIFIED_STATUS_NO_CANDIDATE = "No matching article"


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


def flatten_verified_event(record):
    """Flatten one anomaly-news-verdicts message into flat display fields.

    Backward compatible with older messages, which lack unique_editors,
    total_byte_changes, similarity_threshold, and closest_article (those
    only carried matched_article, and only when the match passed the
    threshold) -- missing fields simply come back as None.

    The verification status is derived from the `matched` flag computed by
    the matcher itself (similarity_score >= its threshold), never
    re-derived in the UI, so the two can't disagree.
    """
    article = (
        record.get("closest_article")
        or record.get("matched_article")
        or {}
    )
    similarity_score = record.get("similarity_score")

    if record.get("matched"):
        status = VERIFIED_STATUS_MATCHED
    elif similarity_score is not None:
        status = VERIFIED_STATUS_BELOW_THRESHOLD
    else:
        status = VERIFIED_STATUS_NO_CANDIDATE

    url = article.get("url")
    domain = None
    if isinstance(url, str) and url:
        domain = urlparse(url).netloc or None

    return {
        "page_title": record.get("page_title"),
        "window_start": record.get("window_start"),
        "window_end": record.get("window_end"),
        "edit_count": record.get("edit_count"),
        "unique_editors": record.get("unique_editors"),
        "total_byte_changes": record.get("total_byte_changes"),
        "anomaly_score": record.get("anomaly_score"),
        "news_title": article.get("title"),
        "news_domain": domain,
        "news_url": url,
        "news_language": article.get("language"),
        "news_seen_at": (
            article.get("gdelt_seen_at") or article.get("observed_at")
        ),
        "similarity_score": similarity_score,
        "similarity_threshold": record.get("similarity_threshold"),
        "matched": bool(record.get("matched")),
        "status": status,
        "evaluated_at": record.get("evaluated_at"),
    }
