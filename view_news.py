"""Pretty-print recent articles published to the news-topic Kafka topic."""

import argparse
import json
import os
import textwrap
from collections import Counter

from kafka import KafkaConsumer

DEFAULT_KAFKA_SERVER = "localhost:9092"
DEFAULT_TOPIC = "news-topic"
TITLE_WRAP_WIDTH = 88


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tail and pretty-print articles from the news-topic Kafka topic."
    )
    parser.add_argument(
        "--offset",
        choices=["earliest", "latest"],
        default="latest",
        help=(
            "Where to start reading from. 'latest' (default) shows only "
            "new articles published after this script starts -- keep the "
            "producer running in another terminal to see them arrive. "
            "'earliest' replays everything still retained on the topic."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of articles to display (default: 25).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help=(
            "Seconds to wait for messages before giving up (default: 90). "
            "GDELT's producer polls every 60s and GDELT itself only "
            "publishes new files during ~15-minute heartbeat windows, so "
            "you may need to wait a while with --offset latest."
        ),
    )
    return parser.parse_args()


def print_article(index, article):
    """Print one article as a short multi-line "card" instead of a wide row.

    Card layout avoids the extremely long single lines a table would need
    for title + url together, which either overflow the terminal width
    (causing ugly wrapping) or misalign when titles contain wide CJK
    characters that occupy two terminal columns each.
    """
    language = article.get("language") or "??"
    # "gdelt_seen_at" is when GDELT's crawler observed the page during its
    # TOC batch, not the article's original publish date -- GDELT doesn't
    # reliably expose that. Every article in a given poll's TOC file shares
    # the same gdelt_seen_at value.
    when = article.get("gdelt_seen_at") or article.get("observed_at") or "unknown time"
    title = article.get("title") or "(no title)"
    url = article.get("url") or "(no url)"

    print(f"[{index:>3}] {language:<5} {when}")
    for line in textwrap.wrap(title, TITLE_WRAP_WIDTH) or [title]:
        print(f"      {line}")
    print(f"      {url}")
    print()


def main():
    args = parse_args()
    kafka_server = os.environ.get("KAFKA_SERVER", DEFAULT_KAFKA_SERVER)
    topic = os.environ.get("GDELT_KAFKA_TOPIC", DEFAULT_TOPIC)

    print(f"Connecting to {topic} on {kafka_server} (offset={args.offset})...")
    print(f"Waiting up to {args.timeout}s for up to {args.limit} article(s)...\n")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_server,
        auto_offset_reset=args.offset,
        consumer_timeout_ms=args.timeout * 1000,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    articles = []
    try:
        for message in consumer:
            articles.append(message.value)
            if len(articles) >= args.limit:
                break
    finally:
        consumer.close()

    if not articles:
        print(
            "No articles received. Is producer/gdelt_producer.py running? "
            "Try --offset earliest to replay the topic's history instead."
        )
        return

    articles.sort(key=lambda article: article.get("gdelt_seen_at") or "", reverse=True)

    print("=" * 78)
    print(f"📰 NEWS-TOPIC STREAM ({len(articles)} article(s))")
    print("=" * 78)
    print()
    for index, article in enumerate(articles, start=1):
        print_article(index, article)
    print("=" * 78)

    languages = Counter(article.get("language") or "unknown" for article in articles)
    print("\nLanguage distribution:")
    for language, count in languages.most_common():
        print(f"  {language:>8}: {count}")


if __name__ == "__main__":
    main()
