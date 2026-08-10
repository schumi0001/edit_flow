#!/usr/bin/env python3
"""Download real historical Wikipedia edits, normalize them, and train the model.

Unlike the live EventStreams feed (which only shows edits as they happen),
the MediaWiki `recentchanges` API lets us page backward through edits that
already happened, bounded by a date range -- this is genuine historical
data, not a tap on the live stream. The recentchanges table only retains
roughly the last 30 days of edits, so this can't reach further back than
that.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL_TEMPLATE = "https://{wiki}/w/api.php"
USER_AGENT = "WikiPulse/1.0 (student big-data project; historical training fetch)"


def _parse_timestamp(value: str) -> int:
    try:
        return int(
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except (TypeError, ValueError):
        return 0


def fetch_recent_changes(wiki: str, newest: datetime, oldest: datetime, limit: int) -> list[dict]:
    """Page backward through real historical edits, from `newest` down to `oldest`."""
    api_url = API_URL_TEMPLATE.format(wiki=wiki)
    params = {
        "action": "query",
        "list": "recentchanges",
        "rcprop": "title|user|timestamp|sizes|flags",
        "rctype": "edit|new",
        "rcnamespace": "0",
        "rclimit": "500",
        "rcdir": "older",
        "rcstart": newest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rcend": oldest.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json",
        "formatversion": "2",
    }

    records: list[dict] = []
    rccontinue = None

    while len(records) < limit:
        query = dict(params)
        if rccontinue:
            query["rccontinue"] = rccontinue

        url = f"{api_url}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        changes = payload.get("query", {}).get("recentchanges", [])
        if not changes:
            break

        records.extend(changes)
        print(f"  fetched {len(records)} edits so far...")

        rccontinue = payload.get("continue", {}).get("rccontinue")
        if not rccontinue:
            break

        time.sleep(0.2)  # be polite to the API

    return records[:limit]


def normalize_and_write(changes: list[dict], output_path: Path) -> int:
    """Write MediaWiki API edit records as JSONL matching the historical training schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for change in changes:
            page_title = change.get("title")
            user = change.get("user")
            if not page_title or not user:
                continue

            old_length = change.get("oldlen")
            new_length = change.get("newlen")
            byte_change = (
                new_length - old_length
                if old_length is not None and new_length is not None
                else 0
            )

            normalized = {
                "page_title": page_title,
                "user": user,
                "bot": bool(change.get("bot", False)),
                "minor": bool(change.get("minor", False)),
                "timestamp": _parse_timestamp(change.get("timestamp")),
                "byte_change": byte_change,
            }

            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real historical Wikipedia edits and train the model"
    )
    parser.add_argument(
        "--wiki",
        default="en.wikipedia.org",
        help="Wiki domain to pull historical edits from",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help=(
            "How many days of history to pull, counting back from now. "
            "The recentchanges table only retains roughly the last 30 "
            "days, so larger values will likely return nothing for the "
            "oldest portion of the range."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150000,
        help=(
            "Maximum number of historical edits to fetch. The fetch loop "
            "already pages through everything MediaWiki's recentchanges "
            "table has for the requested --days window and stops early "
            "once it runs out, so this is a ceiling, not a target -- "
            "150000 (~7.5x the old default of 20000) pulls much closer to "
            "the full ~30-day window instead of stopping well short of it. "
            "Expect a few hundred paginated requests (roughly 5-10 "
            "minutes) at the new default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "historical"),
        help="Directory where the normalized history file will be written",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional output model path, forwarded to train_model.py",
    )
    args = parser.parse_args()

    newest = datetime.now(timezone.utc)
    oldest = newest - timedelta(days=args.days)

    print(
        f"Fetching up to {args.limit} historical edits from {args.wiki} "
        f"between {oldest.isoformat()} and {newest.isoformat()}..."
    )
    changes = fetch_recent_changes(args.wiki, newest, oldest, args.limit)

    if not changes:
        print(
            "No historical edits were returned. The recentchanges table "
            "may not retain data that far back -- try a smaller --days "
            "value."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    normalized_path = output_dir / f"{args.wiki}.recentchanges.jsonl"
    written = normalize_and_write(changes, normalized_path)
    print(f"Wrote {written} normalized historical edits to {normalized_path}")

    command = [
        sys.executable,
        str(ROOT / "models" / "train_model.py"),
        "--history-jsonl",
        str(normalized_path),
    ]
    if args.model_path:
        command.extend(["--model-path", args.model_path])
    subprocess.run(command, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
