import os
from collections import deque
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from pipeline_manager import (
    PYTHON_PROCESSES,
    get_last_failure,
    get_pipeline_status,
    is_model_trained,
    read_log,
    start_pipeline,
    stop_pipeline,
    train_model,
)
from anomaly_utils import flatten_verified_event, parse_anomaly_message

# Anomaly/verified-event windows are 15 minutes wide and slide every 5
# minutes, so a single page shows up as several near-duplicate messages
# (one per overlapping window) and Kafka retains every one of them --
# without a recency cutoff, old severe anomalies (higher |score|) would
# permanently outrank fresher ones at the top of a score-sorted table.
ANOMALY_RECENCY_MINUTES = int(os.environ.get("ANOMALY_RECENCY_MINUTES", "60"))

# The live edit stream (producer/wikipedia_producer.py) is hardcoded to
# English Wikipedia's recent-changes SSE feed, so page_title always
# resolves against en.wikipedia.org.
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"


def wikipedia_page_url(page_title):
    if not page_title:
        return None
    return WIKIPEDIA_BASE_URL + quote(str(page_title).replace(" ", "_"))


def _dedup_latest_per_page(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Keep only the most recent row per page_title.

    Sliding-window aggregation re-emits the same page_title once per
    overlapping window as it keeps accumulating edits, so the same anomaly
    shows up several times with slightly different (always increasing)
    stats. Keeping only the latest avoids the table filling up with near-
    duplicates of the same underlying event.
    """
    return (
        df.sort_values(timestamp_column, ascending=False)
        .drop_duplicates(subset=["page_title"], keep="first")
    )

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="WikiPulse Dashboard",
    page_icon="🌐",
    layout="wide",
)

KAFKA_SERVER = "localhost:9092"

STATE_BADGES = {
    "running": ("🟢", "Running"),
    "ready": ("🟢", "Ready"),
    "stopped": ("🔴", "Stopped"),
    "missing": ("🔴", "Missing"),
    "blocked": ("🟡", "Blocked"),
}


# ---------------------------------------------------------
# Kafka topic reading (shared by all intelligence sections)
# ---------------------------------------------------------

@st.cache_data(ttl=8, show_spinner=False)
def read_topic_messages(topic, limit=400, timeout_ms=3000):
    """Read up to the last `limit` JSON messages from a Kafka topic.

    Returns [] when Kafka is unreachable or the topic is empty. A bounded
    deque keeps memory flat even if the topic grows large.
    """
    messages = deque(maxlen=limit)

    try:
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_SERVER,
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
        )

        for message in consumer:
            parsed = parse_anomaly_message(message.value)
            if parsed is not None:
                messages.append(parsed)

        consumer.close()
    except Exception:
        return []

    return list(messages)


def format_window(start, end):
    """Render a page/window pair like 'Aug 10 14:30 – 14:45' compactly."""
    try:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        if pd.isna(start_time) or pd.isna(end_time):
            raise ValueError
        return f"{start_time:%b %d %H:%M} – {end_time:%H:%M}"
    except (ValueError, TypeError):
        if start or end:
            return f"{start or '?'} – {end or '?'}"
        return None


# ---------------------------------------------------------
# Pipeline control panel
# ---------------------------------------------------------

if "pipeline_message" not in st.session_state:
    st.session_state.pipeline_message = None

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None


status = get_pipeline_status()

with st.sidebar:
    st.header("Pipeline health")

    for component, entry in status.items():
        icon, state_label = STATE_BADGES.get(entry["state"], ("⚪", "Unknown"))
        st.write(f"{icon} **{component}:** {state_label}")

        if entry.get("detail"):
            st.caption(entry["detail"])

        # Crash log for managed processes that died on their own.
        if entry["kind"] == "process" and entry["state"] != "running":
            failure = get_last_failure(entry["process"])

            if failure:
                with st.expander(f"Why did {component} stop?"):
                    st.code(failure["log_tail"], language="text")

    if not is_model_trained():
        st.warning(
            "The anomaly model has not been trained yet. Anomaly "
            "detection (and therefore news matching results) stays "
            "disabled until you train it. Training uses the committed "
            "historical dataset and takes well under a minute."
        )

        if st.button("Train model", use_container_width=True):
            with st.spinner("Training anomaly model..."):
                try:
                    output = train_model()
                    last_line = output.splitlines()[-1] if output else ""
                    st.session_state.pipeline_message = (
                        f"Model trained. {last_line}"
                    )
                    st.session_state.pipeline_error = None
                except Exception as error:
                    st.session_state.pipeline_error = str(error)
                    st.session_state.pipeline_message = None

            st.rerun()

    start_column, stop_column = st.columns(2)

    with start_column:
        if st.button(
            "Start",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = start_pipeline()
                st.session_state.pipeline_message = (
                    " · ".join(result["messages"])
                )
                st.session_state.pipeline_error = (
                    "\n\n".join(result["errors"]) or None
                )
            except Exception as error:
                st.session_state.pipeline_error = str(error)
                st.session_state.pipeline_message = None

            st.rerun()

    with stop_column:
        if st.button(
            "Stop",
            use_container_width=True,
        ):
            try:
                messages = stop_pipeline(stop_docker=True)
                st.session_state.pipeline_message = (
                    " · ".join(messages)
                )
                st.session_state.pipeline_error = None
            except Exception as error:
                st.session_state.pipeline_error = str(error)
                st.session_state.pipeline_message = None

            st.rerun()

    if st.session_state.pipeline_message:
        st.success(st.session_state.pipeline_message)

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)

    st.divider()

    auto_refresh = st.checkbox(
        "Auto-refresh dashboard",
        value=True,
    )

    refresh_seconds = st.selectbox(
        "Refresh interval",
        options=[5, 10, 15, 30, 60],
        index=1,
        format_func=lambda seconds: f"{seconds} seconds",
        disabled=not auto_refresh,
    )

    if auto_refresh:
        st_autorefresh(
            interval=refresh_seconds * 1000,
            key="wikipulse_auto_refresh",
        )

    if st.button(
        "Refresh now",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    for process_name, spec in PYTHON_PROCESSES.items():
        with st.expander(f"{spec['label']} log"):
            st.code(read_log(process_name), language="text")

# dashboard/app.py -> project root -> data/raw_wikipedia_edits
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw_wikipedia_edits"


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def get_data_signature():
    """
    Return information about the current Parquet files.

    When Spark creates a new file, this signature changes,
    causing Streamlit to reload the dataset.
    """
    if not DATA_DIRECTORY.exists():
        return ()

    return tuple(
        (
            file.name,
            file.stat().st_size,
            file.stat().st_mtime_ns,
        )
        for file in sorted(DATA_DIRECTORY.glob("*.parquet"))
    )


@st.cache_data(show_spinner="Loading Wikipedia edits...", max_entries=1)
def load_data(data_signature):
    if not data_signature:
        return pd.DataFrame()

    # Stopping/killing the Spark writer mid-batch can leave a 0-byte
    # part file behind; pyarrow refuses to open those, which would
    # otherwise crash the whole dashboard on the next refresh.
    parquet_files = [
        str(DATA_DIRECTORY / name)
        for name, size, _ in data_signature
        if size > 0
    ]

    if not parquet_files:
        return pd.DataFrame()

    df = pd.read_parquet(parquet_files)

    # Convert Wikimedia's Unix timtamp into a readable datetime.
    if "timestamp" in df.columns:
        df["event_time"] = pd.to_datetime(
            df["timestamp"],
            unit="s",
            errors="coerce",
            utc=True,
        )

    if "kafka_timestamp" in df.columns:
        df["kafka_timestamp"] = pd.to_datetime(
            df["kafka_timestamp"],
            errors="coerce",
            utc=True,
        )

    if "bot" in df.columns:
        df["bot"] = df["bot"].fillna(False).astype(bool)

    if "minor" in df.columns:
        df["minor"] = df["minor"].fillna(False).astype(bool)

    if "byte_change" in df.columns:
        df["byte_change"] = pd.to_numeric(
            df["byte_change"],
            errors="coerce",
        ).fillna(0)

    return df


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🌐 WikiPulse")

st.caption(
    "Live English Wikipedia edit activity with anomaly detection and "
    "GDELT news correlation: Wikimedia + GDELT → Kafka → Spark / "
    "scikit-learn / Qdrant → this dashboard"
)


data_signature = get_data_signature()

try:
    df = load_data(data_signature)
except Exception as error:
    st.error("The dashboard could not read the Parquet dataset.")
    st.exception(error)
    df = pd.DataFrame()


if df.empty:
    st.warning(
        "No Wikipedia edit data is available yet. Click Start Pipeline "
        "in the sidebar, then give Spark a few seconds to write its "
        "first Parquet batch."
    )


# ---------------------------------------------------------
# Summary metrics and edit charts (need Parquet data)
# ---------------------------------------------------------

if not df.empty:
    total_edits = len(df)
    bot_edits = int(df["bot"].sum())
    human_edits = total_edits - bot_edits
    minor_edits = int(df["minor"].sum())
    total_byte_change = int(df["byte_change"].sum())

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

    metric_1.metric("Total edits", f"{total_edits:,}")
    metric_2.metric("Human edits", f"{human_edits:,}")
    metric_3.metric("Bot edits", f"{bot_edits:,}")
    metric_4.metric("Minor edits", f"{minor_edits:,}")
    metric_5.metric(
        "Net byte change",
        f"{total_byte_change:+,}",
    )

    st.subheader("Edit activity over time")

    valid_times = df.dropna(subset=["event_time"]).copy()

    if not valid_times.empty:
        edits_over_time = (
            valid_times
            .set_index("event_time")
            .resample("1min")
            .size()
            .rename("edits")
        )

        st.line_chart(
            edits_over_time,
            y="edits",
            x_label="Time",
            y_label="Number of edits",
        )
    else:
        st.info("No valid event timestamps are currently available.")

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Most-edited pages")

        top_pages = (
            df["page_title"]
            .dropna()
            .value_counts()
            .head(10)
            .rename_axis("Page")
            .rename("Edits")
        )

        st.bar_chart(
            top_pages,
            horizontal=True,
            x_label="Number of edits",
            y_label="Page",
        )

    with right_column:
        st.subheader("Most-active users")

        top_users = (
            df["user"]
            .dropna()
            .value_counts()
            .head(10)
            .rename_axis("User")
            .rename("Edits")
        )

        st.bar_chart(
            top_users,
            horizontal=True,
            x_label="Number of edits",
            y_label="User",
        )


# ---------------------------------------------------------
# Anomaly alerts (wikipedia-anomalies topic)
# ---------------------------------------------------------

st.subheader("Anomaly alerts")

scorer_status = status.get("Anomaly scorer (Spark)", {})

anomaly_messages = [
    message
    for message in read_topic_messages("wikipedia-anomalies")
    if "page_title" in message and "anomaly_score" in message
]

if anomaly_messages:
    anomaly_df = pd.DataFrame(anomaly_messages)
    if "recent_comments" not in anomaly_df.columns:
        anomaly_df["recent_comments"] = ""

    anomaly_df["window_end_ts"] = pd.to_datetime(
        anomaly_df.get("window_end"), errors="coerce"
    )
    cutoff = pd.Timestamp.now() - pd.Timedelta(minutes=ANOMALY_RECENCY_MINUTES)
    recent_anomaly_df = anomaly_df[anomaly_df["window_end_ts"] >= cutoff]
    recent_anomaly_df = _dedup_latest_per_page(recent_anomaly_df, "window_end_ts")
else:
    anomaly_df = pd.DataFrame()
    recent_anomaly_df = pd.DataFrame()

if not recent_anomaly_df.empty:
    # Most-recently-detected anomaly first, so the table reads like a live
    # feed rather than a fixed leaderboard.
    recent_anomaly_df = recent_anomaly_df.sort_values(by="window_end_ts", ascending=False)
    recent_anomaly_df["wikipedia_url"] = recent_anomaly_df["page_title"].apply(
        wikipedia_page_url
    )
    recent_anomaly_df["detected_at"] = recent_anomaly_df["window_end_ts"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    st.dataframe(
        recent_anomaly_df[
            [
                "page_title",
                "wikipedia_url",
                "detected_at",
                "anomaly_score",
                "edit_count",
                "unique_editors",
                "total_byte_changes",
                "recent_comments",
            ]
        ].rename(
            columns={
                "page_title": "Page",
                "wikipedia_url": "Wikipedia page",
                "detected_at": "Detected at",
                "anomaly_score": "Anomaly score",
                "edit_count": "Edits",
                "unique_editors": "Editors",
                "total_byte_changes": "Bytes changed",
                "recent_comments": "Recent edit summaries",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Wikipedia page": st.column_config.LinkColumn(
                "Wikipedia page", display_text="Open page"
            ),
            "Recent edit summaries": st.column_config.TextColumn(
                "Recent edit summaries", width="large"
            ),
            "Anomaly score": st.column_config.NumberColumn(
                "Anomaly score", format="%.3f"
            ),
        },
    )
    st.caption(
        f"{len(recent_anomaly_df)} distinct page(s) anomalous in the last "
        f"{ANOMALY_RECENCY_MINUTES} minutes (deduplicated to each page's "
        f"latest window; {len(anomaly_df)} total anomaly messages received)."
    )
elif not anomaly_df.empty:
    st.info(
        f"No anomalies in the last {ANOMALY_RECENCY_MINUTES} minutes -- "
        f"{len(anomaly_df)} older anomaly message(s) exist in the topic "
        "but have aged out of this view."
    )
elif scorer_status.get("state") == "blocked":
    st.warning(
        "Anomaly detection is disabled because the model has not been "
        "trained. Use the Train model button in the sidebar, then "
        "Start Pipeline."
    )
elif scorer_status.get("state") == "running":
    st.info(
        "The anomaly scorer is running and waiting for data. It scores "
        "15-minute windows, so the first alerts can take a while — and "
        "only windows that actually look anomalous are published."
    )
else:
    st.info(
        "The anomaly scorer is not running. Click Start Pipeline in "
        "the sidebar to launch it."
    )

# ---------------------------------------------------------
# Anomalies verified against real news (verified-events topic)
# ---------------------------------------------------------

st.subheader("Anomalies verified against real news")

matcher_status = status.get("News matcher (Qdrant)", {})

verified_messages = [
    message
    for message in read_topic_messages("verified-events")
    if "page_title" in message and "matched" in message
]

if verified_messages:
    verified_df = pd.DataFrame(
        flatten_verified_event(message)
        for message in verified_messages
    )

    # flatten_verified_event() keeps a fixed field set; recent_comments
    # rides along from the raw messages (row order is preserved 1:1).
    verified_df["recent_comments"] = [
        message.get("recent_comments") or ""
        for message in verified_messages
    ]

    # The same page/window can be re-evaluated as windows update;
    # keep only the most recent verdict for each.
    verified_df = verified_df.drop_duplicates(
        subset=["page_title", "window_start"], keep="last"
    )

    verified_df["wikipedia_url"] = verified_df["page_title"].apply(
        wikipedia_page_url
    )

    # Verified matches first, then higher similarity first
    # (rows with no candidate article sort last).
    verified_df = verified_df.sort_values(
        by=["matched", "similarity_score"],
        ascending=[False, False],
    )

    # Each row is an aggregated page/window of Wikipedia activity,
    # not one individual edit.
    verified_df["window"] = [
        format_window(start, end)
        for start, end in zip(
            verified_df["window_start"], verified_df["window_end"]
        )
    ]
    verified_df["news_seen_at"] = pd.to_datetime(
        verified_df["news_seen_at"], errors="coerce", utc=True
    )

    total_evaluated = len(verified_df)
    verified_count = int(verified_df["matched"].sum())
    below_threshold_count = int(
        (~verified_df["matched"] & verified_df["similarity_score"].notna()).sum()
    )
    no_candidate_count = int(verified_df["similarity_score"].isna().sum())
    average_verified_similarity = (
        verified_df.loc[verified_df["matched"], "similarity_score"].mean()
        if verified_count
        else None
    )

    summary_1, summary_2, summary_3, summary_4, summary_5 = st.columns(5)
    summary_1.metric("Anomalies evaluated", f"{total_evaluated:,}")
    summary_2.metric("Verified", f"{verified_count:,}")
    summary_3.metric("Below threshold", f"{below_threshold_count:,}")
    summary_4.metric("No candidate article", f"{no_candidate_count:,}")
    summary_5.metric(
        "Avg verified similarity",
        f"{average_verified_similarity:.3f}"
        if average_verified_similarity is not None
        else "—",
    )

    st.dataframe(
        verified_df[
            [
                "page_title",
                "wikipedia_url",
                "window",
                "edit_count",
                "unique_editors",
                "total_byte_changes",
                "anomaly_score",
                "recent_comments",
                "news_title",
                "news_domain",
                "news_language",
                "news_seen_at",
                "similarity_score",
                "similarity_threshold",
                "status",
                "news_url",
            ]
        ].rename(
            columns={
                "page_title": "Wikipedia page",
                "wikipedia_url": "Wikipedia link",
                "window": "Activity window (UTC)",
                "edit_count": "Edits",
                "unique_editors": "Editors",
                "total_byte_changes": "Bytes changed",
                "anomaly_score": "Anomaly score",
                "recent_comments": "Recent edit summaries",
                "news_title": "Matched news",
                "news_domain": "Source",
                "news_language": "Language",
                "news_seen_at": "News crawled at",
                "similarity_score": "Similarity",
                "similarity_threshold": "Threshold",
                "status": "Verification",
                "news_url": "News URL",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Wikipedia link": st.column_config.LinkColumn(
                "Wikipedia link", display_text="Open page"
            ),
            "Recent edit summaries": st.column_config.TextColumn(
                "Recent edit summaries", width="medium"
            ),
            "Matched news": st.column_config.TextColumn(
                "Matched news", width="medium"
            ),
            "News URL": st.column_config.LinkColumn(
                "News URL", display_text="Open article"
            ),
            "Similarity": st.column_config.NumberColumn(
                "Similarity", format="%.3f"
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold", format="%.2f"
            ),
            "Anomaly score": st.column_config.NumberColumn(
                "Anomaly score", format="%.3f"
            ),
            "News crawled at": st.column_config.DatetimeColumn(
                "News crawled at", format="MMM DD HH:mm"
            ),
        },
    )

    st.caption(
        "Each row is one anomalous window of Wikipedia page activity "
        "compared against the closest recent GDELT article. Similarity "
        "is the cosine similarity of the two title embeddings "
        "(−1 to 1, higher is closer); the matcher marks a row Verified "
        "when similarity ≥ its threshold. Below-threshold rows still "
        "show the closest candidate to explain why they weren't "
        "verified. Older records (before the payload gained editor/"
        "byte/threshold fields) show blanks in those columns."
    )
elif matcher_status.get("state") == "running":
    st.info(
        "The news matcher is running and waiting for anomalies to "
        "evaluate. Verdicts appear here as soon as the scorer flags "
        "its first anomaly."
    )
else:
    st.info(
        "The news matcher is not running. Click Start Pipeline in the "
        "sidebar to launch it (it needs the Qdrant container, which "
        "Start Pipeline also brings up)."
    )

# ---------------------------------------------------------
# Recent GDELT news (news-topic)
# ---------------------------------------------------------

st.subheader("Recent GDELT news")

gdelt_status = status.get("GDELT news producer", {})

news_messages = [
    message
    for message in read_topic_messages("news-topic")
    if "title" in message and "url" in message
]

if news_messages:
    news_df = pd.DataFrame(news_messages)

    news_df["domain"] = news_df["url"].apply(
        lambda url: urlparse(url).netloc if isinstance(url, str) else None
    )

    for time_column in ("gdelt_seen_at", "observed_at"):
        if time_column in news_df.columns:
            news_df[time_column] = pd.to_datetime(
                news_df[time_column],
                errors="coerce",
                utc=True,
            )

    if "observed_at" in news_df.columns:
        news_df = news_df.sort_values("observed_at", ascending=False)

    display_columns = [
        column
        for column in (
            "title",
            "domain",
            "language",
            "gdelt_seen_at",
            "observed_at",
            "url",
        )
        if column in news_df.columns
    ]

    st.dataframe(
        news_df.head(50)[display_columns].rename(
            columns={
                "title": "Title",
                "domain": "Source",
                "language": "Language",
                "gdelt_seen_at": "GDELT crawled at",
                "observed_at": "Ingested at",
                "url": "URL",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "URL": st.column_config.LinkColumn("URL", display_text="Open article"),
        },
    )

    st.caption(
        f"Showing the 50 most recently ingested of {len(news_df)} "
        "articles read from news-topic. \"GDELT crawled at\" is when "
        "GDELT's crawler saw the URL — not the article's original "
        "publication date."
    )
elif gdelt_status.get("state") == "running":
    st.info(
        "The GDELT producer is running and waiting for data. GDELT "
        "publishes in roughly 15-minute heartbeat windows, so the "
        "first articles can take several minutes to arrive."
    )
else:
    st.info(
        "The GDELT news producer is not running. Click Start Pipeline "
        "in the sidebar to launch it."
    )

# ---------------------------------------------------------
# Largest changes and latest edits (need Parquet data)
# ---------------------------------------------------------

if not df.empty:
    st.subheader("Largest content changes")

    positive_column, negative_column = st.columns(2)

    change_columns = [
        "event_time",
        "page_title",
        "user",
        "bot",
        "byte_change",
    ]

    with positive_column:
        st.markdown("#### Largest additions")

        largest_additions = (
            df[df["byte_change"] > 0]
            .nlargest(10, "byte_change")[change_columns]
            .rename(
                columns={
                    "event_time": "Time",
                    "page_title": "Page",
                    "user": "User",
                    "bot": "Bot",
                    "byte_change": "Bytes",
                }
            )
        )

        st.dataframe(
            largest_additions,
            hide_index=True,
            use_container_width=True,
        )

    with negative_column:
        st.markdown("#### Largest removals")

        largest_removals = (
            df[df["byte_change"] < 0]
            .nsmallest(10, "byte_change")[change_columns]
            .rename(
                columns={
                    "event_time": "Time",
                    "page_title": "Page",
                    "user": "User",
                    "bot": "Bot",
                    "byte_change": "Bytes",
                }
            )
        )

        st.dataframe(
            largest_removals,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Latest Wikipedia edits")

    latest_columns = [
        "event_time",
        "page_title",
        "user",
        "event_type",
        "bot",
        "minor",
        "byte_change",
        "partition",
        "offset",
    ]

    latest_edits = (
        df.sort_values("event_time", ascending=False)
        .head(100)[latest_columns]
        .rename(
            columns={
                "event_time": "Time",
                "page_title": "Page",
                "user": "User",
                "event_type": "Type",
                "bot": "Bot",
                "minor": "Minor",
                "byte_change": "Byte change",
                "partition": "Kafka partition",
                "offset": "Kafka offset",
            }
        )
    )

    st.dataframe(
        latest_edits,
        hide_index=True,
        use_container_width=True,
        height=500,
    )

    with st.expander("Dataset information"):
        parquet_file_count = len(data_signature)

        st.write(f"Parquet files: **{parquet_file_count:,}**")
        st.write(f"Records loaded: **{len(df):,}**")
        st.write(f"Dataset location: `{DATA_DIRECTORY}`")

        st.markdown("**Available columns:**")
        st.code("\n".join(df.columns))


st.caption(
    "This dashboard displays the raw-data layer plus anomaly scoring "
    "and news-correlation results from the intelligence phase."
)
