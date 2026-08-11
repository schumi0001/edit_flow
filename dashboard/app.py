import os
from collections import deque
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import streamlit as st

from pipeline_manager import (
    get_last_failure,
    get_pipeline_status,
    is_model_trained,
    start_pipeline,
    stop_pipeline,
    train_model,
)
from anomaly_utils import flatten_verified_event, parse_anomaly_message

# Anomaly windows are 15 minutes wide and slide every 5 minutes, so the
# same page re-emits often. Alerts use all messages read from the topic,
# dedupe to the latest window per page, then show at most this many rows
# (search filters the full unique-page set before the cap).
ANOMALY_TABLE_MAX_ROWS = int(os.environ.get("ANOMALY_TABLE_MAX_ROWS", "20"))

# The live edit stream (producer/wikipedia_producer.py) is hardcoded to
# English Wikipedia's recent-changes SSE feed, so page_title always
# resolves against en.wikipedia.org.
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"

# All dashboard timestamps are shown in US Eastern (EST/EDT). Pipeline
# / Kafka payloads stay UTC; we only convert at display time.
DISPLAY_TZ = "America/New_York"


def wikipedia_page_url(page_title):
    if not page_title:
        return None
    return WIKIPEDIA_BASE_URL + quote(str(page_title).replace(" ", "_"))


def to_eastern(values):
    """Treat naive datetimes as UTC and convert to US Eastern for display."""
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(ts, pd.Series):
        return ts.dt.tz_convert(DISPLAY_TZ)
    if pd.isna(ts):
        return ts
    return ts.tz_convert(DISPLAY_TZ)


def format_eastern_timestamp(values, fmt="%Y-%m-%d %H:%M ET"):
    """Format UTC/naive timestamps as Eastern Time strings for tables."""
    eastern = to_eastern(values)
    if isinstance(eastern, pd.Series):
        return eastern.dt.strftime(fmt)
    if pd.isna(eastern):
        return None
    return eastern.strftime(fmt)


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

@st.cache_data(ttl=20, show_spinner=False)
def read_topic_messages(topic, limit=400, timeout_ms=3000, keep_matched=False):
    """Read up to the last `limit` JSON messages from a Kafka topic.

    Returns [] when Kafka is unreachable or the topic is empty. A bounded
    deque keeps memory flat even if the topic grows large.

    When ``keep_matched`` is True (used for anomaly-news-verdicts), every
    message with ``matched: true`` is retained even if it falls outside the
    trailing ``limit`` window -- confirmed news matches are rare and must
    not be dropped just because many unmatched evaluations arrived later.
    """
    messages = deque(maxlen=limit)
    matched_messages = []

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
            if parsed is None:
                continue
            messages.append(parsed)
            if keep_matched and parsed.get("matched"):
                matched_messages.append(parsed)

        consumer.close()
    except Exception:
        return []

    if not keep_matched or not matched_messages:
        return list(messages)

    # Union matched (all time in topic scan) with the recent window.
    merged = {}
    for record in list(matched_messages) + list(messages):
        key = (
            record.get("page_title"),
            record.get("window_start"),
            record.get("window_end"),
            record.get("evaluated_at"),
            record.get("similarity_score"),
            record.get("matched"),
        )
        merged[key] = record
    return list(merged.values())


def format_window(start, end):
    """Render a page/window pair in Eastern Time, e.g. 'Aug 10 10:30 – 10:45 ET'."""
    try:
        start_time = to_eastern(start)
        end_time = to_eastern(end)
        if pd.isna(start_time) or pd.isna(end_time):
            raise ValueError
        return f"{start_time:%b %d %H:%M} – {end_time:%H:%M} ET"
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

    # Fragment-based refresh updates tables without a full-page rerun
    # (avoids the gray flash from streamlit-autorefresh).
    auto_refresh = st.checkbox(
        "Auto-refresh live tables",
        value=True,
        help=(
            "Re-reads Kafka/Parquet in place every interval. "
            "Uses a Streamlit fragment so the rest of the page "
            "does not gray out."
        ),
    )

    refresh_seconds = st.selectbox(
        "Refresh interval",
        options=[15, 30, 60, 120],
        index=1,
        format_func=lambda seconds: f"{seconds} seconds",
        disabled=not auto_refresh,
        help=(
            "30s is enough for anomaly windows (5-minute slides) "
            "without constant UI churn."
        ),
    )

    if st.button(
        "Refresh now",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

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

    # Convert Wikimedia's Unix timestamp into Eastern Time for display.
    if "timestamp" in df.columns:
        df["event_time"] = to_eastern(
            pd.to_datetime(
                df["timestamp"],
                unit="s",
                errors="coerce",
                utc=True,
            )
        )

    if "kafka_timestamp" in df.columns:
        df["kafka_timestamp"] = to_eastern(
            pd.to_datetime(
                df["kafka_timestamp"],
                errors="coerce",
                utc=True,
            )
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

# Only the tables/metrics below auto-refresh. Sidebar controls and the
# page chrome stay put, so interval ticks do not gray out the whole UI.
_run_every = timedelta(seconds=refresh_seconds) if auto_refresh else None


@st.fragment(run_every=_run_every)
def render_live_tables():
    # Re-read pipeline status inside the fragment so empty-state captions
    # stay accurate without a full-page rerun.
    status = get_pipeline_status()

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

        valid_times = df.dropna(subset=["event_time"])

        if not valid_times.empty:
            # This Parquet lake only contains edits captured while *this*
            # pipeline instance was running -- not all of Wikipedia -- so
            # plotting the full min→max span leaves long empty gaps that
            # look like "Wikipedia was quiet." Restrict to the last few
            # hours of captured data, and chart edits-per-minute (not a
            # daily total).
            ACTIVITY_LOOKBACK_HOURS = 6
            latest_event = valid_times["event_time"].max()
            recent_times = valid_times[
                valid_times["event_time"]
                >= latest_event - pd.Timedelta(hours=ACTIVITY_LOOKBACK_HOURS)
            ]

            # Group by minute instead of resample(): resample fills every
            # empty minute across the span and makes the Vega chart laggy.
            edits_over_time = (
                recent_times["event_time"]
                .dt.floor("min")
                .value_counts()
                .sort_index()
                .rename("edits_per_minute")
            )

            st.line_chart(
                edits_over_time,
                y="edits_per_minute",
                x_label="Time (ET)",
                y_label="Edits / minute",
            )
            st.caption(
                f"Edits per minute for the last {ACTIVITY_LOOKBACK_HOURS} hours "
                f"of data this pipeline captured ({len(recent_times):,} edits "
                f"across {len(edits_over_time):,} active minutes) — not a "
                "full English Wikipedia history."
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
        for message in read_topic_messages("wikipedia-anomalies", limit=2000)
        if "page_title" in message and "anomaly_score" in message
    ]

    if anomaly_messages:
        anomaly_df = pd.DataFrame(anomaly_messages)
        if "recent_comments" not in anomaly_df.columns:
            anomaly_df["recent_comments"] = ""

        anomaly_df["window_end_ts"] = pd.to_datetime(
            anomaly_df.get("window_end"), errors="coerce", utc=True
        )
        pages_df = _dedup_latest_per_page(anomaly_df, "window_end_ts")
        pages_df = pages_df.sort_values(by="window_end_ts", ascending=False)
    else:
        anomaly_df = pd.DataFrame()
        pages_df = pd.DataFrame()

    if not anomaly_df.empty:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Unique pages", f"{len(pages_df):,}")
        metric_b.metric(
            "Anomaly messages read",
            f"{len(anomaly_df):,}",
        )
        metric_c.metric(
            "Showing up to",
            f"{ANOMALY_TABLE_MAX_ROWS}",
        )
        st.caption(
            "News confirmation is in the verified table below — use the "
            "page filter here to look up a specific anomalous page."
        )

    if not pages_df.empty:
        # Clear must happen before the text_input is instantiated —
        # Streamlit forbids mutating a widget key after the widget runs.
        if st.session_state.pop("anomaly_page_filter_clear", False):
            st.session_state.anomaly_page_filter = ""

        if "anomaly_page_filter" not in st.session_state:
            st.session_state.anomaly_page_filter = ""

        st.markdown("Filter by page title")
        filter_col, clear_col = st.columns([4, 1], vertical_alignment="bottom")
        with filter_col:
            page_filter = st.text_input(
                "Filter by page title",
                placeholder="e.g. Colombia earthquake",
                key="anomaly_page_filter",
                label_visibility="collapsed",
            ).strip()
        with clear_col:
            if st.button(
                "Clear",
                key="anomaly_page_filter_clear_btn",
                use_container_width=True,
                disabled=not bool(st.session_state.get("anomaly_page_filter")),
            ):
                st.session_state.anomaly_page_filter_clear = True
                st.rerun()

        filtered_df = pages_df
        if page_filter:
            filtered_df = pages_df[
                pages_df["page_title"]
                .astype(str)
                .str.contains(page_filter, case=False, na=False)
            ]

        if filtered_df.empty:
            st.info(f'No pages match "{page_filter}".')
        else:
            display_df = filtered_df.head(ANOMALY_TABLE_MAX_ROWS).copy()
            display_df["wikipedia_url"] = display_df["page_title"].apply(
                wikipedia_page_url
            )
            display_df["detected_at"] = to_eastern(
                display_df["window_end_ts"]
            ).dt.strftime("%Y-%m-%d %H:%M ET")

            st.dataframe(
                display_df[
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
                        "detected_at": "Detected at (ET)",
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
                f"Showing {len(display_df)} of {len(filtered_df)} "
                f"{'filtered ' if page_filter else ''}"
                f"unique page(s) "
                f"({len(anomaly_df)} total anomaly messages read). "
                "Use the filter to find a specific page (e.g. one listed "
                "under verified matches below)."
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
    # Anomalies verified against real news (anomaly-news-verdicts)
    # ---------------------------------------------------------

    st.subheader("Anomalies verified against real news")

    matcher_status = status.get("News matcher (Qdrant)", {})

    verified_messages = [
        message
        for message in read_topic_messages(
            "anomaly-news-verdicts",
            limit=5000,
            keep_matched=True,
        )
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

        # Topic stores every verdict (matched true/false). Metrics use the
        # full evaluation set; the table below shows confirmed matches only.
        total_evaluated = len(verified_df)
        verified_count = int(verified_df["matched"].sum())
        below_threshold_count = int(
            (
                ~verified_df["matched"]
                & verified_df["similarity_score"].notna()
            ).sum()
        )
        no_candidate_count = int(verified_df["similarity_score"].isna().sum())
        confirmation_rate = (
            verified_count / total_evaluated if total_evaluated else 0.0
        )
        average_verified_similarity = (
            verified_df.loc[verified_df["matched"], "similarity_score"].mean()
            if verified_count
            else None
        )

        summary_1, summary_2, summary_3, summary_4, summary_5 = st.columns(5)
        summary_1.metric("Anomalies evaluated", f"{total_evaluated:,}")
        summary_2.metric("Verified", f"{verified_count:,}")
        summary_3.metric(
            "% confirmed",
            f"{100 * confirmation_rate:.1f}%",
        )
        summary_4.metric("Below threshold", f"{below_threshold_count:,}")
        summary_5.metric(
            "Avg verified similarity",
            f"{average_verified_similarity:.3f}"
            if average_verified_similarity is not None
            else "—",
        )
        st.caption(
            f"Metrics cover every verdict on anomaly-news-verdicts "
            f"({no_candidate_count:,} had no news candidate at all). "
            "The table below lists confirmed matches only."
        )

        matched_df = verified_df[verified_df["matched"]].copy()

        if not matched_df.empty:
            # Sliding 15-minute / 5-minute windows re-emit the same page
            # repeatedly as edits accumulate; each emission is matched
            # independently (often to the same news URL with a slightly
            # different similarity). Keep one row per page: the latest
            # window_end.
            matched_df["window_end_ts"] = pd.to_datetime(
                matched_df["window_end"], errors="coerce", utc=True
            )
            before_dedup = len(matched_df)
            matched_df = _dedup_latest_per_page(matched_df, "window_end_ts")

            matched_df["wikipedia_url"] = matched_df["page_title"].apply(
                wikipedia_page_url
            )
            matched_df["window"] = [
                format_window(start, end)
                for start, end in zip(
                    matched_df["window_start"], matched_df["window_end"]
                )
            ]
            matched_df = matched_df.sort_values(
                by="window_end_ts", ascending=False
            )

            st.dataframe(
                matched_df[
                    [
                        "page_title",
                        "wikipedia_url",
                        "news_url",
                        "news_title",
                        "news_domain",
                        "similarity_score",
                        "anomaly_score",
                        "window",
                        "edit_count",
                        "unique_editors",
                        "total_byte_changes",
                        "recent_comments",
                    ]
                ].rename(
                    columns={
                        "page_title": "Wikipedia page",
                        "wikipedia_url": "Wikipedia link",
                        "news_url": "News URL",
                        "news_title": "Matched news",
                        "news_domain": "Source",
                        "similarity_score": "Similarity",
                        "anomaly_score": "Anomaly score",
                        "window": "Activity window (ET)",
                        "edit_count": "Edits",
                        "unique_editors": "Editors",
                        "total_byte_changes": "Bytes changed",
                        "recent_comments": "Recent edit summaries",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Wikipedia link": st.column_config.LinkColumn(
                        "Wikipedia link", display_text="Open page"
                    ),
                    "News URL": st.column_config.LinkColumn(
                        "News URL", display_text="Open article"
                    ),
                    "Matched news": st.column_config.TextColumn(
                        "Matched news", width="medium"
                    ),
                    "Recent edit summaries": st.column_config.TextColumn(
                        "Recent edit summaries", width="medium"
                    ),
                    "Similarity": st.column_config.NumberColumn(
                        "Similarity", format="%.3f"
                    ),
                    "Anomaly score": st.column_config.NumberColumn(
                        "Anomaly score", format="%.3f"
                    ),
                },
            )

            st.caption(
                f"{len(matched_df)} confirmed news match(es) "
                f"(latest window per page; {before_dedup} matched messages "
                f"before dedupe, {len(verified_df)} anomalies evaluated). "
                "Each row is an anomalous Wikipedia window whose closest "
                "GDELT article cleared the similarity threshold."
            )
        elif matcher_status.get("state") == "running":
            st.info(
                f"{len(verified_df)} anomal(ies) evaluated so far, but none "
                "have cleared the news-similarity threshold yet."
            )
        else:
            st.info(
                f"{len(verified_df)} anomal(ies) were evaluated, but none "
                "matched news. Start Pipeline to keep evaluating new alerts."
            )
    elif matcher_status.get("state") == "running":
        st.info(
            "The news matcher is running and waiting for anomalies to "
            "evaluate. Confirmed matches appear here when similarity "
            "clears the threshold."
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
                news_df[time_column] = to_eastern(news_df[time_column])

        if "observed_at" in news_df.columns:
            news_df = news_df.sort_values("observed_at", ascending=False)

        display_news = news_df.head(50).copy()
        for time_column in ("gdelt_seen_at", "observed_at"):
            if time_column in display_news.columns:
                display_news[time_column] = display_news[time_column].dt.strftime(
                    "%Y-%m-%d %H:%M ET"
                )

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
            if column in display_news.columns
        ]

        st.dataframe(
            display_news[display_columns].rename(
                columns={
                    "title": "Title",
                    "domain": "Source",
                    "language": "Language",
                    "gdelt_seen_at": "GDELT crawled at (ET)",
                    "observed_at": "Ingested at (ET)",
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
            "publication date. Times are US Eastern."
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
                .copy()
            )
            largest_additions["event_time"] = format_eastern_timestamp(
                largest_additions["event_time"]
            )
            largest_additions = largest_additions.rename(
                columns={
                    "event_time": "Time (ET)",
                    "page_title": "Page",
                    "user": "User",
                    "bot": "Bot",
                    "byte_change": "Bytes",
                }
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
                .copy()
            )
            largest_removals["event_time"] = format_eastern_timestamp(
                largest_removals["event_time"]
            )
            largest_removals = largest_removals.rename(
                columns={
                    "event_time": "Time (ET)",
                    "page_title": "Page",
                    "user": "User",
                    "bot": "Bot",
                    "byte_change": "Bytes",
                }
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
            .copy()
        )
        latest_edits["event_time"] = format_eastern_timestamp(
            latest_edits["event_time"]
        )
        latest_edits = latest_edits.rename(
            columns={
                "event_time": "Time (ET)",
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

render_live_tables()
