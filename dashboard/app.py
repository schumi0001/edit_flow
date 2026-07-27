from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from pipeline_manager import (
    get_pipeline_status,
    read_log,
    start_pipeline,
    stop_pipeline,
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="WikiPulse Dashboard",
    page_icon="🌐",
    layout="wide",
)

# ---------------------------------------------------------
# Pipeline control panel
# ---------------------------------------------------------

if "pipeline_message" not in st.session_state:
    st.session_state.pipeline_message = None

if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None


with st.sidebar:
    st.header("Pipeline Control")

    status = get_pipeline_status()

    for component, running in status.items():
        icon = "🟢" if running else "🔴"
        state = "Running" if running else "Stopped"
        st.write(f"{icon} **{component}:** {state}")

    start_column, stop_column = st.columns(2)

    with start_column:
        if st.button(
            "Start",
            type="primary",
            use_container_width=True,
        ):
            try:
                messages = start_pipeline()
                st.session_state.pipeline_message = (
                    " · ".join(messages)
                )
                st.session_state.pipeline_error = None
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
                messages = stop_pipeline(stop_kafka=True)
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

    with st.expander("Spark log"):
        st.code(read_log("spark"), language="text")

    with st.expander("Producer log"):
        st.code(read_log("producer"), language="text")

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


@st.cache_data(show_spinner="Loading Wikipedia edits...")
def load_data(data_signature):
    if not data_signature:
        return pd.DataFrame()

    df = pd.read_parquet(DATA_DIRECTORY)

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
    "Live English Wikipedia edit activity collected through "
    "Wikimedia → Kafka → Spark → Parquet"
)

if "refresh_message" not in st.session_state:
    st.session_state.refresh_message = False

# if st.button("Refresh data", type="primary"):
#     st.cache_data.clear()
#     st.session_state.refresh_message = True
#     st.rerun()

if st.session_state.refresh_message:
    st.success("Data refreshed successfully.")
    st.session_state.refresh_message = False


data_signature = get_data_signature()

try:
    df = load_data(data_signature)
except Exception as error:
    st.error("The dashboard could not read the Parquet dataset.")
    st.exception(error)
    st.stop()


if df.empty:
    st.warning(
        "No Wikipedia edit data is available yet. Start the producer "
        "and Spark stream, then refresh this page."
    )
    st.code(
        "python producer/wikipedia_producer.py\n\n"
        "spark-submit \\\n"
        "  --packages "
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 \\\n"
        "  spark/wiki_stream.py"
    )
    st.stop()


# ---------------------------------------------------
# Summary metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Edit activity over time
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Top pages and users
# ---------------------------------------------------------

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
# Largest changes
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Latest edits table
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

with st.expander("Dataset information"):
    parquet_file_count = len(data_signature)

    st.write(f"Parquet files: **{parquet_file_count:,}**")
    st.write(f"Records loaded: **{total_edits:,}**")
    st.write(f"Dataset location: `{DATA_DIRECTORY}`")

    st.markdown("**Available columns:**")
    st.code("\n".join(df.columns))


st.caption(
    "This dashboard currently displays the raw-data layer. "
    "Trend detection, anomaly scoring, and news correlation will "
    "be added in the intelligence phase."
)
