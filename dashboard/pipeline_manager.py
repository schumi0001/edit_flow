import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRECTORY = PROJECT_ROOT / ".runtime"
LOG_DIRECTORY = RUNTIME_DIRECTORY / "logs"

RUNTIME_DIRECTORY.mkdir(exist_ok=True)
LOG_DIRECTORY.mkdir(exist_ok=True)

KAFKA_HOST = "localhost"
KAFKA_PORT = 9092
QDRANT_PORT = 6333

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_detector.joblib"

# Every Python process the dashboard manages. The "spark" and "producer"
# names are kept from the original pipeline so pid files written by earlier
# dashboard versions keep being recognized (no duplicate processes after an
# upgrade). Order matters: it is the start order (upstream Spark consumers
# first, then producers, then downstream matchers).
PYTHON_PROCESSES = {
    "spark": {
        "label": "Raw edit archiver (Spark)",
        "script": PROJECT_ROOT / "spark" / "wiki_stream.py",
    },
    "inference": {
        "label": "Anomaly scorer (Spark)",
        "script": PROJECT_ROOT / "spark" / "ml_inference_stream.py",
    },
    "producer": {
        "label": "Wikipedia producer",
        "script": PROJECT_ROOT / "producer" / "wikipedia_producer.py",
    },
    "gdelt": {
        "label": "GDELT news producer",
        "script": PROJECT_ROOT / "producer" / "gdelt_producer.py",
    },
    "matcher": {
        "label": "News matcher (Qdrant)",
        "script": PROJECT_ROOT / "vectordb" / "match_events.py",
    },
}


def _pid_file(process_name):
    return RUNTIME_DIRECTORY / f"{process_name}.json"


def _log_file(process_name):
    return LOG_DIRECTORY / f"{process_name}.log"


def _exit_file(process_name):
    return RUNTIME_DIRECTORY / f"{process_name}_last_exit.json"


def _read_process_info(process_name):
    path = _pid_file(process_name)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_process_running(process_name):
    info = _read_process_info(process_name)

    if not info:
        return False

    pid = info.get("pid")
    command = info.get("command") or []
    script_path = command[-1] if command else None

    try:
        process = psutil.Process(pid)

        if not process.is_running():
            return False

        if process.status() == psutil.STATUS_ZOMBIE:
            return False

        # Prevent accidentally treating a reused PID as our process by
        # checking the launched script path is still part of its cmdline
        # (works even for launchers like spark-submit that exec into a
        # different binary under the same PID). We deliberately don't
        # compare create_time here: this environment's clock drifts by
        # several seconds within moments of process start, which made
        # that comparison fail almost immediately and caused the pipeline
        # to spawn duplicate producer/Spark processes on every restart.
        return script_path is not None and any(
            script_path in arg for arg in process.cmdline()
        )

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _record_unexpected_exit(process_name, pid):
    exit_path = _exit_file(process_name)

    try:
        existing = json.loads(exit_path.read_text())
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        existing = None

    # Already recorded this same death; avoid re-reading a
    # multi-megabyte log file on every status poll.
    if existing and existing.get("pid") == pid:
        return

    log_path = _log_file(process_name)
    log_tail = ""

    if log_path.exists():
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            log_tail = "\n".join(lines[-40:])
        except OSError:
            log_tail = ""

    exit_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "detected_at": time.time(),
                "log_tail": log_tail,
            },
            indent=2,
        )
    )


def get_last_failure(process_name):
    exit_path = _exit_file(process_name)

    if not exit_path.exists():
        return None

    try:
        return json.loads(exit_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _start_process(process_name, command):
    if is_process_running(process_name):
        return False

    log_path = _log_file(process_name)

    with log_path.open("a") as log:
        log.write(
            f"\n\nStarting {process_name}: "
            f"{' '.join(map(str, command))}\n"
        )
        log.flush()

        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
            start_new_session=True,
        )

    time.sleep(1)

    if process.poll() is not None:
        raise RuntimeError(
            f"{process_name} stopped during startup. "
            f"Check {log_path}."
        )

    process_info = {
        "pid": process.pid,
        "create_time": psutil.Process(process.pid).create_time(),
        "command": [str(item) for item in command],
    }

    _pid_file(process_name).write_text(
        json.dumps(process_info, indent=2)
    )

    # A fresh start means any previously recorded death is no
    # longer the current story for this process.
    _exit_file(process_name).unlink(missing_ok=True)

    return True


def _stop_process(process_name, timeout=10):
    info = _read_process_info(process_name)

    if not info:
        return False

    pid = info.get("pid")
    still_ours = is_process_running(process_name)

    # Remove the pid file before signaling the process, not after.
    # Otherwise a status poll landing in the SIGTERM/wait window below
    # would see "pid file present but process not running" and record
    # this intentional stop as an unexpected exit.
    _pid_file(process_name).unlink(missing_ok=True)

    if still_ours:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)

            try:
                psutil.Process(pid).wait(timeout=timeout)
            except psutil.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)

        except (
            psutil.NoSuchProcess,
            ProcessLookupError,
            PermissionError,
        ):
            pass

    return True


def _running_compose_services():
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "ps",
                "--status",
                "running",
                "--services",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        return set(result.stdout.split())

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return set()


def is_kafka_running():
    return "kafka" in _running_compose_services()


def is_qdrant_running():
    return "qdrant" in _running_compose_services()


def _wait_for_port(port, timeout=60):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection(
                (KAFKA_HOST, port),
                timeout=2,
            ):
                return True
        except OSError:
            time.sleep(1)

    return False


def model_path():
    override = os.environ.get("ANOMALY_MODEL_PATH")
    return Path(override) if override else DEFAULT_MODEL_PATH


def is_model_trained():
    return model_path().exists()


def train_model(timeout=900):
    """Train the anomaly model once, synchronously, via models/train_model.py.

    This is only meant to be triggered explicitly from the UI when the model
    file is missing -- never automatically on pipeline start.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "models" / "train_model.py"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    output = "\n".join(
        part.strip()
        for part in (result.stdout, result.stderr)
        if part and part.strip()
    )

    if result.returncode != 0 or not is_model_trained():
        tail = "\n".join(output.splitlines()[-15:])
        raise RuntimeError(
            f"Model training failed (exit code {result.returncode}):\n{tail}"
        )

    return output


def _check_for_unexpected_exit(process_name):
    if is_process_running(process_name):
        return True

    # A pid file surviving past a dead process means nobody called
    # _stop_process for it -- it died on its own. Capture why.
    info = _read_process_info(process_name)

    if info:
        _record_unexpected_exit(process_name, info.get("pid"))

    return False


def get_pipeline_status():
    """Describe every managed component.

    Returns an ordered mapping of display label -> entry, where an entry is:
      kind    "service" (Docker), "process" (managed Python), or "artifact"
      state   "running" / "stopped" / "blocked" / "ready" / "missing"
      process pid-file name (only for kind == "process")
      detail  optional short human explanation
    """
    services = _running_compose_services()
    model_ready = is_model_trained()

    status = {
        "Kafka": {
            "kind": "service",
            "state": "running" if "kafka" in services else "stopped",
        },
        "Qdrant": {
            "kind": "service",
            "state": "running" if "qdrant" in services else "stopped",
        },
    }

    for name, spec in PYTHON_PROCESSES.items():
        running = _check_for_unexpected_exit(name)
        entry = {
            "kind": "process",
            "process": name,
            "state": "running" if running else "stopped",
        }

        if name == "inference" and not running and not model_ready:
            entry["state"] = "blocked"
            entry["detail"] = "Needs the trained anomaly model"

        if name == "matcher" and not running and "qdrant" not in services:
            entry["detail"] = "Needs the Qdrant container"

        status[spec["label"]] = entry

    status["Anomaly model"] = {
        "kind": "artifact",
        "state": "ready" if model_ready else "missing",
        "detail": (
            None
            if model_ready
            else "Use the Train model button below"
        ),
    }

    return status


def _try_start(name, messages, errors):
    """Start one managed process, reporting instead of raising."""
    label = PYTHON_PROCESSES[name]["label"]
    script = PYTHON_PROCESSES[name]["script"]

    try:
        started = _start_process(
            name,
            [sys.executable, script],
        )
    except RuntimeError as error:
        errors.append(f"{label}: {error}")
        return False

    messages.append(
        f"{label} started"
        if started
        else f"{label} was already running"
    )
    return started


def start_pipeline():
    """Start the full system. Never restarts components already running.

    Returns {"messages": [...], "errors": [...]}. Only a Kafka startup
    failure raises, since nothing downstream can work without it;
    every other component failure is reported in "errors" so the rest
    of the pipeline still comes up.
    """
    messages = []
    errors = []

    services = _running_compose_services()

    if {"kafka", "qdrant"} - services:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "kafka", "qdrant"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or "Kafka/Qdrant containers could not be started."
            )

        messages.append("Kafka and Qdrant containers started")
    else:
        messages.append("Kafka and Qdrant were already running")

    if not _wait_for_port(KAFKA_PORT):
        raise RuntimeError(
            "Kafka started, but port 9092 did not become ready."
        )

    if not _wait_for_port(QDRANT_PORT, timeout=30):
        errors.append(
            "Qdrant port 6333 did not become ready; "
            "news matching may not work."
        )

    # Launched via plain `python`, not spark-submit: the Spark scripts set
    # spark.driver.memory / spark.master themselves via SparkSession.builder,
    # and those only take effect if the driver JVM doesn't already exist
    # when that code runs. spark-submit starts the JVM (with Spark's
    # default 1g heap) before the script executes, silently overriding
    # the script's own memory tuning -- which matters on this
    # memory-constrained setup.
    spark_started = _try_start("spark", messages, errors)

    if is_model_trained():
        inference_started = _try_start("inference", messages, errors)
    else:
        inference_started = False
        messages.append(
            "Anomaly scorer skipped (train the anomaly model first)"
        )

    # Give the Spark streaming queries time to initialize before the
    # producer begins publishing new events.
    if spark_started or inference_started:
        time.sleep(8)

    _try_start("producer", messages, errors)
    _try_start("gdelt", messages, errors)
    _try_start("matcher", messages, errors)

    return {"messages": messages, "errors": errors}


def stop_pipeline(stop_docker=True):
    """Stop every managed Python process, then the Docker containers.

    Only stops containers (docker compose stop) -- never removes volumes,
    so Kafka topics and the Qdrant index survive. Checkpoints, Parquet
    output, and the trained model on disk are not touched at all.
    """
    messages = []

    # Producers first so no new events arrive while consumers drain,
    # then the downstream consumers.
    for name in ("producer", "gdelt", "matcher", "inference", "spark"):
        if _stop_process(name):
            messages.append(f"{PYTHON_PROCESSES[name]['label']} stopped")

    if stop_docker:
        running = _running_compose_services() & {"kafka", "qdrant"}

        if running:
            result = subprocess.run(
                ["docker", "compose", "stop", *sorted(running)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                messages.append(
                    " and ".join(sorted(running)).capitalize() + " stopped"
                )
            else:
                messages.append("Docker containers could not be stopped")

    return messages or ["Pipeline was already stopped"]


def read_log(process_name, line_count=30):
    log_path = _log_file(process_name)

    if not log_path.exists():
        return f"No {process_name} log is available yet."

    try:
        lines = log_path.read_text(
            errors="replace"
        ).splitlines()

        return "\n".join(lines[-line_count:])
    except OSError as error:
        return f"Could not read log: {error}"
