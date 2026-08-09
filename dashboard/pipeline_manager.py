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


def is_kafka_running():
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

        running_services = result.stdout.splitlines()
        return "kafka" in running_services

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def _wait_for_kafka(timeout=60):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection(
                (KAFKA_HOST, KAFKA_PORT),
                timeout=2,
            ):
                return True
        except OSError:
            time.sleep(1)

    return False


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
    return {
        "Kafka": is_kafka_running(),
        "Spark": _check_for_unexpected_exit("spark"),
        "Producer": _check_for_unexpected_exit("producer"),
    }


def start_pipeline():
    messages = []

    if not is_kafka_running():
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "kafka"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or "Kafka could not be started."
            )

        messages.append("Kafka started")
    else:
        messages.append("Kafka was already running")

    if not _wait_for_kafka():
        raise RuntimeError(
            "Kafka started, but port 9092 did not become ready."
        )

    # Launched via plain `python`, not spark-submit: wiki_stream.py sets
    # spark.driver.memory / spark.master itself via SparkSession.builder,
    # and those only take effect if the driver JVM doesn't already exist
    # when that code runs. spark-submit starts the JVM (with Spark's
    # default 1g heap) before the script executes, silently overriding
    # the script's own memory tuning -- which matters on this
    # memory-constrained devcontainer.
    spark_started = _start_process(
        "spark",
        [
            sys.executable,
            PROJECT_ROOT / "spark" / "wiki_stream.py",
        ],
    )

    messages.append(
        "Spark started"
        if spark_started
        else "Spark was already running"
    )

    # Give Spark time to initialize its streaming query before
    # the producer begins publishing new events.
    if spark_started:
        time.sleep(8)

    producer_started = _start_process(
        "producer",
        [
            sys.executable,
            PROJECT_ROOT
            / "producer"
            / "wikipedia_producer.py",
        ],
    )

    messages.append(
        "Producer started"
        if producer_started
        else "Producer was already running"
    )

    return messages


def stop_pipeline(stop_kafka=True):
    messages = []

    if _stop_process("producer"):
        messages.append("Producer stopped")

    if _stop_process("spark"):
        messages.append("Spark stopped")

    if stop_kafka and is_kafka_running():
        result = subprocess.run(
            ["docker", "compose", "stop", "kafka"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode == 0:
            messages.append("Kafka stopped")
        else:
            messages.append("Kafka could not be stopped")

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
