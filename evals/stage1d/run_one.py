"""Run one sealed Stage 1D diagnosis-only trajectory in the verified Linux runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import shutil
import subprocess
import threading
import time


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: object) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _inside(path: str | Path, root: str | Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
        return True
    except ValueError:
        return False


def _packet_sha(root: Path, path: Path) -> str | None:
    """Hash a packet file without following any symlink component (Linux runtime)."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("the evaluation runner requires POSIX no-follow file access")
    relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                return None
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    except OSError:
        return None
    finally:
        os.close(directory)


def _snapshot_packet(root: Path, destination: Path) -> dict:
    """Copy regular bytes only; archive link targets as text, never dereference them."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("the evaluation runner requires POSIX no-follow snapshots")
    destination.mkdir()
    result: dict = {"policy": "regular files only; symlink targets are metadata, never copied",
                    "symlinks": [], "nonregular": [], "errors": []}

    def visit(directory: int, output: Path, prefix: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = prefix / entry.name
                try:
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        result["symlinks"].append({"path": str(relative), "target": os.readlink(entry.name, dir_fd=directory)})
                    elif stat.S_ISDIR(metadata.st_mode):
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                        try:
                            folder = output / entry.name
                            folder.mkdir()
                            visit(child, folder, relative)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(metadata.st_mode):
                        descriptor = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                        with os.fdopen(descriptor, "rb") as source:
                            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                                raise OSError("packet node changed from regular file")
                            with (output / entry.name).open("xb") as target:
                                shutil.copyfileobj(source, target)
                    else:
                        result["nonregular"].append({"path": str(relative), "mode": stat.S_IFMT(metadata.st_mode)})
                except OSError as exc:
                    result["errors"].append({"path": str(relative), "error": repr(exc)})

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(descriptor, destination, Path())
    finally:
        os.close(descriptor)
    return result


def run(manifest_path: Path, label: str) -> None:
    if not hasattr(os, "O_NOFOLLOW") or os.name != "posix":
        raise RuntimeError("the evaluation runner requires the verified Linux runtime")
    manifest_path = Path(manifest_path).resolve()
    expected = manifest_path.with_suffix(".sha256").read_text().strip()
    if sha(manifest_path) != expected:
        raise ValueError("frozen manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "1D" or manifest.get("repair_endpoint") != "NOT_EVALUATED":
        raise ValueError("manifest is not the Stage 1D diagnosis-only contract")
    if label not in manifest["runs"]:
        raise ValueError("unknown packet label")
    arm = manifest["runs"][label]
    for filename, digest in manifest["frozen_files"].items():
        path = Path(filename)
        if not path.is_file() or sha(path) != digest:
            raise ValueError("frozen input/code changed: " + filename)
    out = Path(arm["evidence"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = Path(arm["work"]).resolve()
    candidate_before = sha(work / "candidate.py")
    save(out / "start.json", {
        "manifest_sha256": sha(manifest_path), "label": label,
        "requested_model": manifest["model"], "reasoning_effort": manifest["reasoning_effort"],
        "argv": arm["argv"], "started_unix": time.time(),
        "wall_limit_s": manifest["wall_limit_s"], "tool_limit": manifest["tool_limit"],
        "measurement_job_limit": manifest["measurement_job_limit"],
        "candidate_sha256_before": candidate_before,
        "overall_status": "PARTIAL", "evaluation_scope": "CONDITIONAL_DIAGNOSIS_ONLY",
        "repair_endpoint": "NOT_EVALUATED",
    })
    started = time.monotonic()
    events: list[dict] = []
    event_times: list[dict] = []
    parse_errors: list[str] = []
    reader_errors: list[str] = []
    violations: list[str] = []
    counts = {"started_tools": 0, "completed_tools": 0}
    process: subprocess.Popen[bytes] | None = None
    server = None
    stop_event = threading.Event()
    finished: float | None = None

    def stop() -> None:
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def reader(stream) -> None:
        try:
            with (out / "trace.jsonl").open("xb") as target:
                for line in iter(stream.readline, b""):
                    target.write(line); target.flush()
                    try:
                        event = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError):
                        parse_errors.append(line[:200].decode("utf-8", errors="replace")); continue
                    events.append(event)
                    item = event.get("item", {})
                    event_times.append({"offset_seconds": time.monotonic() - started,
                                        "type": event.get("type"), "item_type": item.get("type")})
                    kind = item.get("type")
                    if kind and kind not in {"command_execution", "reasoning", "agent_message"}:
                        violations.append("forbidden tool: " + str(kind)); stop()
                    if kind == "command_execution":
                        if event.get("type") == "item.started":
                            counts["started_tools"] += 1
                            if counts["started_tools"] > manifest["tool_limit"]:
                                violations.append("tool call cap exceeded"); stop()
                        elif event.get("type") == "item.completed":
                            counts["completed_tools"] += 1
        except Exception as exc:
            reader_errors.append(repr(exc)); stop()

    status = "launch_failed"
    launch_error = None
    deadline = started + manifest["wall_limit_s"]
    try:
        try:
            from .execution_transport import MeasurementServer
        except ImportError:
            from execution_transport import MeasurementServer  # type: ignore[no-redef]
        server = MeasurementServer(work, out, arm["python"], arm["source_sha256"], stop_event, deadline - 2)
        server.start()
        env = os.environ.copy()
        env["PATH"] = manifest["path_prefix"] + ":" + env.get("PATH", "")
        process = subprocess.Popen(arm["argv"], cwd=work, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=(out / "stderr.txt").open("xb"),
                                   start_new_session=True, env=env)
        thread = threading.Thread(target=reader, args=(process.stdout,), daemon=True)
        thread.start()
        process.stdin.write(Path(arm["prompt"]).read_bytes()); process.stdin.close()
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic() - 2)); status = "exited"
        except subprocess.TimeoutExpired:
            status = "wall_budget_exhausted"; violations.append(status); stop()
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        finished = time.monotonic(); stop_event.set()
        server.join(max(0, deadline - time.monotonic())); thread.join(max(0, deadline - time.monotonic()))
        if thread.is_alive(): reader_errors.append("stdout reader did not close")
    except Exception as exc:
        launch_error = repr(exc)
    finally:
        stop(); stop_event.set()
        if server: server.join(max(0, deadline - time.monotonic()))
        finals = [event for event in events if event.get("type") == "turn.completed"]
        usage = finals[-1].get("usage") if finals else None
        if usage and usage.get("output_tokens", 0) > manifest["output_token_limit"]:
            violations.append("output token budget exceeded; backend hard cap unavailable")
        candidate_after = _packet_sha(work, work / "candidate.py")
        packet_drift = [name for name, digest in manifest["frozen_files"].items()
                        if _inside(name, work) and _packet_sha(work, Path(name)) != digest]
        wall = (finished or time.monotonic()) - started
        try:
            snapshot = _snapshot_packet(work, out / "workspace-after")
        except (OSError, RuntimeError) as exc:
            snapshot = {"symlinks": [], "nonregular": [], "errors": [{"path": ".", "error": repr(exc)}]}
        save(out / "workspace-snapshot.json", snapshot)
        if snapshot["errors"]:
            violations.append("packet snapshot incomplete; see workspace-snapshot.json")
        record = {
            "status": status, "exit_code": process.returncode if process else None,
            "wall_seconds": wall, "launch_error": launch_error, **counts,
            "usage": usage, "cuda_active_seconds": None, "monetary_cost": None,
            "turn_complete_count": len(finals),
            "thread_ids": [e["thread_id"] for e in events if e.get("type") == "thread.started"],
            "reported_models": [e["model"] for e in events if e.get("model")],
            "parse_errors": parse_errors, "reader_errors": reader_errors,
            "violations": violations, "packet_drift": packet_drift,
            "measurement_jobs": server.jobs if server else 0,
            "measurement_server_error": server.last_error if server else None,
            "source_sha256_after": _packet_sha(work, work / "workload.py"),
            "candidate_sha256_before": candidate_before, "candidate_sha256_after": candidate_after,
            "overall_status": "PARTIAL", "evaluation_scope": "CONDITIONAL_DIAGNOSIS_ONLY",
            "repair_endpoint": "NOT_EVALUATED",
            "tool_events": [e for e in events if e.get("item", {}).get("type") == "command_execution"],
        }
        if record["measurement_server_error"]:
            violations.append("measurement server failed")
        if wall > manifest["wall_limit_s"]:
            violations.append("session wall cap exceeded")
        record["violations"] = violations
        save(out / "record.json", record)
        save(out / "event-arrival-times.json", event_times)
        save(out / "hashes.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p != out / "hashes.json"})
        print(json.dumps({k: v for k, v in record.items() if k != "tool_events"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path); parser.add_argument("label")
    args = parser.parse_args(); run(args.manifest, args.label)
