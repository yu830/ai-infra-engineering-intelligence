"""Trusted fixed-source Stage 1D measurement transport.

This is a small versioned adaptation of the Stage 1C transport.  It executes
only the evaluator-owned ``cases/e010/workload.py`` and never imports or runs
``candidate.py`` or ``proposal.py``.  The latter is a static Agent artifact;
patch validation is intentionally NOT_EVALUATED in this scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
import uuid
from typing import Any


VALID_MODES = frozenset(
    {"deployment", "policy-control", "context-control", "training-boundary", "dispatch-audit"}
)
REQUEST_KEYS = frozenset({"protocol", "request_id", "mode", "output"})
REQUEST_PROTOCOL = "stage1d-fixed-source-transport-v1"
REQUEST_PREFIX = "measurement-request-"
RECEIPT_PREFIX = "measurement-receipt-"
REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
MAX_JOBS = 8
JOB_TIMEOUT_SECONDS = 60.0
POLL_SECONDS = 0.05
PAIR_COUNT = 12
WARMUP_COUNT = 3
PAIR_SEED = 20261019


class TransportError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_output(value: Any) -> str:
    if not isinstance(value, str) or OUTPUT_RE.fullmatch(value) is None:
        raise TransportError("output must be a safe basename")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise TransportError("output must not contain a path separator")
    return value


def validate_request(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != REQUEST_KEYS:
        raise TransportError("request fields differ from the fixed protocol")
    if value.get("protocol") != REQUEST_PROTOCOL:
        raise TransportError("unknown request protocol")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise TransportError("request_id must be a lowercase UUID hex string")
    mode = value.get("mode")
    if mode not in VALID_MODES:
        raise TransportError("invalid mode")
    return {
        "protocol": REQUEST_PROTOCOL,
        "request_id": request_id,
        "mode": mode,
        "output": _safe_output(value.get("output")),
    }


def _safe_child(root: Path, name: str) -> Path:
    root = root.resolve()
    raw = root / name
    if raw.is_symlink():
        raise TransportError("output target must not be a symlink")
    candidate = raw.resolve()
    try:
        if os.path.commonpath((str(root), str(candidate))) != str(root):
            raise TransportError("path escapes packet root")
    except ValueError as exc:
        raise TransportError("path roots are incompatible") from exc
    return candidate


def _copy_tree_without_symlinks(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise TransportError("output is a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_symlink():
            raise TransportError("output contains a symlink")
        if item.is_dir():
            _copy_tree_without_symlinks(item, target)
        else:
            shutil.copy2(item, target)


def _hash_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    process.kill()


@dataclass(frozen=True)
class _Accepted:
    request: dict[str, str]
    receipt: Path


class MeasurementServer:
    """Execute only the fixed trusted E010 workload, at most eight jobs."""

    def __init__(self, work: Path, evidence: Path, python: str, expected_sha256: str,
                 stop_event: threading.Event, deadline: float) -> None:
        self.work = Path(work).resolve()
        self.evidence = Path(evidence).resolve()
        self.python = os.path.abspath(os.fspath(python))
        if SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("expected workload SHA256 is invalid")
        self.expected_sha256 = expected_sha256
        self.stop_event = stop_event
        self.deadline = float(deadline)
        self._thread: threading.Thread | None = None
        self._processed: set[str] = set()
        self._jobs = 0
        self._active: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def jobs(self) -> int:
        with self._lock:
            return self._jobs

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def start(self) -> "MeasurementServer":
        self.work.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, name="stage1d-measurement-server", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _serve(self) -> None:
        try:
            while not self.stop_event.is_set() and time.monotonic() < self.deadline:
                for path in sorted(self.work.glob(f"{REQUEST_PREFIX}*.json")):
                    request_id = path.name[len(REQUEST_PREFIX):-5]
                    if not REQUEST_ID_RE.fullmatch(request_id) or request_id in self._processed:
                        continue
                    self._processed.add(request_id)
                    self._handle(path, request_id)
                time.sleep(POLL_SECONDS)
        except Exception as exc:
            with self._lock:
                self._last_error = repr(exc)
        finally:
            if self._active is not None:
                _terminate(self._active)

    def _handle(self, request_path: Path, request_id: str) -> None:
        receipt_path = self.work / f"{RECEIPT_PREFIX}{request_id}.json"
        try:
            request = validate_request(json.loads(request_path.read_text(encoding="utf-8")))
            if request["request_id"] != request_id:
                raise TransportError("request id differs from filename")
        except Exception as exc:
            _atomic_json(receipt_path, {"receipt_version": 1, "request_id": request_id,
                                        "status": "rejected", "request": None,
                                        "error": repr(exc), "evidence": None})
            return
        with self._lock:
            if self._jobs >= MAX_JOBS:
                _atomic_json(receipt_path, {"receipt_version": 1, "request_id": request_id,
                                            "status": "rejected", "request": request,
                                            "error": "measurement job limit reached", "evidence": None})
                return
            self._jobs += 1
            job_number = self._jobs
        self._execute(_Accepted(request, receipt_path), job_number)

    def _execute(self, accepted: _Accepted, job_number: int) -> None:
        request = accepted.request
        request_id = request["request_id"]
        evidence_root = self.evidence / "evaluator-tools" / request_id
        evidence_root.mkdir(parents=True, exist_ok=False)
        source = self.work / "workload.py"
        source_before = sha256(source) if source.is_file() else None
        record: dict[str, Any] = {
            "request_id": request_id, "job_number": job_number, "request": request,
            "expected_workload_sha256": self.expected_sha256,
            "workload_sha256_before": source_before, "python": self.python,
            "work": str(self.work), "started_unix": time.time(),
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            "pairs": PAIR_COUNT, "warmups": WARMUP_COUNT, "seed": PAIR_SEED,
            "patch_validation": "NOT_EVALUATED",
            "overall_status": "PARTIAL",
            "evaluation_scope": "CONDITIONAL_DIAGNOSIS_ONLY",
            "repair_endpoint": "NOT_EVALUATED",
        }
        _atomic_json(evidence_root / "request.json", request)
        argv = [self.python, "-I", str(source), "--mode", request["mode"],
                "--pairs", str(PAIR_COUNT), "--warmups", str(WARMUP_COUNT),
                "--seed", str(PAIR_SEED), "--output", request["output"]]
        _atomic_json(evidence_root / "argv.json", {"argv": argv, "cwd": str(self.work)})
        if source_before != self.expected_sha256:
            record.update(status="rejected", error="trusted workload source SHA256 mismatch", finished_unix=time.time())
            _atomic_json(evidence_root / "process.json", record)
            self._finish(evidence_root)
            _atomic_json(accepted.receipt, {"receipt_version": 1, "request_id": request_id,
                                            "status": "rejected", "request": request,
                                            "error": record["error"], "evidence": f"evaluator-tools/{request_id}"})
            return
        output = _safe_child(self.work, request["output"])
        if output.exists() or output.is_symlink():
            record.update(status="failed", error="requested output directory already exists", finished_unix=time.time())
            _atomic_json(evidence_root / "process.json", record)
            self._finish(evidence_root)
            _atomic_json(accepted.receipt, {"receipt_version": 1, "request_id": request_id,
                                            "status": "failed", "request": request,
                                            "error": record["error"], "evidence": f"evaluator-tools/{request_id}"})
            return
        started = time.monotonic()
        stdout_path, stderr_path = evidence_root / "stdout.txt", evidence_root / "stderr.txt"
        process: subprocess.Popen[bytes] | None = None
        timeout_reason: str | None = None
        stdout = b""
        stderr = b""
        out_file = err_file = None
        try:
            out_file, err_file = stdout_path.open("xb"), stderr_path.open("xb")
            process = subprocess.Popen(argv, cwd=self.work, stdin=subprocess.DEVNULL,
                                       stdout=out_file, stderr=err_file,
                                       start_new_session=(os.name == "posix"),
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                                       env=os.environ.copy())
            with self._lock:
                self._active = process
            while process.poll() is None:
                if self.stop_event.is_set():
                    timeout_reason = "server_stop_requested"; _terminate(process); break
                if time.monotonic() >= self.deadline:
                    timeout_reason = "trial_deadline_exhausted"; _terminate(process); break
                if time.monotonic() - started >= JOB_TIMEOUT_SECONDS:
                    timeout_reason = "job_timeout_exhausted"; _terminate(process); break
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                _terminate(process)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _terminate(process)
        except Exception as exc:
            timeout_reason = timeout_reason or "host_launch_or_capture_error"
            if err_file:
                err_file.write(("\n" + repr(exc) + "\n" + traceback.format_exc()).encode())
                err_file.flush()
            else:
                stderr_path.write_text(repr(exc) + "\n" + traceback.format_exc(), encoding="utf-8")
            if process:
                _terminate(process)
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass
        finally:
            if out_file: out_file.close()
            if err_file: err_file.close()
            with self._lock:
                self._active = None
        stdout = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr = stderr_path.read_bytes() if stderr_path.is_file() else b""
        returncode = process.returncode if process else None
        source_after = sha256(source) if source.is_file() else None
        output_present = output.is_dir()
        copy_error: str | None = None
        run_data = None
        status = "completed"
        error = None
        if timeout_reason:
            status, error = "failed", timeout_reason
        elif returncode != 0:
            status, error = "failed", f"workload exit code {returncode}"
        elif source_after != self.expected_sha256:
            status, error = "failed", "trusted workload source changed during execution"
        elif not output_present:
            status, error = "failed", "trusted workload did not create output directory"
        if output_present:
            try:
                _copy_tree_without_symlinks(output, evidence_root / "work-output")
            except Exception as exc:
                copy_error = repr(exc)
        if status == "completed":
            if copy_error:
                status, error = "failed", f"run evidence copy failed: {copy_error}"
            elif not (output / "run.json").is_file():
                status, error = "failed", "run evidence run.json is missing"
            else:
                try:
                    run_data = json.loads((output / "run.json").read_text(encoding="utf-8"))
                    run_status = run_data.get("status") if isinstance(run_data, dict) else None
                    if run_status != "completed":
                        status, error = "failed", "trusted workload reported status " + str(run_status)
                except Exception as exc:
                    status, error = "failed", f"run evidence read failed: {exc!r}"
        record.update(status=status, error=error, returncode=returncode,
                      finished_unix=time.time(), wall_seconds=time.monotonic() - started,
                      workload_sha256_after=source_after, timeout_reason=timeout_reason,
                      work_output_present=output_present,
                      work_output_copied=output_present and copy_error is None,
                      work_output_copy_error=copy_error)
        _atomic_json(evidence_root / "process.json", record)
        self._finish(evidence_root)
        receipt: dict[str, Any] = {"receipt_version": 1, "request_id": request_id,
                                   "status": status, "request": request,
                                   "result": run_data, "error": error,
                                   "returncode": returncode,
                                   "evidence": f"evaluator-tools/{request_id}"}
        if stdout:
            receipt["stdout_tail"] = stdout[-4096:].decode("utf-8", errors="replace")
        if stderr:
            receipt["stderr_tail"] = stderr[-4096:].decode("utf-8", errors="replace")
        _atomic_json(accepted.receipt, receipt)

    @staticmethod
    def _finish(root: Path) -> None:
        _atomic_json(root / "hashes.json", _hash_files(root))
