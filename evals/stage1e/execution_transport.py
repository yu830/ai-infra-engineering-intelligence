"""Fixed-source diagnostic worker with private execution and read-only publication.

Versioned from Stage1D. This transport never executes, imports or copies an
Agent-authored proposal into the worker's cwd. OS/native-tool permissions must
independently protect packet inputs and observations before a formal run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time
import traceback
from urllib.parse import urlsplit

from evals.stage1d.execution_transport import (
    _atomic_json, _safe_output, _terminate,
)
from evals.stage1d.run_one import _snapshot_packet

PROTOCOL = "stage1e-fixed-source-v1"
REQUEST_RE = re.compile(r"request-([0-9a-f]{32})\.json\Z")
MODE_RE = re.compile(r"[a-z][a-z0-9_-]{0,48}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_KEYS = {"protocol", "request_id", "mode", "output"}
VIEW_VERSION = "stage1e-measurements-v3"
# Evaluator predictions/classifications are not physical observations. The
# unfiltered run/events stay in private evidence and are delivered for review.
# Expose all measured bytes/strides/aliases, but do not give the Agent gold
# expectations merely because the qualification driver records them.
EVALUATOR_FIELDS = {"potential_materialized_rhs_bytes", "potential_temporary_rhs_bytes",
                    "potential_materialized_bytes", "materialization_observed",
                    "expanded_rhs_shape", "flat_rhs_shape"}
PRIVATE_METADATA = {"argv", "source_path", "cwd", "pid", "python_executable", "python_prefix",
                    "python_base_prefix", "torch_module", "source_files_sha256",
                    "nvidia_smi_compute_apps", "traceback", "source_files"}


def execution_environment() -> dict[str, str]:
    """Preserve runtime controls and credential-free proxy routing, not host secrets."""
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "TERM", "SHELL",
               "XDG_RUNTIME_DIR", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
               "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
               "PYTORCH_NO_CUDA_MEMORY_CACHING", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
               "TORCH_CUDNN_V8_API_ENABLED", "CUDA_MODULE_LOADING"}
    result = {key:value for key,value in os.environ.items() if key in allowed}
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if not value:
            continue
        try:
            parsed = urlsplit(value)
            if (parsed.scheme in {'http','https','socks5','socks5h'} and parsed.hostname
                    and parsed.username is None and parsed.password is None
                    and parsed.path in ('','/') and not parsed.query and not parsed.fragment):
                result[key] = value
        except ValueError:
            pass  # Invalid routing is omitted, never printed or handed to tools.
    return result


def measurement_view(value):
    if isinstance(value, str):
        return re.sub(r"/(?:home|mnt|tmp)/[^\s'\"\],)]+", "<private-path>", value)
    if isinstance(value, list):
        return [measurement_view(item) for item in value]
    if isinstance(value, dict):
        result = {key: measurement_view(item) for key, item in value.items()
                  if key not in EVALUATOR_FIELDS | PRIVATE_METADATA
                  and not any(prefix in key for prefix in ('/home/','/mnt/','/tmp/'))}
        if result.get("name") == "storage_materialization":
            result["name"] = "storage_probe"
        if isinstance(result.get("required_rows"), list):
            result["required_rows"] = ["storage_probe" if item == "storage_materialization" else item
                                       for item in result["required_rows"]]
        return result
    return value


def project_observations(stage: Path):
    for name in ("run.json", "events.jsonl"):
        path = stage / name
        if not path.is_file():
            continue
        data = regular_bytes(path).decode("utf-8")
        if name.endswith(".jsonl"):
            content = "".join(json.dumps(measurement_view(json.loads(line)), ensure_ascii=False) + "\n"
                              for line in data.splitlines())
            path.write_text(content, encoding="utf-8")
        else:
            _atomic_json(path, measurement_view(json.loads(data)))


def regular_bytes(path: Path, limit: int | None = None) -> bytes:
    """Open every path component without following symlinks (verified Linux)."""
    absolute = Path(os.path.abspath(path))
    directory = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("expected regular file")
            data = stream.read() if limit is None else stream.read(limit + 1)
            if limit is not None and len(data) > limit:
                raise ValueError(f"file exceeds{limit}bytes")
            return data
    finally:
        os.close(directory)


def sha(path: Path) -> str:
    return hashlib.sha256(regular_bytes(path)).hexdigest()


def hash_inventory(root: Path) -> tuple[dict, list]:
    """Keep failed-job nonregular paths explicit; never hash their targets."""
    hashes, omitted = {}, []
    for path in sorted(root.rglob("*")):
        if path == root / "hashes.json":
            continue
        name = str(path.relative_to(root))
        try:
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                omitted.append({"path": name, "kind": "nonregular", "mode": stat.S_IFMT(mode)})
                continue
            hashes[name] = sha(path)
        except (OSError, ValueError) as exc:
            omitted.append({"path": name, "kind": "unreadable", "error": repr(exc)})
    return hashes, omitted


def read_request(directory: Path, filename: str) -> dict:
    """Bounded regular-file read; reject directory/leaf symlink traversal."""
    root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("request is not a regular file")
            data = stream.read(8193)
            if len(data) > 8192:
                raise ValueError("request exceeds8192bytes")
            value = json.loads(data)
    finally:
        os.close(root_fd)
    if not isinstance(value, dict) or set(value) != REQUEST_KEYS:
        raise ValueError("request fields differ from protocol")
    return value


class MeasurementServer:
    def __init__(self, work: Path, evidence: Path, worker: dict,
                 stop_event: threading.Event, deadline: float):
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("verified POSIX runtime required")
        self.work = Path(work).resolve()
        self.evidence = Path(evidence).resolve()
        if self.evidence.is_relative_to(self.work):
            raise ValueError("private worker evidence must be outside the Agent packet")
        self.requests = self.work / "requests"
        self.observations = self.work / "observations"
        self.staging = self.observations / ".staging"
        for directory in (self.requests, self.observations, self.staging):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("packet directory must already exist without a symlink")
        self.source = Path(worker["source"]).absolute()
        if self.source.is_symlink() or not self.source.is_file():
            raise ValueError("approved source must be a regular source file")
        if self.source.resolve().is_relative_to(self.work):
            raise ValueError("approved source must be evaluator-owned outside the packet")
        self.python = os.path.abspath(worker["python"])  # preserve venv executable spelling
        self.output_argument = worker.get("output_argument", "absolute")
        if self.output_argument not in {"absolute", "basename"}:
            raise ValueError("unknown fixed worker output argument contract")
        self.expected_sha = worker["sha256"]
        if not isinstance(self.expected_sha, str) or not SHA_RE.fullmatch(self.expected_sha):
            raise ValueError("invalid approved source hash")
        self.modes = worker["modes"]
        if not isinstance(self.modes, dict) or not self.modes:
            raise ValueError("no fixed worker modes")
        for name, args in self.modes.items():
            if not isinstance(name, str) or not MODE_RE.fullmatch(name):
                raise ValueError("invalid fixed mode")
            if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                raise ValueError("worker arguments must be frozen strings")
            if "--output" in args:
                raise ValueError("output is evaluator-owned")
        self.max_jobs = worker["max_jobs"]
        self.timeout = worker["timeout_seconds"]
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 32:
            raise ValueError("invalid job budget")
        if type(self.timeout) not in (int, float) or not 0 < self.timeout <= 300:
            raise ValueError("invalid job timeout")
        self.stop_event, self.deadline = stop_event, deadline
        self.jobs = 0
        self.launched_jobs = 0
        self.rejections = 0
        self.request_errors = []
        self.last_error = None
        self._processed = set()
        self._active = None
        self._thread = None

    def start(self):
        self.evidence.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, name="stage1e-fixed-worker", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def _serve(self):
        try:
            while not self.stop_event.is_set() and time.monotonic() < self.deadline:
                for request_path in sorted(self.requests.iterdir()):
                    match = REQUEST_RE.fullmatch(request_path.name)
                    if not match or match[1] in self._processed:
                        continue
                    self._processed.add(match[1])
                    try:
                        self._handle(request_path.name, match[1])
                    except Exception as exc:
                        # An isolated request failure gets a terminal receipt and
                        # does not silently terminate subsequent diagnostic work.
                        error = {"request_id": match[1], "error": repr(exc),
                                 "traceback": traceback.format_exc()}
                        self.request_errors.append(error)
                        _atomic_json(self.evidence / f"request-error-{match[1]}.json", error)
                        self._receipt(match[1], {
                            "protocol": PROTOCOL, "request_id": match[1],
                            "status": "failed", "error": "host_request_failure: " + repr(exc),
                            "evaluation_scope": "DIAGNOSIS_ONLY", "patch_validation": "NOT_EVALUATED"})
                self.stop_event.wait(0.05)
        except Exception:
            self.last_error = traceback.format_exc()
            _atomic_json(self.evidence / "server-error.json", {"error": self.last_error})
        finally:
            if self._active:
                _terminate(self._active)

    def _receipt(self, request_id, value):
        _atomic_json(self.observations / f"receipt-{request_id}.json", value)

    def _reject(self, request_id, error):
        self.rejections += 1
        value = {"protocol": PROTOCOL, "request_id": request_id, "status": "rejected",
                 "error": error, "worker_launched": False, "patch_validation": "NOT_EVALUATED"}
        _atomic_json(self.evidence / f"rejection-{request_id}.json", value)
        self._receipt(request_id, value)

    def _handle(self, filename, request_id):
        try:
            request = read_request(self.requests, filename)
            if request["protocol"] != PROTOCOL or request["request_id"] != request_id:
                raise ValueError("request protocol/id mismatch")
            if request["mode"] not in self.modes:
                raise ValueError("unknown fixed mode")
            name = _safe_output(request["output"])
            public = self.observations / name
            if public.exists() or public.is_symlink():
                raise ValueError("publication name already exists")
            if self.jobs >= self.max_jobs:
                raise ValueError("measurement job budget exhausted")
        except Exception as exc:
            self._reject(request_id, repr(exc))
            return
        self.jobs += 1
        root = self.evidence / "evaluator-tools" / request_id
        root.mkdir(parents=True, exist_ok=False)
        _atomic_json(root / "request.json", request)
        cwd = root / "private-work"
        cwd.mkdir()
        output = (cwd if self.output_argument == "basename" else root) / "work-output"
        source_copy = root / "source-approved.py"
        process = None
        started = time.monotonic()
        record = {"request_id": request_id, "job_number": self.jobs, "request": request,
                  "source_sha256_expected": self.expected_sha, "started_unix": time.time(),
                  "python": self.python, "job_timeout_seconds": self.timeout,
                  "evaluation_scope": "DIAGNOSIS_ONLY", "patch_validation": "NOT_EVALUATED",
                  "worker_launched": False, "status": "failed", "returncode": None,
                  "error": None, "gpu_active_seconds": None, "monetary_cost": None}
        record["output_argument_contract"] = self.output_argument
        result = None
        try:
            source_bytes = regular_bytes(self.source)
            if hashlib.sha256(source_bytes).hexdigest() != self.expected_sha:
                raise ValueError("approved source changed before execution")
            source_copy.write_bytes(source_bytes)
            argv = [self.python, "-I", str(source_copy), *self.modes[request["mode"]],
                    "--output", output.name if self.output_argument == "basename" else str(output)]
            _atomic_json(root / "argv.json", {"argv": argv, "cwd": str(cwd)})
            with (root / "stdout.txt").open("xb") as out, (root / "stderr.txt").open("xb") as err:
                process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, start_new_session=True,
                                           env=execution_environment())
                self._active = process
                self.launched_jobs += 1
                record["worker_launched"] = True
                while process.poll() is None:
                    if self.stop_event.is_set():
                        raise TimeoutError("trial stopped")
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError("trial deadline exhausted")
                    if time.monotonic() - started >= self.timeout:
                        raise TimeoutError("measurement timeout exhausted")
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
            record["returncode"] = process.returncode
            if sha(source_copy) != self.expected_sha or sha(self.source) != self.expected_sha:
                raise ValueError("approved source changed during execution")
            if not (output / "run.json").is_file():
                raise ValueError("worker run.json missing")
            result = json.loads(regular_bytes(output / "run.json"))
            if not isinstance(result, dict):
                raise ValueError("worker run.json must be an object")
            record["driver_status"] = result.get("status")
            if result.get("status") == "not_run_resource":
                finalization = result.get("finalization")
                if not isinstance(finalization, dict) or finalization.get("status") != "not_run_resource":
                    raise ValueError("resource stop lacks matching finalization")
                record.update(status="not_run_resource", error="worker resource gate; no completed measurement")
            elif process.returncode != 0:
                raise RuntimeError(f"worker exit code {process.returncode}")
            elif result.get("status") != "completed":
                raise ValueError("worker did not report completed status")
            else:
                finalization = result.get("finalization")
                if not isinstance(finalization, dict) or finalization.get("status") != "completed" or finalization.get("issues") != []:
                    raise ValueError("worker finalization missing or failed")
                record["status"] = "completed"
        except Exception as exc:
            record["error"] = repr(exc)
            if process and process.poll() is None:
                _terminate(process)
                process.wait(timeout=2)
            if process:
                record["returncode"] = process.returncode
        finally:
            self._active = None
        try:
            source_after = sha(source_copy)
        except (OSError, ValueError):
            source_after = None
        record.update(finished_unix=time.time(), wall_seconds=time.monotonic() - started,
                      source_sha256_after=source_after)
        _atomic_json(root / "process.json", record)
        # .staging is an existing permission-denied child of read-only
        # observations. Only a complete regular-file snapshot is renamed into
        # the visible result namespace; failed staging is retained, not exposed
        # as a completed observation. Both paths are on the same filesystem.
        publication = False
        if output.is_dir():
            try:
                stage = self.staging / request_id
                snapshot = _snapshot_packet(output, stage)
                _atomic_json(root / "publication-snapshot.json", snapshot)
                if snapshot["symlinks"] or snapshot["nonregular"] or snapshot["errors"]:
                    raise ValueError("worker publication contains links, special files or snapshot errors")
                project_observations(stage)
                stage.rename(public)
                publication = True
            except Exception as exc:
                record.update(status="failed", error=f"publication failed: {exc!r}")
                _atomic_json(root / "process.json", record)
        _, omitted = hash_inventory(root)
        seal_status = "complete" if not omitted else "incomplete"
        if omitted:
            record.update(status="failed", evidence_seal="incomplete",
                          error="job seal incomplete; see unhashed-paths.json")
            _atomic_json(root / "unhashed-paths.json", {"status": "incomplete", "paths": omitted})
        else:
            record["evidence_seal"] = "complete"
        _atomic_json(root / "process.json", record)
        receipt = {"protocol": PROTOCOL, "request_id": request_id,
                   "mode": request["mode"], "status": record["status"], "error": record["error"],
                   "driver_status": record.get("driver_status"),
                   "worker_launched": record["worker_launched"], "returncode": record["returncode"],
                   "source_sha256": self.expected_sha,
                   "raw_result": f"observations/{name}/run.json" if publication and not omitted else None,
                   "raw_directory": f"observations/{name}" if publication and not omitted else None,
                   "evidence_seal": seal_status,
                   "observation_view_version": VIEW_VERSION,
                   "raw_result_scope": "measurement view; unfiltered evaluator run and events retained privately",
                   "evaluation_scope": "DIAGNOSIS_ONLY", "patch_validation": "NOT_EVALUATED"}
        _atomic_json(root / "receipt.json", receipt)
        # receipt.json is part of the final nested seal.
        hashes, final_omitted = hash_inventory(root)
        if final_omitted != omitted:
            raise ValueError("job evidence changed during final sealing")
        _atomic_json(root / "hashes.json", hashes)
        self._receipt(request_id, receipt)
