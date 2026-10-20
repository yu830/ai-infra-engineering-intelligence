"""Strict Stage 2A case/worker/transport preflight contracts.

Stage 1F stopped because a worker could finish its mathematical observation
while the evaluator transport rejected the result for a missing finalization
field.  Stage 2A puts that boundary in front of a future freeze call.  The
preflight deliberately reuses the existing Stage 1E ``MeasurementServer`` and
packet-local client.  It records the actual receipt, process record and
published ``run.json`` for every fixed mode; ``require_preflight`` rereads all
of them before a case can be frozen.

This module does not run an Agent and does not accept an Agent-authored source
proposal.  A worker is an evaluator-owned regular source file whose bytes are
hashed before and after every preflight.  CPU-only contract tests can create a
synthetic proof with ``write_synthetic_proof``; they never stand in for the
real E012 MeasurementServer smoke.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence


CONTRACT_VERSION = "stage2a-case-contract-v1"
PROOF_VERSION = "stage2a-preflight-proof-v1"
TRANSPORT_PROTOCOL = "stage1e-fixed-source-v1"
DEFAULT_VALID_FOR_SECONDS = 24 * 60 * 60
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
MODE_RE = re.compile(r"[a-z][a-z0-9_-]{0,48}\Z")
SAFE_OUTPUT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class ContractError(ValueError):
    """Raised when a proof or fixed worker contract cannot be accepted."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} is missing or not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _json_read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(value))
    os.replace(temporary, path)


def _is_regular(path: Path, *, reject_symlink: bool = True) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    if reject_symlink and stat.S_ISLNK(mode):
        return False
    return stat.S_ISREG(mode)


def _sha256(path: Path) -> str:
    if not _is_regular(path):
        raise ContractError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_module_path(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise ContractError(f"cannot resolve tool module {module_name}")
    return Path(spec.origin).resolve()


def _absolute_file(path_value: Any, field: str, *, reject_symlink: bool = True) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ContractError(f"{field} must be a non-empty path")
    path = Path(path_value)
    if not path.is_absolute():
        raise ContractError(f"{field} must be absolute")
    if reject_symlink and path.is_symlink():
        raise ContractError(f"{field} must not be a symlink: {path}")
    path = path.resolve(strict=False)
    if not _is_regular(path):
        raise ContractError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _current_launcher_path() -> Path:
    """Return the launcher spelling for this Python environment."""
    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.prefix) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if candidate.is_file():
            return Path(os.path.abspath(candidate))
    return Path(os.path.abspath(sys.executable))


def _sha_field(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA256")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ContractError(f"evidence path is outside proof root: {path}") from exc
    return relative.as_posix()


def _safe_relative(root: Path, value: Any, field: str, *, regular: bool = True) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{field} must be a relative POSIX path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ContractError(f"{field} escapes its proof root")
    path = root.joinpath(*candidate.parts)
    # Resolve every path component.  A symlink in a path is illegal evidence,
    # even when the final target happens to be inside the proof root.
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise ContractError(f"{field} contains a symlink: {value}")
        except OSError as exc:
            raise ContractError(f"cannot inspect {field}: {exc}") from exc
    if regular and not _is_regular(path):
        raise ContractError(f"{field} must reference a regular file: {value}")
    return path


def _path_sha(root: Path, value: Any, field: str) -> tuple[Path, str]:
    path = _safe_relative(root, value, field)
    return path, _sha256(path)


def _json_object(path: Path, field: str) -> dict[str, Any]:
    value = _json_read(path)
    if not isinstance(value, dict):
        raise ContractError(f"{field} must contain a JSON object")
    return value


def _normalise_modes(modes: Any, field: str) -> dict[str, list[str]]:
    if not isinstance(modes, dict) or not modes:
        raise ContractError(f"{field} must be a non-empty mode-to-argv object")
    result: dict[str, list[str]] = {}
    for mode, args in modes.items():
        if not isinstance(mode, str) or not MODE_RE.fullmatch(mode):
            raise ContractError(f"{field} has an invalid mode: {mode!r}")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ContractError(f"{field}[{mode!r}] must be a list of strings")
        if "--output" in args:
            raise ContractError("worker argv cannot own the evaluator output argument")
        result[mode] = list(args)
    return result


def _normalise_worker(worker: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    if not isinstance(worker, Mapping):
        raise ContractError("worker spec must be an object")
    forbidden = {"proposal", "proposal_path", "agent_source", "source_proposal"}
    if forbidden.intersection(worker):
        raise ContractError("Stage 2A accepts only an approved fixed worker source")
    source = _absolute_file(worker.get("source"), "worker.source")
    declared_hash = _sha_field(worker.get("sha256"), "worker.sha256")
    actual_hash = _sha256(source)
    if actual_hash != declared_hash:
        raise ContractError("worker source hash does not match the current source bytes")
    python_value = worker.get("python")
    if not isinstance(python_value, str) or not python_value:
        raise ContractError("worker.python is missing")
    python_path = Path(python_value)
    if not python_path.is_absolute():
        raise ContractError("worker.python must be absolute")
    # Preserve the caller's virtualenv spelling for subprocess execution.  A
    # venv interpreter is often a symlink whose real target does not retain the
    # venv's site-packages when invoked with ``-I``; hash the resolved target
    # only in the runtime binding.
    python_path = Path(os.path.abspath(python_path))
    python_target = python_path.resolve(strict=False)
    if not _is_regular(python_target):
        raise ContractError(f"worker.python is not a regular executable file: {python_path}")
    modes = _normalise_modes(worker.get("modes"), "worker.modes")
    max_jobs = worker.get("max_jobs")
    if type(max_jobs) is not int or not 1 <= max_jobs <= 32:
        raise ContractError("worker.max_jobs must be an integer in [1, 32]")
    if max_jobs < len(modes):
        raise ContractError("worker.max_jobs must cover every fixed case mode")
    timeout = worker.get("timeout_seconds")
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ContractError("worker.timeout_seconds must be in (0, 300]")
    output_argument = worker.get("output_argument", "absolute")
    if output_argument not in {"absolute", "basename"}:
        raise ContractError("worker.output_argument is not a fixed transport option")
    result = {
        "python": str(python_path),
        "source": str(source),
        "sha256": declared_hash,
        "modes": modes,
        "max_jobs": max_jobs,
        "timeout_seconds": timeout,
        "output_argument": output_argument,
        "source_kind": "approved_fixed",
    }
    # Retain only scalar, explicitly named metadata.  Arbitrary caller text is
    # not part of the acceptance identity and cannot turn a proposal into an
    # approved worker.
    if isinstance(worker.get("label"), str):
        result["label"] = worker["label"]
    return result, source


def _normalise_case(case: Mapping[str, Any] | str, worker_modes: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    if isinstance(case, str):
        value: dict[str, Any] = {"case_id": case}
    elif isinstance(case, Mapping):
        value = dict(case)
    else:
        raise ContractError("case spec must be an object or case id string")
    case_id = value.get("case_id", value.get("case"))
    if not isinstance(case_id, str) or not case_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", case_id):
        raise ContractError("case.case_id must be a safe non-empty identifier")
    modes = value.get("modes", list(worker_modes))
    if not isinstance(modes, list) or not modes or not all(isinstance(item, str) for item in modes):
        raise ContractError("case.modes must be a non-empty list of strings")
    if len(set(modes)) != len(modes):
        raise ContractError("case.modes contains duplicates")
    if set(modes) != set(worker_modes):
        raise ContractError("case.modes must cover exactly every fixed worker mode")
    result: dict[str, Any] = {
        "case_id": case_id,
        "modes": list(modes),
    }
    if "mode_row_counts" in value:
        counts = value["mode_row_counts"]
        if not isinstance(counts, Mapping) or set(counts) != set(modes):
            raise ContractError("case.mode_row_counts must cover exactly every case mode")
        if not all(type(counts[mode]) is int and counts[mode] > 0 for mode in modes):
            raise ContractError("case.mode_row_counts values must be positive integers")
        result["mode_row_counts"] = {mode: counts[mode] for mode in modes}
    for key in ("schema", "version", "description"):
        if key in value:
            if not isinstance(value[key], str) or not value[key]:
                raise ContractError(f"case.{key} must be a non-empty string")
            result[key] = value[key]
    return result


def _normalise_tool(spec: Mapping[str, Any] | None, name: str, default_module: str) -> dict[str, Any]:
    value = dict(spec or {})
    module = value.get("module", default_module)
    if not isinstance(module, str) or not module:
        raise ContractError(f"{name}.module is missing")
    path_value = value.get("path")
    path = _absolute_file(path_value, f"{name}.path") if path_value is not None else _resolve_module_path(module)
    actual_hash = _sha256(path)
    if value.get("sha256") is not None and _sha_field(value.get("sha256"), f"{name}.sha256") != actual_hash:
        raise ContractError(f"{name} hash does not match current tool bytes")
    result: dict[str, Any] = {
        "name": name,
        "module": module,
        "path": str(path),
        "sha256": actual_hash,
    }
    if value.get("protocol") is not None:
        if not isinstance(value["protocol"], str) or not value["protocol"]:
            raise ContractError(f"{name}.protocol must be a non-empty string")
        result["protocol"] = value["protocol"]
    result["identity_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _runtime_record(runtime: Mapping[str, Any] | None, worker_python: str) -> dict[str, Any]:
    value = dict(runtime or {})
    python_value = value.get("python", worker_python)
    worker_launcher = Path(os.path.abspath(worker_python))
    current_launcher = _current_launcher_path()
    if worker_launcher != current_launcher:
        raise ContractError("worker.python must be the current process virtualenv launcher")
    if Path(os.path.abspath(str(python_value))) != worker_launcher:
        raise ContractError("runtime.python and worker.python must use the same launcher spelling")
    # A virtualenv's interpreter is commonly a symlink to the managed Python
    # binary.  Record the resolved executable bytes while keeping source and
    # tool files strict non-symlink evidence.
    python_path = _absolute_file(python_value, "runtime.python", reject_symlink=False)
    if str(Path(worker_python).resolve(strict=False)) != str(python_path.resolve(strict=False)):
        raise ContractError("runtime.python and worker.python must identify the same executable")
    python_hash = _sha256(python_path)
    if value.get("python_sha256") is not None and _sha_field(value["python_sha256"], "runtime.python_sha256") != python_hash:
        raise ContractError("runtime Python hash does not match current executable")
    observed = {
        "python": str(python_path),
        "python_sha256": python_hash,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    for key in ("torch_version", "torch_cuda_version", "cuda_available", "device"):
        if key in value:
            observed[key] = value[key]
    result = {**observed, "identity_sha256": hashlib.sha256(_canonical(observed)).hexdigest()}
    result["launcher"] = str(worker_launcher)
    result["identity_sha256"] = hashlib.sha256(_canonical({k: v for k, v in result.items() if k != "identity_sha256"})).hexdigest()
    return result


def _default_tool_specs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"module": "evals.stage1e.execution_transport", "protocol": TRANSPORT_PROTOCOL},
        {"module": "evals.stage1e.measure", "protocol": TRANSPORT_PROTOCOL},
    )


def _collect_runtime_observations(runtime: dict[str, Any]) -> dict[str, Any]:
    """Add runtime package values only when they are observable in this process."""
    result = dict(runtime)
    try:
        import torch  # type: ignore

        result.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            result["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on runtime installation
        result["torch_probe_error"] = repr(exc)
    result["identity_sha256"] = hashlib.sha256(_canonical({k: v for k, v in result.items() if k != "identity_sha256"})).hexdigest()
    return result


def _wait_for_json(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_regular(path):
            value = _json_read(path)
            if not isinstance(value, dict):
                raise ContractError(f"receipt is not a JSON object: {path}")
            return value
        time.sleep(0.05)
    raise ContractError(f"receipt timeout: {path}")


def _failure_walk(value: Any, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if key in {"issues", "failed_observation_checks", "errors"} and isinstance(item, list) and item:
                failures.append(child)
            failures.extend(_failure_walk(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_failure_walk(item, f"{path}[{index}]"))
    return failures


def _explicit_observation_failures(value: Any, path: str = "") -> list[str]:
    """Find explicit failed/error observation states without interpreting NaN."""
    failures: list[str] = []
    if isinstance(value, Mapping):
        status = value.get("status")
        if isinstance(status, str) and status.lower() in {"failed", "failure", "error"}:
            failures.append(f"{path}.status" if path else "status")
        for key in ("error", "failure", "failure_reason", "exception"):
            if key in value and value[key] not in (None, "", [], {}):
                failures.append(f"{path}.{key}" if path else key)
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            failures.extend(_explicit_observation_failures(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_explicit_observation_failures(item, f"{path}[{index}]"))
    return failures


def _strict_receipt(receipt: Mapping[str, Any], *, mode: str, output: str, worker: Mapping[str, Any], protocol: str) -> list[str]:
    errors: list[str] = []
    required = {
        "protocol",
        "request_id",
        "mode",
        "status",
        "error",
        "driver_status",
        "worker_launched",
        "returncode",
        "source_sha256",
        "raw_result",
        "raw_directory",
        "evidence_seal",
    }
    missing = sorted(required.difference(receipt))
    if missing:
        errors.append("receipt missing fields: " + ",".join(missing))
        return errors
    if receipt.get("protocol") != protocol:
        errors.append("receipt protocol mismatch")
    if receipt.get("mode") != mode:
        errors.append("receipt mode mismatch")
    if receipt.get("status") != "completed":
        errors.append("receipt status is not completed")
    if receipt.get("driver_status") != "completed":
        errors.append("receipt driver_status is not completed")
    if receipt.get("worker_launched") is not True:
        errors.append("receipt worker_launched is not true")
    if type(receipt.get("returncode")) is not int or receipt.get("returncode") != 0:
        errors.append("receipt returncode is not zero")
    if receipt.get("error") is not None:
        errors.append("receipt contains an error")
    if receipt.get("source_sha256") != worker.get("sha256"):
        errors.append("receipt source hash mismatch")
    expected_raw = f"observations/{output}/run.json"
    expected_directory = f"observations/{output}"
    if receipt.get("raw_result") != expected_raw:
        errors.append("receipt raw_result does not name the published run")
    if receipt.get("raw_directory") != expected_directory:
        errors.append("receipt raw_directory does not name the published directory")
    if receipt.get("evidence_seal") != "complete":
        errors.append("receipt evidence seal is not complete")
    return errors


def _strict_run(
    run: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    mode: str,
    worker: Mapping[str, Any],
    private_raw: bool,
) -> list[str]:
    """Validate the actual worker schema, before or after measurement_view."""
    errors: list[str] = []
    required = {"schema", "case", "mode", "status", "source_sha256", "rows", "errors", "finalization"}
    if private_raw:
        required.add("source_path")
    missing = sorted(required.difference(run))
    if missing:
        errors.append(("raw" if private_raw else "publication") + " missing fields: " + ",".join(missing))
        return errors
    if not isinstance(run.get("schema"), str) or not run["schema"]:
        errors.append("run schema is missing or empty")
    elif case.get("schema") is not None and run.get("schema") != case.get("schema"):
        errors.append("run schema does not match case schema")
    if run.get("case") != case.get("case_id"):
        errors.append("run case mismatch")
    if run.get("mode") != mode:
        errors.append("run mode mismatch")
    if run.get("status") != "completed":
        errors.append("run status is not completed")
    if run.get("source_sha256") != worker.get("sha256"):
        errors.append("run source hash mismatch")
    if private_raw and (not isinstance(run.get("source_path"), str) or not run["source_path"]):
        errors.append("raw source_path is missing or empty")
    rows = run.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(item, Mapping) for item in rows):
        errors.append("run rows must be a non-empty list of objects")
    expected_row_count = case.get("mode_row_counts", {}).get(mode) if isinstance(case.get("mode_row_counts"), Mapping) else None
    if expected_row_count is not None and isinstance(rows, list) and len(rows) != expected_row_count:
        errors.append("run rows count does not match the fixed case mode contract")
    if run.get("errors") != []:
        errors.append("run errors are not explicitly empty")
    finalization = run.get("finalization")
    if not isinstance(finalization, Mapping):
        errors.append("run finalization is missing")
    else:
        if finalization.get("status") != "completed":
            errors.append("run finalization status is not completed")
        if finalization.get("issues") != []:
            errors.append("run finalization issues are not explicitly empty")
        if type(finalization.get("rows_observed")) is not int:
            errors.append("run finalization rows_observed is missing or invalid")
        elif isinstance(rows, list) and finalization.get("rows_observed") != len(rows):
            errors.append("run finalization rows_observed does not match rows")
        for key in ("failed_observation_checks", "errors"):
            if key in finalization and finalization.get(key) != []:
                errors.append(f"run finalization {key} is not explicitly empty")
    errors.extend(
        f"run contains non-empty failure field: {item}"
        for item in _failure_walk(run)
        if item not in {"finalization.issues", "finalization.failed_observation_checks", "finalization.errors", "errors"}
    )
    errors.extend(f"run contains explicit observation failure: {item}" for item in _explicit_observation_failures(run))
    return errors


def _strict_publication(run: Mapping[str, Any], *, case: Mapping[str, Any], mode: str, worker: Mapping[str, Any]) -> list[str]:
    """Validate the public measurement view, including its real schema."""
    return _strict_run(run, case=case, mode=mode, worker=worker, private_raw=False)


def _validate_event_file(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"events line {index} is not JSON")
                continue
            if isinstance(event, Mapping):
                if event.get("status") == "failed":
                    errors.append(f"events line {index} reports failed status")
                if _failure_walk(event):
                    errors.append(f"events line {index} contains non-empty failure fields")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"cannot read events: {exc}")
    return errors


def _validate_job_row(
    proof_root: Path,
    row: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    worker: Mapping[str, Any],
    protocol: str,
) -> list[str]:
    errors: list[str] = []
    required = {
        "mode",
        "output",
        "request_id",
        "receipt_path",
        "receipt_sha256",
        "receipt",
        "process_path",
        "process_sha256",
        "publication_path",
        "raw_path",
        "private_raw_path",
        "publication_sha256",
        "raw_sha256",
        "private_raw_sha256",
        "view_check",
        "validation",
    }
    missing = sorted(required.difference(row))
    if missing:
        return ["row missing fields: " + ",".join(missing)]
    mode = row.get("mode")
    output = row.get("output")
    if not isinstance(mode, str) or mode not in case.get("modes", []):
        errors.append("row mode is not an admitted case mode")
    if not isinstance(output, str) or not SAFE_OUTPUT_RE.fullmatch(output):
        errors.append("row output is not a safe basename")
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        errors.append("row request_id is invalid")
    if errors:
        return errors
    try:
        receipt_path, receipt_hash = _path_sha(proof_root, row["receipt_path"], "row.receipt_path")
        process_path, process_hash = _path_sha(proof_root, row["process_path"], "row.process_path")
        publication_path, publication_hash = _path_sha(proof_root, row["publication_path"], "row.publication_path")
        raw_path, raw_hash = _path_sha(proof_root, row["raw_path"], "row.raw_path")
        private_raw_path, private_raw_hash = _path_sha(proof_root, row["private_raw_path"], "row.private_raw_path")
    except ContractError as exc:
        return [str(exc)]
    if receipt_hash != row.get("receipt_sha256"):
        errors.append("receipt hash does not match current bytes")
    if process_hash != row.get("process_sha256"):
        errors.append("process hash does not match current bytes")
    if publication_hash != row.get("publication_sha256"):
        errors.append("publication hash does not match current bytes")
    if raw_hash != row.get("raw_sha256"):
        errors.append("raw hash does not match current bytes")
    if private_raw_hash != row.get("private_raw_sha256"):
        errors.append("private raw hash does not match current bytes")
    if row.get("raw_path") != row.get("private_raw_path"):
        errors.append("raw and private raw paths differ")
    if row.get("raw_sha256") != row.get("private_raw_sha256"):
        errors.append("raw and private raw hashes differ")
    receipt = _json_object(receipt_path, "receipt")
    if receipt != row.get("receipt"):
        errors.append("proof receipt snapshot differs from current receipt")
    errors.extend(_strict_receipt(receipt, mode=mode, output=output, worker=worker, protocol=protocol))
    if receipt.get("request_id") != request_id:
        errors.append("receipt request_id mismatch")
    process = _json_object(process_path, "process record")
    if process.get("request_id") != request_id:
        errors.append("process request_id mismatch")
    if process.get("source_sha256_expected") != worker.get("sha256"):
        errors.append("process expected source hash mismatch")
    if process.get("source_sha256_after") != worker.get("sha256"):
        errors.append("process after source hash mismatch")
    if process.get("status") != "completed" or process.get("returncode") != 0 or process.get("worker_launched") is not True:
        errors.append("process record is not a completed zero-exit worker")
    private_run = _json_object(private_raw_path, "private raw run")
    errors.extend(_strict_run(private_run, case=case, mode=mode, worker=worker, private_raw=True))
    run = _json_object(publication_path, "published run")
    errors.extend(_strict_publication(run, case=case, mode=mode, worker=worker))
    try:
        from evals.stage1e.execution_transport import measurement_view

        if measurement_view(private_run) != run:
            errors.append("public publication is not the measurement view of raw run")
    except Exception as exc:
        errors.append(f"cannot recompute measurement view: {exc!r}")
    if not isinstance(row.get("view_check"), Mapping) or row["view_check"].get("passed") is not True:
        errors.append("view check did not pass")
    events_value = row.get("events_path")
    if events_value is not None:
        try:
            events_path, events_hash = _path_sha(proof_root, events_value, "row.events_path")
            if events_hash != row.get("events_sha256"):
                errors.append("events hash does not match current bytes")
            errors.extend(_validate_event_file(events_path))
        except ContractError as exc:
            errors.append(str(exc))
    validation = row.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        errors.append("row validation did not pass")
    return errors


def _worker_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("python", "source", "sha256", "modes", "max_jobs", "timeout_seconds", "output_argument"):
        if key in expected and actual.get(key) != expected.get(key):
            errors.append(f"worker {key} mismatch")
    return errors


def _proof_sidecar(proof_path: Path) -> Path:
    return proof_path.with_name(proof_path.name + ".sha256")


def _write_proof(proof_path: Path, proof: Mapping[str, Any]) -> Path:
    _json_write(proof_path, dict(proof))
    _proof_sidecar(proof_path).write_text(_sha256(proof_path) + "\n", encoding="ascii")
    return proof_path


def _write_synthetic_proof(proof_path: Path, proof: Mapping[str, Any]) -> Path:
    """Test-only writer; it does not make synthetic evidence trustworthy."""
    return _write_proof(Path(proof_path), proof)


def _make_row(
    *,
    root: Path,
    work: Path,
    evidence: Path,
    request_id: str,
    mode: str,
    output: str,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    case: Mapping[str, Any],
    worker: Mapping[str, Any],
    protocol: str,
) -> dict[str, Any]:
    job_root = evidence / "evaluator-tools" / request_id
    process_path = job_root / "process.json"
    publication_path = work / "observations" / output / "run.json"
    raw_candidates = (
        job_root / "work-output" / "run.json",
        job_root / "private-work" / "work-output" / "run.json",
    )
    private_raw_path = next((candidate for candidate in raw_candidates if _is_regular(candidate)), raw_candidates[0])
    events_path = work / "observations" / output / "events.jsonl"
    validation_errors: list[str] = []
    if _is_regular(process_path) and _is_regular(publication_path) and _is_regular(private_raw_path) and _is_regular(receipt_path):
        try:
            process = _json_object(process_path, "process")
            raw_run = _json_object(private_raw_path, "raw run")
            run = _json_object(publication_path, "publication")
            validation_errors.extend(_strict_receipt(receipt, mode=mode, output=output, worker=worker, protocol=protocol))
            if process.get("status") != "completed" or process.get("returncode") != 0:
                validation_errors.append("process is not completed with returncode 0")
            validation_errors.extend(_strict_run(raw_run, case=case, mode=mode, worker=worker, private_raw=True))
            validation_errors.extend(_strict_publication(run, case=case, mode=mode, worker=worker))
            from evals.stage1e.execution_transport import measurement_view

            view_passed = measurement_view(raw_run) == run
            if not view_passed:
                validation_errors.append("public publication is not the measurement view of raw run")
        except ContractError as exc:
            validation_errors.append(str(exc))
            view_passed = False
    else:
        view_passed = False
        if not _is_regular(process_path):
            validation_errors.append("process record is missing")
        if not _is_regular(publication_path):
            validation_errors.append("publication run.json is missing")
        if not _is_regular(private_raw_path):
            validation_errors.append("private raw run.json is missing")
        if not _is_regular(receipt_path):
            validation_errors.append("receipt is missing")
    result: dict[str, Any] = {
        "mode": mode,
        "output": output,
        "request_id": request_id,
        "receipt_path": _relative(root, receipt_path),
        "receipt_sha256": _sha256(receipt_path) if _is_regular(receipt_path) else None,
        "receipt": deepcopy(dict(receipt)),
        "process_path": _relative(root, process_path),
        "process_sha256": _sha256(process_path) if _is_regular(process_path) else None,
        "publication_path": _relative(root, publication_path),
        "raw_path": _relative(root, private_raw_path),
        "private_raw_path": _relative(root, private_raw_path),
        "publication_sha256": _sha256(publication_path) if _is_regular(publication_path) else None,
        "raw_sha256": _sha256(private_raw_path) if _is_regular(private_raw_path) else None,
        "private_raw_sha256": _sha256(private_raw_path) if _is_regular(private_raw_path) else None,
        "view_check": {"passed": view_passed},
        "validation": {"passed": not validation_errors, "errors": validation_errors},
    }
    if _is_regular(events_path):
        result["events_path"] = _relative(root, events_path)
        result["events_sha256"] = _sha256(events_path)
    return result


def preflight_case(
    case: Mapping[str, Any] | str,
    worker: Mapping[str, Any],
    transport: Mapping[str, Any] | None = None,
    client: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    output_root: Path | str | None = None,
    *,
    valid_for_seconds: int | float = DEFAULT_VALID_FOR_SECONDS,
    wait_seconds: int | float | None = None,
) -> Path:
    """Run every fixed mode through the real MeasurementServer and write proof.

    The function returns a proof path for both PASS and FAIL preflights.  A
    worker failure is evidence and remains available for inspection; only
    malformed setup contracts raise before a packet is started.
    """
    if output_root is None:
        raise ContractError("output_root is required")
    if type(valid_for_seconds) not in (int, float) or valid_for_seconds <= 0:
        raise ContractError("valid_for_seconds must be positive")
    root = Path(output_root).resolve()
    if root.exists():
        raise ContractError(f"refuse to overwrite existing preflight root: {root}")
    root.mkdir(parents=True)
    proof_path = root / "preflight-proof.json"
    worker_value, source_path = _normalise_worker(worker)
    case_value = _normalise_case(case, worker_value["modes"])
    default_transport, default_client = _default_tool_specs()
    transport_value = _normalise_tool(transport or default_transport, "transport", default_transport["module"])
    client_value = _normalise_tool(client or default_client, "client", default_client["module"])
    if transport_value.get("protocol") not in {None, TRANSPORT_PROTOCOL}:
        raise ContractError("unsupported transport protocol")
    if client_value.get("protocol") not in {None, TRANSPORT_PROTOCOL}:
        raise ContractError("unsupported client protocol")
    runtime_value = _collect_runtime_observations(_runtime_record(runtime, worker_value["python"]))

    # Importing here keeps CPU proof validation usable on Windows without
    # importing the POSIX-only server at module import time.
    from evals.stage1e import execution_transport
    from evals.stage1e.measure import create_request, validate_receipt

    if execution_transport.PROTOCOL != TRANSPORT_PROTOCOL:
        raise ContractError("Stage 1E transport protocol changed")
    actual_transport_path = Path(execution_transport.__file__).resolve()
    actual_client_module = __import__("evals.stage1e.measure", fromlist=["measure"])
    actual_client_path = Path(actual_client_module.__file__).resolve()
    if transport_value["module"] != "evals.stage1e.execution_transport" or Path(transport_value["path"]).resolve() != actual_transport_path:
        raise ContractError("transport declaration does not identify the imported MeasurementServer")
    if client_value["module"] != "evals.stage1e.measure" or Path(client_value["path"]).resolve() != actual_client_path:
        raise ContractError("client declaration does not identify the imported Stage 1E client")
    work = root / "packet"
    evidence = root / "private-evidence"
    (work / "requests").mkdir(parents=True)
    (work / "observations" / ".staging").mkdir(parents=True)
    modes = {mode: worker_value["modes"][mode] for mode in case_value["modes"]}
    measurement = {
        "protocol": execution_transport.PROTOCOL,
        "modes": list(modes),
        "wait_seconds": float(wait_seconds if wait_seconds is not None else worker_value["timeout_seconds"] + 10),
        "source_sha256": worker_value["sha256"],
    }
    _json_write(work / "measurement.json", measurement)
    worker_for_server = dict(worker_value)
    worker_for_server["modes"] = modes
    stop = threading.Event()
    server = execution_transport.MeasurementServer(
        work,
        evidence,
        worker_for_server,
        stop,
        time.monotonic() + max(60.0, float(measurement["wait_seconds"]) * len(modes) + 30.0),
    )
    server.start()
    rows: list[dict[str, Any]] = []
    try:
        for index, mode in enumerate(case_value["modes"], start=1):
            output = f"{case_value['case_id'].lower()}-{index:02d}-{mode}"
            request_id = None
            receipt_path = work / "observations" / "receipt-unknown.json"
            receipt: dict[str, Any] = {
                "protocol": TRANSPORT_PROTOCOL,
                "request_id": None,
                "mode": mode,
                "status": "failed",
                "error": "receipt unavailable",
            }
            try:
                request_id, receipt_path, config = create_request(work, mode, output)
                receipt = _wait_for_json(receipt_path, float(measurement["wait_seconds"]))
                try:
                    validate_receipt(receipt, request_id, mode, output, config)
                except Exception as exc:
                    # Retain the client validator failure as evidence; the
                    # strict Stage 2A validator decides acceptance later.
                    receipt = dict(receipt)
                    receipt.setdefault("client_validation_error", repr(exc))
            except Exception as exc:
                receipt["error"] = f"preflight request failure: {exc!r}"
            if request_id is None:
                # No valid request id means no path can be fabricated.  Keep a
                # synthetic row-shaped failure that require_preflight rejects.
                row = {
                    "mode": mode,
                    "output": output,
                    "request_id": "0" * 32,
                    "receipt_path": _relative(root, receipt_path),
                    "receipt_sha256": _sha256(receipt_path) if _is_regular(receipt_path) else None,
                    "receipt": receipt,
                    "process_path": _relative(root, evidence / "missing-process.json"),
                    "process_sha256": None,
                    "publication_path": _relative(root, work / "observations" / output / "run.json"),
                    "raw_path": _relative(root, evidence / "missing-raw.json"),
                    "private_raw_path": _relative(root, evidence / "missing-raw.json"),
                    "publication_sha256": None,
                    "raw_sha256": None,
                    "private_raw_sha256": None,
                    "view_check": {"passed": False},
                    "validation": {"passed": False, "errors": ["request did not produce a receipt"]},
                }
            else:
                row = _make_row(
                    root=root,
                    work=work,
                    evidence=evidence,
                    request_id=request_id,
                    mode=mode,
                    output=output,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    case=case_value,
                    worker=worker_value,
                    protocol=execution_transport.PROTOCOL,
                )
            rows.append(row)
    finally:
        stop.set()
        server.join(10)

    # Re-read every binding after all workers have finished.  A source/tool
    # mutation is a failed proof, never a warning or a defaulted field.
    binding_errors: list[str] = []
    try:
        if _sha256(source_path) != worker_value["sha256"]:
            binding_errors.append("worker source changed during preflight")
        if _sha256(Path(transport_value["path"])) != transport_value["sha256"]:
            binding_errors.append("transport tool changed during preflight")
        if _sha256(Path(client_value["path"])) != client_value["sha256"]:
            binding_errors.append("client tool changed during preflight")
    except ContractError as exc:
        binding_errors.append(str(exc))
    proof = {
        "proof_version": PROOF_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS" if not binding_errors and all(row["validation"]["passed"] for row in rows) else "FAIL",
        "created_at": _timestamp(_now()),
        "expires_at": _timestamp(_now() + timedelta(seconds=float(valid_for_seconds))),
        "case": case_value,
        "worker": worker_value,
        "transport": transport_value,
        "client": client_value,
        "runtime": runtime_value,
        "packet_root": "packet",
        "private_evidence_root": "private-evidence",
        "modes": list(case_value["modes"]),
        "rows": rows,
        "server": {
            "protocol": execution_transport.PROTOCOL,
            "jobs": server.jobs,
            "launched_jobs": server.launched_jobs,
            "rejections": server.rejections,
            "request_errors": server.request_errors,
            "last_error": server.last_error,
            "stopped": not server._thread or not server._thread.is_alive(),
        },
        "binding_errors": binding_errors,
        "scope": "Stage2A transport preflight; no Agent/model run",
    }
    _write_proof(proof_path, proof)
    return proof_path


def _validate_proof(proof_path: Path, worker_spec: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_regular(proof_path):
        raise ContractError(f"proof is not a regular file: {proof_path}")
    sidecar = _proof_sidecar(proof_path)
    if not _is_regular(sidecar):
        raise ContractError("proof sidecar seal is missing")
    expected_sidecar = sidecar.read_text(encoding="ascii").strip()
    if expected_sidecar != _sha256(proof_path):
        raise ContractError("proof sidecar seal does not match proof bytes")
    proof = _json_object(proof_path, "proof")
    for key in ("proof_version", "contract_version", "status", "created_at", "expires_at", "case", "worker", "transport", "client", "runtime", "packet_root", "rows", "modes"):
        if key not in proof:
            raise ContractError(f"proof missing field: {key}")
    if proof["proof_version"] != PROOF_VERSION or proof["contract_version"] != CONTRACT_VERSION:
        raise ContractError("proof version mismatch")
    if proof["status"] != "PASS":
        raise ContractError(f"preflight proof status is {proof['status']!r}")
    now = _now()
    created = _parse_timestamp(proof["created_at"], "proof.created_at")
    expires = _parse_timestamp(proof["expires_at"], "proof.expires_at")
    if expires <= created:
        raise ContractError("proof expiry is not after creation")
    if now >= expires:
        raise ContractError("preflight proof has expired")
    if created > now + timedelta(minutes=5):
        raise ContractError("preflight proof creation time is in the future")
    worker, source_path = _normalise_worker(proof["worker"])
    expected_worker, _ = _normalise_worker(worker_spec)
    errors = _worker_matches(expected_worker, worker)
    if errors:
        raise ContractError("; ".join(errors))
    case = proof["case"]
    if not isinstance(case, Mapping):
        raise ContractError("proof.case is not an object")
    case_value = _normalise_case(case, worker["modes"])
    if dict(case_value) != dict(case):
        raise ContractError("proof case identity contains unapproved fields or values")
    modes = proof["modes"]
    if modes != case_value["modes"]:
        raise ContractError("proof modes do not match case modes")
    transport = proof["transport"]
    client = proof["client"]
    runtime = proof["runtime"]
    if not isinstance(transport, Mapping) or not isinstance(client, Mapping) or not isinstance(runtime, Mapping):
        raise ContractError("proof tool/runtime bindings are not objects")
    transport_path = _absolute_file(transport.get("path"), "proof.transport.path")
    client_path = _absolute_file(client.get("path"), "proof.client.path")
    from evals.stage1e import execution_transport as current_transport
    current_client = __import__("evals.stage1e.measure", fromlist=["measure"])
    if transport.get("module") != "evals.stage1e.execution_transport" or transport_path != Path(current_transport.__file__).resolve():
        raise ContractError("proof transport does not identify the current imported MeasurementServer")
    if client.get("module") != "evals.stage1e.measure" or client_path != Path(current_client.__file__).resolve():
        raise ContractError("proof client does not identify the current imported Stage 1E client")
    if _sha256(transport_path) != transport.get("sha256"):
        raise ContractError("transport tool bytes changed after preflight")
    if _sha256(client_path) != client.get("sha256"):
        raise ContractError("client tool bytes changed after preflight")
    if transport.get("protocol") != TRANSPORT_PROTOCOL or client.get("protocol") != TRANSPORT_PROTOCOL:
        raise ContractError("proof transport/client protocol is not the fixed Stage 1E protocol")
    for tool_name, tool in (("transport", transport), ("client", client)):
        identity = dict(tool)
        declared_identity = identity.pop("identity_sha256", None)
        if not isinstance(declared_identity, str) or hashlib.sha256(_canonical(identity)).hexdigest() != declared_identity:
            raise ContractError(f"proof {tool_name} identity seal is invalid")
    runtime_python = _absolute_file(runtime.get("python"), "proof.runtime.python", reject_symlink=False)
    if _sha256(runtime_python) != runtime.get("python_sha256"):
        raise ContractError("runtime executable changed after preflight")
    current_launcher = _current_launcher_path()
    worker_launcher = Path(os.path.abspath(worker["python"]))
    declared_launcher = runtime.get("launcher", runtime.get("python"))
    if "launcher" in runtime:
        if Path(os.path.abspath(str(declared_launcher))) != worker_launcher:
            raise ContractError("proof runtime and worker launcher differ")
    elif Path(os.path.abspath(str(declared_launcher))).resolve(strict=False) != worker_launcher.resolve(strict=False):
        raise ContractError("proof runtime executable target differs from worker launcher")
    if worker_launcher != current_launcher:
        raise ContractError("proof worker launcher is not the current process virtualenv launcher")
    if str(runtime_python.resolve(strict=False)) != str(worker_launcher.resolve(strict=False)):
        raise ContractError("proof runtime executable target differs from worker launcher")
    runtime_identity = dict(runtime)
    declared_runtime_identity = runtime_identity.pop("identity_sha256", None)
    if not isinstance(declared_runtime_identity, str) or hashlib.sha256(_canonical(runtime_identity)).hexdigest() != declared_runtime_identity:
        raise ContractError("proof runtime identity seal is invalid")
    root = proof_path.parent
    rows = proof["rows"]
    if not isinstance(rows, list) or len(rows) != len(modes):
        raise ContractError("proof rows do not cover every case mode exactly once")
    seen: set[str] = set()
    row_errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            row_errors.append("proof row is not an object")
            continue
        mode = row.get("mode")
        if mode in seen:
            row_errors.append(f"duplicate proof row for mode {mode}")
        if isinstance(mode, str):
            seen.add(mode)
        row_errors.extend(f"{mode}: {error}" for error in _validate_job_row(root, row, case=case_value, worker=worker, protocol=str(transport.get("protocol", TRANSPORT_PROTOCOL))))
    if seen != set(modes):
        row_errors.append("proof rows do not cover exactly the admitted modes")
    if proof.get("binding_errors") not in (None, []):
        row_errors.append("proof contains binding errors")
    if row_errors:
        raise ContractError("; ".join(row_errors))
    return proof


def require_preflight(proof: Path | str, worker_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Require a current, independently reread PASS proof for a fixed worker."""
    return _validate_proof(Path(proof).resolve(), worker_spec)


def freeze_case(
    proof: Path | str,
    worker_spec: Mapping[str, Any],
    output: Path | str,
    *,
    prepare: Callable[..., Any] | None = None,
    prepare_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Gate a future case freeze before invoking an optional old preparer.

    ``prepare`` is an intentionally injected callable: the Stage 2A gate owns
    proof validation, while the existing Stage 1F packet builder remains the
    caller's explicit implementation.  A missing or stale proof therefore
    fails before that callable can be entered.
    """
    validated = require_preflight(proof, worker_spec)
    target = Path(output).resolve()
    if target.exists():
        raise ContractError(f"refuse to overwrite case freeze output: {target}")
    target.mkdir(parents=True)
    callback_result = None
    if prepare is not None:
        callback_result = prepare(**dict(prepare_kwargs or {}))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN",
        "case": validated["case"],
        "worker": validated["worker"],
        "preflight_proof": _relative(target, Path(proof).resolve()) if Path(proof).resolve().is_relative_to(target) else str(Path(proof).resolve()),
        "preflight_proof_sha256": _sha256(Path(proof).resolve()),
        "preflight_scope": validated["scope"],
        "delegate_result": str(callback_result) if isinstance(callback_result, Path) else callback_result,
    }
    manifest_path = target / "case-manifest.json"
    _json_write(manifest_path, manifest)
    return manifest_path


def _load_json_argument(path: str) -> Any:
    return _json_read(Path(path).resolve())


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2A fixed worker preflight contract")
    sub = parser.add_subparsers(dest="command", required=True)
    p_preflight = sub.add_parser("preflight", help="run all fixed modes through MeasurementServer")
    p_preflight.add_argument("--case", required=True, type=Path, help="case JSON")
    p_preflight.add_argument("--worker", required=True, type=Path, help="fixed worker JSON")
    p_preflight.add_argument("--transport", type=Path)
    p_preflight.add_argument("--client", type=Path)
    p_preflight.add_argument("--runtime", type=Path)
    p_preflight.add_argument("--output-root", required=True, type=Path)
    p_preflight.add_argument("--valid-for-seconds", type=float, default=DEFAULT_VALID_FOR_SECONDS)
    p_require = sub.add_parser("require", help="validate a preflight proof and current worker")
    p_require.add_argument("--proof", required=True, type=Path)
    p_require.add_argument("--worker", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            case = _load_json_argument(str(args.case))
            worker = _load_json_argument(str(args.worker))
            transport = _load_json_argument(str(args.transport)) if args.transport else None
            client = _load_json_argument(str(args.client)) if args.client else None
            runtime = _load_json_argument(str(args.runtime)) if args.runtime else None
            path = preflight_case(case, worker, transport, client, runtime, args.output_root, valid_for_seconds=args.valid_for_seconds)
            proof = _json_read(path)
            print(json.dumps({"proof": str(path), "status": proof.get("status")}, ensure_ascii=False))
            return 0 if proof.get("status") == "PASS" else 1
        proof = require_preflight(args.proof, _load_json_argument(str(args.worker)))
        print(json.dumps({"proof": str(args.proof), "status": proof["status"], "modes": proof["modes"]}, ensure_ascii=False))
        return 0
    except ContractError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by integration command
    raise SystemExit(_main())
