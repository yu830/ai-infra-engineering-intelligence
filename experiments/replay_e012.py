"""Portable, offline numerical audit for the public E012 development fixture.

The public fixture contains observations produced by the first-party E012
runner.  This module consumes only those published values.  It does not import
PyTorch, start a model, use CUDA, reopen a frozen ``original_path`` or inspect
an Agent trace.  The result is an offline numerical readback plus a small,
mechanical case-evidence score; it is deliberately not the private Agent
semantic rubric.

Both a fixture directory and a deterministic ZIP are accepted.  In either
case the input is copied or extracted into a fresh temporary directory before
reading.  ``frozen-coverage.json`` is the only authority for resolving the
three observation roles.  ``original_path`` is treated as an opaque URI and is
never converted to a filesystem path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Iterator
import zipfile


FIXTURE_SCHEMA = "stage2a-e012-public-fixture-v1"
MANIFEST_SCHEMA = "stage2a-public-delivery-manifest-v1"
PROVENANCE_SCHEMA = "stage2a-e012-public-provenance-v1"
OBSERVATION_SCHEMA = "stage2a-e012-public-observation-v1"
EXPECTED_SCHEMA = "stage2a-e012-expected-score-v1"
DELIVERY_MANIFEST = "DELIVERY-MANIFEST.json"
FROZEN_COVERAGE = "frozen-coverage.json"
EXPECTED_SCORE = "expected-score.json"
PROVENANCE = "provenance.json"
CHUNK_SIZE = 1024 * 1024
ROLES = ("baseline", "path-control", "reference-control")
ROLE_ALIASES = {
    "baseline": "baseline",
    "path-control": "path-control",
    "path_control": "path-control",
    "reference": "reference-control",
    "reference-control": "reference-control",
    "reference_control": "reference-control",
}


class ReplayFailure(ValueError):
    """Raised when a fixture is incomplete, inconsistent or unsafe to audit."""


class NonStandardJSON(ReplayFailure):
    """Raised for JSON NaN/Infinity or another non-standard JSON token."""


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayFailure(message)


def _reject_constant(token: str) -> Any:
    raise NonStandardJSON(f"non-standard JSON numeric token: {token}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayFailure(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NonStandardJSON(f"non-finite JSON number at {path}; use explicit string NaN/Inf/-Inf in values_flat")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def strict_json_bytes(data: bytes, label: str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReplayFailure(f"{label} is not UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ReplayFailure) as exc:
        if isinstance(exc, ReplayFailure):
            raise
        raise ReplayFailure(f"invalid JSON: {label}") from exc
    _reject_nonfinite(value)
    return value


def strict_json_file(path: Path) -> Any:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReplayFailure(f"cannot read JSON member: {path}") from exc
    return strict_json_bytes(data, path.as_posix())


def _validate_member_name(name: str) -> None:
    _require(isinstance(name, str) and bool(name), "empty package member name")
    _require("\\" not in name, f"non-POSIX package member name: {name!r}")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), f"absolute package member name: {name!r}")
    _require(".." not in path.parts, f"parent traversal package member name: {name!r}")
    _require(name != ".", f"invalid package member name: {name!r}")


def _relative_path(root: Path, member: str) -> Path:
    """Resolve a package member below *root* after validating its POSIX name."""

    _validate_member_name(member)
    candidate = root.joinpath(*PurePosixPath(member).parts)
    root_resolved = root.resolve()
    try:
        candidate_resolved = candidate.resolve(strict=True)
        candidate_resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ReplayFailure(f"package member is missing or escapes staged root: {member!r}") from exc
    _require(candidate_resolved.is_file(), f"package member is not a regular file: {member!r}")
    _require(not candidate.is_symlink(), f"symlink package member: {member!r}")
    return candidate_resolved


def _ensure_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReplayFailure(f"symlinks are not allowed in fixture package: {path}")


def _hex_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} must be a 64-character SHA-256")
    _require(value == value.lower() and all(char in "0123456789abcdef" for char in value), f"{label} is not lowercase hexadecimal")
    return value


def _scan_for_private_text(value: Any, path: str = "$") -> None:
    """Reject common machine/Agent metadata accidentally copied to public data."""

    forbidden_key_fragments = (
        "argv",
        "pid",
        "uid",
        "trace",
        "agent",
        "prompt",
        "issue_body",
        "formal_answer",
        "source_path",
        "python_executable",
        "events_file",
        "command",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in forbidden_key_fragments):
                raise ReplayFailure(f"private/Agent field is not allowed in public fixture: {path}.{key}")
            _scan_for_private_text(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_for_private_text(item, f"{path}[{index}]")
    elif isinstance(value, str):
        # URLs and case:// URIs are allowed.  Local absolute paths are not.
        if re.search(r"(?i)(?:^[A-Z]:[\\/]|\\\\|/(?:mnt|home|tmp|Users|var)/)", value):
            raise ReplayFailure(f"local/private path is not allowed in public fixture: {path}")


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    _require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    if nonnegative:
        _require(result >= 0.0, f"{label} must be non-negative")
    return result


def _number_or_special(value: Any, label: str) -> float:
    if isinstance(value, str):
        _require(value in {"NaN", "Inf", "-Inf"}, f"{label} has invalid special value")
        return {"NaN": math.nan, "Inf": math.inf, "-Inf": -math.inf}[value]
    _require(type(value) in (int, float), f"{label} must be a number or explicit NaN/Inf string")
    result = float(value)
    _require(math.isfinite(result), f"{label} must use explicit NaN/Inf strings")
    return result


def _decode_values(values: Any, label: str) -> list[float]:
    _require(isinstance(values, list), f"{label} must be a list")
    return [_number_or_special(value, f"{label}[{index}]") for index, value in enumerate(values)]


def _shape(value: Any, label: str) -> list[int]:
    _require(isinstance(value, list) and value, f"{label} must be a non-empty list")
    result: list[int] = []
    for index, item in enumerate(value):
        _require(type(item) is int and item > 0, f"{label}[{index}] must be a positive integer")
        result.append(item)
    return result


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _rows(values: list[float], shape: list[int]) -> tuple[list[float], list[float]]:
    row_size = _product(shape[1:]) if len(shape) > 1 else 1
    _require(len(values) % shape[0] == 0, "values_flat cannot be partitioned into first-axis rows")
    return values[:row_size], values[row_size : 2 * row_size]


def _descriptor(value: Any, label: str) -> tuple[dict[str, Any], list[float]]:
    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {
        "shape",
        "dtype",
        "device",
        "numel",
        "nan_count",
        "inf_count",
        "finite",
        "row1_finite",
        "row1_absmax_finite_values",
        "values_flat",
    }
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    shape = _shape(value.get("shape"), f"{label}.shape")
    values = _decode_values(value.get("values_flat"), f"{label}.values_flat")
    expected_numel = _product(shape)
    _require(type(value.get("numel")) is int and value["numel"] == expected_numel, f"{label}.numel mismatch")
    _require(len(values) == expected_numel, f"{label}.values_flat length mismatch")
    nan_count = sum(math.isnan(item) for item in values)
    inf_count = sum(math.isinf(item) for item in values)
    finite = all(math.isfinite(item) for item in values)
    row0, row1 = _rows(values, shape)
    row1_finite = all(math.isfinite(item) for item in row1)
    finite_row1_values = [abs(item) for item in row1 if math.isfinite(item)]
    row1_absmax = max(finite_row1_values, default=0.0)
    _require(type(value.get("nan_count")) is int and value["nan_count"] == nan_count, f"{label}.nan_count mismatch")
    _require(type(value.get("inf_count")) is int and value["inf_count"] == inf_count, f"{label}.inf_count mismatch")
    _require(type(value.get("finite")) is bool and value["finite"] == finite, f"{label}.finite mismatch")
    _require(type(value.get("row1_finite")) is bool and value["row1_finite"] == row1_finite, f"{label}.row1_finite mismatch")
    declared_absmax = _finite_number(value.get("row1_absmax_finite_values"), f"{label}.row1_absmax_finite_values")
    _require(math.isclose(declared_absmax, row1_absmax, rel_tol=1e-6, abs_tol=1e-6), f"{label}.row1_absmax mismatch")
    if "dtype" in value:
        _require(isinstance(value["dtype"], str) and value["dtype"], f"{label}.dtype must be text")
    if "device" in value:
        _require(isinstance(value["device"], str) and value["device"], f"{label}.device must be text")
    return {"shape": shape, "values": values, "row0": row0, "row1": row1, "finite": finite, "row1_finite": row1_finite, "nan_count": nan_count, "inf_count": inf_count, "row1_absmax": row1_absmax}, values


def _comparison(value: Any, left: dict[str, Any], right: dict[str, Any], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {"passed", "max_abs_diff", "mean_abs_diff", "atol", "rtol"}
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    _require(type(value.get("passed")) is bool, f"{label}.passed must be boolean")
    atol = _finite_number(value.get("atol"), f"{label}.atol", nonnegative=True)
    rtol = _finite_number(value.get("rtol"), f"{label}.rtol", nonnegative=True)
    _require(left["shape"] == right["shape"], f"{label} array shape mismatch")
    left_values, right_values = left["values"], right["values"]
    _require(len(left_values) == len(right_values), f"{label} array length mismatch")
    finite_pair = all(math.isfinite(a) and math.isfinite(b) for a, b in zip(left_values, right_values))
    if finite_pair:
        deltas = [abs(a - b) for a, b in zip(left_values, right_values)]
        max_diff = max(deltas, default=0.0)
        mean_diff = sum(deltas) / len(deltas) if deltas else 0.0
        expected_passed = all(delta <= atol + rtol * abs(b) for delta, b in zip(deltas, right_values))
        declared_max = _finite_number(value.get("max_abs_diff"), f"{label}.max_abs_diff")
        declared_mean = _finite_number(value.get("mean_abs_diff"), f"{label}.mean_abs_diff")
        # max_abs_diff follows the prior coordinator readback bound.  The
        # looser mean bound covers float32 subtraction/accumulation ordering;
        # neither changes the scientific allclose tolerance above.
        _require(math.isclose(declared_max, max_diff, rel_tol=1e-7, abs_tol=1e-12), f"{label}.max_abs_diff mismatch")
        _require(math.isclose(declared_mean, mean_diff, rel_tol=1e-5, abs_tol=1e-12), f"{label}.mean_abs_diff mismatch")
        _require(value["passed"] == expected_passed, f"{label}.passed mismatch")
        return {"passed": expected_passed, "max_abs_diff": max_diff, "mean_abs_diff": mean_diff, "atol": atol, "rtol": rtol}

    # A non-finite comparison is a valid recorded observation, but it cannot
    # establish reference agreement.  The producer must make that explicit.
    _require(value["passed"] is False, f"{label} must fail when either array is non-finite")
    _require(value.get("max_abs_diff") in (None, "NaN"), f"{label}.max_abs_diff must be null/NaN for non-finite arrays")
    _require(value.get("mean_abs_diff") in (None, "NaN"), f"{label}.mean_abs_diff must be null/NaN for non-finite arrays")
    return {"passed": False, "max_abs_diff": None, "mean_abs_diff": None, "atol": atol, "rtol": rtol}


def _output(value: Any, label: str, expected_shape: list[int] | None = None) -> tuple[dict[str, Any], list[float]]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return _descriptor(value, label)


def _validate_input(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {"shape", "dtype", "cpu_input_sha256", "model_state_sha256", "key_padding_mask", "fully_masked_rows"}
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    shape = _shape(value.get("shape"), f"{label}.shape")
    _require(isinstance(value.get("dtype"), str) and value["dtype"], f"{label}.dtype must be text")
    _hex_digest(value.get("cpu_input_sha256"), f"{label}.cpu_input_sha256")
    _hex_digest(value.get("model_state_sha256"), f"{label}.model_state_sha256")
    mask = value.get("key_padding_mask")
    _require(isinstance(mask, list) and len(mask) == shape[0], f"{label}.key_padding_mask batch mismatch")
    for row_index, row in enumerate(mask):
        _require(isinstance(row, list) and len(row) == (shape[1] if len(shape) > 1 else 1), f"{label}.key_padding_mask row mismatch")
        _require(all(type(item) is bool for item in row), f"{label}.key_padding_mask must contain booleans")
    fully_masked = value.get("fully_masked_rows")
    _require(isinstance(fully_masked, list) and all(type(item) is int and item >= 0 for item in fully_masked), f"{label}.fully_masked_rows must be integer list")
    calculated = [index for index, row in enumerate(mask) if all(row)]
    _require(fully_masked == calculated, f"{label}.fully_masked_rows mismatch")
    return {
        "shape": shape,
        "dtype": value["dtype"],
        "mask": mask,
        "fully_masked_rows": calculated,
        "cpu_input_sha256": value["cpu_input_sha256"],
        "model_state_sha256": value["model_state_sha256"],
    }


def _validate_details(value: Any, label: str) -> dict[str, Any]:
    """Validate reference-policy metadata without treating it as recomputed evidence."""

    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {"attention_layers"}
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    layers = value.get("attention_layers")
    _require(isinstance(layers, list) and layers, f"{label}.attention_layers must be a non-empty list")
    for index, layer in enumerate(layers):
        _require(isinstance(layer, dict), f"{label}.attention_layers[{index}] must be an object")
        _require(set(layer).issubset({"weights_shape", "all_masked_weight_absmax", "all_masked_weights_zero"}), f"unexpected fields in {label}.attention_layers[{index}]")
        shape = layer.get("weights_shape")
        if shape is not None:
            _shape(shape, f"{label}.attention_layers[{index}].weights_shape")
        _finite_number(layer.get("all_masked_weight_absmax"), f"{label}.attention_layers[{index}].all_masked_weight_absmax", nonnegative=True)
        _require(type(layer.get("all_masked_weights_zero")) is bool, f"{label}.attention_layers[{index}].all_masked_weights_zero must be boolean")
    return {"layers": len(layers), "policy": "DECLARED_REFERENCE_POLICY_ONLY"}


def _validate_config(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {"shape", "dtype", "reference_tolerance", "seed", "model_state_sha256"}
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    shape = _shape(value.get("shape"), f"{label}.shape")
    _require(isinstance(value.get("dtype"), str) and value["dtype"], f"{label}.dtype must be text")
    tolerance = value.get("reference_tolerance")
    _require(isinstance(tolerance, dict) and set(tolerance) == {"atol", "rtol"}, f"{label}.reference_tolerance must have atol/rtol")
    atol = _finite_number(tolerance["atol"], f"{label}.reference_tolerance.atol", nonnegative=True)
    rtol = _finite_number(tolerance["rtol"], f"{label}.reference_tolerance.rtol", nonnegative=True)
    _require(atol == 2e-4 and rtol == 2e-4, f"{label}.reference_tolerance must be the fixed E012 2e-4/2e-4 contract")
    _require(type(value.get("seed")) is int and value["seed"] >= 0, f"{label}.seed must be a non-negative integer")
    _hex_digest(value.get("model_state_sha256"), f"{label}.model_state_sha256")
    return {
        "shape": shape,
        "dtype": value["dtype"],
        "tolerance": tolerance,
        "seed": value["seed"],
        "model_state_sha256": value["model_state_sha256"],
    }


def _observation(value: Any, role: str, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    allowed = {"schema", "case", "role", "observation_status", "source_sha256", "config", "rows"}
    _require(set(value).issubset(allowed), f"unexpected fields in {label}: {sorted(set(value) - allowed)}")
    _require(value.get("schema") == OBSERVATION_SCHEMA, f"{label}.schema mismatch")
    _require(value.get("case") == "E012", f"{label}.case mismatch")
    _require(value.get("role") == role, f"{label}.role mismatch")
    _require(value.get("observation_status") == "GPU_OBSERVED", f"{label} is not a published GPU observation")
    source_hash = _hex_digest(value.get("source_sha256"), f"{label}.source_sha256")
    config = _validate_config(value.get("config"), f"{label}.config")
    rows = value.get("rows")
    _require(isinstance(rows, list) and rows, f"{label}.rows must be a non-empty list")
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_label = f"{label}.rows[{index}]"
        _require(isinstance(row, dict), f"{row_label} must be an object")
        allowed_row = {"mask_kind", "input", "fastpath", "slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu"}
        _require(set(row).issubset(allowed_row), f"unexpected fields in {row_label}: {sorted(set(row) - allowed_row)}")
        mask_kind = row.get("mask_kind")
        _require(mask_kind in {"all", "partial", "none"}, f"{row_label}.mask_kind is invalid")
        input_data = _validate_input(row.get("input"), f"{row_label}.input")
        _require(input_data["shape"] == config["shape"], f"{row_label}.input.shape/config.shape mismatch")
        parsed: dict[str, Any] = {"mask_kind": mask_kind, "input": input_data}
        output_keys = ("fastpath", "slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu")
        for output_key in output_keys:
            item = row.get(output_key)
            if item is None:
                parsed[output_key] = None
                continue
            _require(isinstance(item, dict), f"{row_label}.{output_key} must be an object or null")
            if output_key == "slowpath_intervention":
                _require(set(item).issubset({"fastpath_enabled", "output"}), f"unexpected fields in {row_label}.{output_key}")
                _require(set(item) == {"fastpath_enabled", "output"}, f"missing fields in {row_label}.{output_key}")
                _require(item.get("fastpath_enabled") is False, f"{row_label}.{output_key} is not disabled-fastpath intervention")
                output = item.get("output")
            elif output_key in {"independent_reference_cpu", "independent_reference_gpu"}:
                comparison_key = "slow_vs_cpu_reference" if output_key.endswith("cpu") else "slow_vs_gpu_reference"
                _require(set(item) == {"output", comparison_key, "details"}, f"reference fields in {row_label}.{output_key} are incomplete or inconsistent")
                output = item.get("output")
            else:
                _require(set(item) == {"output"}, f"unexpected fields in {row_label}.{output_key}")
                output = item.get("output")
            parsed_output, _ = _output(output, f"{row_label}.{output_key}.output")
            _require(parsed_output["shape"] == config["shape"], f"{row_label}.{output_key}.output.shape mismatch")
            parsed[output_key] = {"output": parsed_output}
            if output_key in {"independent_reference_cpu", "independent_reference_gpu"}:
                details = item.get("details")
                _require(details is not None, f"{row_label}.{output_key}.details is required")
                parsed[output_key]["details"] = _validate_details(details, f"{row_label}.{output_key}.details")
                comparison = item.get(comparison_key)
                if comparison is not None:
                    slow_item = row.get("slowpath_intervention")
                    _require(slow_item is not None, f"{row_label}.{comparison_key} requires slowpath_intervention")
                    slow_parsed, _ = _output(slow_item.get("output"), f"{row_label}.slowpath_intervention.output")
                    checked_comparison = _comparison(comparison, slow_parsed, parsed_output, f"{row_label}.{comparison_key}")
                    _require(checked_comparison["atol"] == config["tolerance"]["atol"], f"{row_label}.{comparison_key}.atol disagrees with config.reference_tolerance")
                    _require(checked_comparison["rtol"] == config["tolerance"]["rtol"], f"{row_label}.{comparison_key}.rtol disagrees with config.reference_tolerance")
                    parsed[output_key][comparison_key] = checked_comparison
                else:
                    raise ReplayFailure(f"{row_label}.{comparison_key} is required")
        parsed_rows.append(parsed)
    expected_masks = {
        "baseline": ["all"],
        "path-control": ["all"],
        "reference-control": ["all", "partial", "none"],
    }[role]
    actual_masks = [row["mask_kind"] for row in parsed_rows]
    _require(actual_masks == expected_masks, f"{label} mask rows must be exactly {expected_masks!r}, got {actual_masks!r}")
    for row in parsed_rows:
        required = {
            "baseline": {"fastpath"},
            "path-control": {"fastpath", "slowpath_intervention"},
            "reference-control": {"slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu"},
        }[role]
        if role == "reference-control" and row["mask_kind"] in {"partial", "none"}:
            required = required | {"fastpath"}
        for key in output_keys:
            if key in required:
                _require(row[key] is not None, f"{label}.{row['mask_kind']}.{key} is required by role contract")
            else:
                _require(row[key] is None, f"{label}.{row['mask_kind']}.{key} is not allowed by role contract")
    return {"role": role, "source_sha256": source_hash, "config": config, "rows": parsed_rows}


def _manifest(root: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    manifest_path = _relative_path(root, DELIVERY_MANIFEST)
    manifest = strict_json_file(manifest_path)
    _require(isinstance(manifest, dict), "DELIVERY-MANIFEST.json must be an object")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "delivery manifest schema mismatch")
    _require(manifest.get("fixture_schema") == FIXTURE_SCHEMA, "delivery manifest fixture schema mismatch")
    _require(manifest.get("case") == "E012", "delivery manifest case mismatch")
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "delivery manifest files must be a non-empty list")
    payloads: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(files):
        _require(isinstance(item, dict), f"manifest files[{index}] must be an object")
        name = item.get("path")
        _validate_member_name(name)
        _require(name != DELIVERY_MANIFEST, "manifest cannot list itself")
        _require(name not in payloads, f"duplicate manifest payload: {name}")
        _require(type(item.get("size_bytes")) is int and item["size_bytes"] >= 0, f"invalid size for manifest payload: {name}")
        digest = _hex_digest(item.get("sha256"), f"manifest SHA-256 for {name}")
        path = _relative_path(root, name)
        _require(path.stat().st_size == item["size_bytes"], f"manifest size mismatch: {name}")
        _require(sha_file(path) == digest, f"manifest SHA-256 mismatch: {name}")
        payloads[name] = {"path": path, "sha256": digest, "size_bytes": item["size_bytes"]}

    _require(FROZEN_COVERAGE in payloads, "manifest must list frozen-coverage.json")
    _require(EXPECTED_SCORE in payloads, "manifest must list expected-score.json")
    _require(PROVENANCE in payloads, "manifest must list provenance.json")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(actual_files == set(payloads) | {DELIVERY_MANIFEST}, "staged files differ from manifest payload set")
    coverage = strict_json_file(payloads[FROZEN_COVERAGE]["path"])
    _require(isinstance(coverage, list), "frozen-coverage.json must be a list")
    _require(type(manifest.get("frozen_paths")) is int and manifest["frozen_paths"] == len(coverage), "manifest frozen_paths disagrees with coverage")
    _require(type(manifest.get("external_binary_fingerprints")) is int and manifest["external_binary_fingerprints"] == 0, "public fixture cannot contain external binary fingerprints")
    role_paths: dict[str, Path] = {}
    role_members: dict[str, str] = {}
    seen_origins: set[str] = set()
    seen_members: set[str] = set()
    for index, item in enumerate(coverage):
        _require(isinstance(item, dict), f"coverage[{index}] must be an object")
        origin = item.get("original_path")
        _require(isinstance(origin, str) and origin.startswith("case://E012/"), f"coverage[{index}].original_path must be an opaque E012 URI")
        _require(origin not in seen_origins, f"duplicate coverage original_path: {origin}")
        seen_origins.add(origin)
        role_name = ROLE_ALIASES.get(origin.removeprefix("case://E012/"))
        _require(role_name in ROLES, f"coverage[{index}] has unknown E012 role URI: {origin}")
        _require(role_name not in role_paths, f"duplicate E012 role mapping: {role_name}")
        member = item.get("archived_as")
        _require(isinstance(member, str) and member, f"coverage[{index}].archived_as must be a member name")
        _validate_member_name(member)
        _require(member in payloads, f"coverage[{index}].archived_as is not a manifest payload: {member}")
        _require(member not in seen_members, f"duplicate coverage archived_as: {member}")
        seen_members.add(member)
        digest = _hex_digest(item.get("sha256"), f"coverage[{index}].sha256")
        _require(payloads[member]["sha256"] == digest, f"coverage/payload SHA mismatch: {member}")
        role_paths[role_name] = payloads[member]["path"]
        role_members[role_name] = member
    _require(set(role_paths) == set(ROLES), f"coverage must map exactly {ROLES}")
    _require(len(coverage) == len(ROLES), "public coverage must contain exactly three role mappings")
    return manifest, role_paths, role_members


def _provenance(root: Path, path: Path) -> dict[str, Any]:
    value = strict_json_file(path)
    _require(isinstance(value, dict), "provenance.json must be an object")
    _require(value.get("schema") == PROVENANCE_SCHEMA, "provenance schema mismatch")
    _require(value.get("case") == "E012", "provenance case mismatch")
    _require(value.get("observation_status") == "GPU_OBSERVED", "provenance must identify a real observed GPU source")
    _hex_digest(value.get("worker_source_sha256"), "provenance.worker_source_sha256")
    _require(value.get("exact_original_environment_replay") == "NOT_RUN", "provenance must preserve exact-replay limitation")
    _require(isinstance(value.get("runner"), str) and value["runner"].startswith("first-party"), "provenance.runner must identify first-party runner")
    _require(isinstance(value.get("upstream_url"), str) and value["upstream_url"].startswith("https://"), "provenance.upstream_url must be an HTTPS URL")
    runtime = value.get("runtime")
    _require(isinstance(runtime, dict), "provenance.runtime must be an object")
    _require(set(runtime).issubset({"python", "torch", "cuda", "device_class", "cuda_observed"}), "provenance.runtime contains non-minimal environment fields")
    for key in ("python", "torch", "cuda", "device_class"):
        _require(isinstance(runtime.get(key), str) and runtime[key], f"provenance.runtime.{key} is required")
    _require(runtime.get("cuda_observed") is True, "provenance cannot claim PASS for an unobserved CUDA run")
    _scan_for_private_text(value, "provenance")
    return value


def _expected_score(path: Path) -> dict[str, Any]:
    value = strict_json_file(path)
    _require(isinstance(value, dict), "expected-score.json must be an object")
    _require(value.get("schema") == EXPECTED_SCHEMA, "expected-score schema mismatch")
    _require(value.get("case") == "E012", "expected-score case mismatch")
    _require(value.get("fixture_kind") == "development_example", "expected-score must be a development example")
    criteria = value.get("criteria")
    _require(isinstance(criteria, dict), "expected-score.criteria must be an object")
    _require(set(criteria) == {"fully_masked_fastpath_nonfinite_region", "slowpath_intervention_finite_region", "reference_error_stats_and_allclose", "negative_controls_finite_and_reference_agree", "fallback_encoder_output_nonzero"}, "expected-score criteria set mismatch")
    for key, item in criteria.items():
        _require(isinstance(item, dict), f"expected-score.criteria.{key} must be an object")
        _require(type(item.get("passed")) is bool, f"expected-score.criteria.{key}.passed must be boolean")
        _require(isinstance(item.get("description"), str) and item["description"], f"expected-score.criteria.{key}.description is required")
    expected = value.get("expected")
    _require(isinstance(expected, dict), "expected-score.expected must be an object")
    _require(type(expected.get("score")) is int and 0 <= expected["score"] <= 5, "expected-score.expected.score is invalid")
    _require(expected.get("max_score") == 5, "expected-score.expected.max_score must be 5")
    _require(expected.get("status") in {"PASS_DEVELOPMENT_EXAMPLE", "NOT_REPRODUCED_DEVELOPMENT_EXAMPLE", "PARTIAL_CASE_EVIDENCE"}, "expected-score.expected.status is invalid")
    return value


def _find_row(observations: dict[str, dict[str, Any]], role: str, mask_kind: str) -> dict[str, Any]:
    for row in observations[role]["rows"]:
        if row["mask_kind"] == mask_kind:
            return row
    raise ReplayFailure(f"{role} observation has no {mask_kind!r} row")


def score_case_evidence(observations: dict[str, dict[str, Any]], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score five mechanical E012 evidence criteria from recomputed arrays.

    The score is case evidence only.  It does not score an Agent, private
    semantic reasoning, or the Stage 1F 2/2 trajectory rubric.
    """

    all_baseline = _find_row(observations, "baseline", "all")
    all_path = _find_row(observations, "path-control", "all")
    all_reference = _find_row(observations, "reference-control", "all")
    partial_reference = _find_row(observations, "reference-control", "partial")
    none_reference = _find_row(observations, "reference-control", "none")
    fast_baseline = all_baseline["fastpath"]["output"]
    slow_path = all_path["slowpath_intervention"]["output"]
    slow_reference = all_reference["slowpath_intervention"]["output"]
    cpu_reference = all_reference["independent_reference_cpu"]["output"]
    gpu_reference = all_reference["independent_reference_gpu"]["output"]

    criteria = {
        "fully_masked_fastpath_nonfinite_region": (
            fast_baseline["nan_count"] + fast_baseline["inf_count"] > 0
            and not all(math.isfinite(item) for item in fast_baseline["row1"])
            and all(math.isfinite(item) for item in fast_baseline["row0"])
        ),
        "slowpath_intervention_finite_region": (
            slow_path["finite"] and slow_path["nan_count"] == 0 and slow_path["inf_count"] == 0 and slow_path["row1_finite"]
        ),
        "reference_error_stats_and_allclose": (
            all_reference["independent_reference_cpu"]["slow_vs_cpu_reference"]["passed"]
            and all_reference["independent_reference_gpu"]["slow_vs_gpu_reference"]["passed"]
            and slow_reference["finite"] and cpu_reference["finite"] and gpu_reference["finite"]
        ),
        "negative_controls_finite_and_reference_agree": (
            all(partial_reference[key]["output"]["finite"] for key in ("fastpath", "slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu"))
            and all(none_reference[key]["output"]["finite"] for key in ("fastpath", "slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu"))
            and partial_reference["independent_reference_cpu"]["slow_vs_cpu_reference"]["passed"]
            and partial_reference["independent_reference_gpu"]["slow_vs_gpu_reference"]["passed"]
            and none_reference["independent_reference_cpu"]["slow_vs_cpu_reference"]["passed"]
            and none_reference["independent_reference_gpu"]["slow_vs_gpu_reference"]["passed"]
        ),
        "fallback_encoder_output_nonzero": (
            slow_reference["row1_finite"] and slow_reference["row1_absmax"] > 0.0
        ),
    }
    result = {
        "schema": "stage2a-e012-case-evidence-score-v1",
        "scope": "mechanical case evidence only; no Agent semantic score, private gold or trajectory replay",
        "criteria": {key: {"passed": bool(value)} for key, value in criteria.items()},
        "score": sum(criteria.values()),
        "max_score": len(criteria),
    }
    result["status"] = "PASS_DEVELOPMENT_EXAMPLE" if result["score"] == result["max_score"] else "NOT_REPRODUCED" if not criteria["fully_masked_fastpath_nonfinite_region"] else "PARTIAL_CASE_EVIDENCE"
    if expected is not None:
        expected_criteria = expected["criteria"]
        for key, actual in result["criteria"].items():
            _require(actual["passed"] == expected_criteria[key]["passed"], f"expected-score criterion mismatch: {key}")
        expected_total = expected["expected"]
        _require(result["score"] == expected_total["score"], "expected-score total mismatch")
        _require(result["max_score"] == expected_total["max_score"], "expected-score max mismatch")
        if expected_total["status"] == "PASS_DEVELOPMENT_EXAMPLE":
            _require(result["status"] == "PASS_DEVELOPMENT_EXAMPLE", "expected-score status mismatch")
        elif expected_total["status"] == "NOT_REPRODUCED_DEVELOPMENT_EXAMPLE":
            _require(result["status"] == "NOT_REPRODUCED", "expected-score non-reproduction status mismatch")
        else:
            _require(result["status"] == "PARTIAL_CASE_EVIDENCE", "expected-score partial status mismatch")
        result["expected_score_audit"] = "PASS_COMPUTED_SCORE_MATCHES_EXPECTED"
    return result


def _validate_cross_role_identity(observations: dict[str, dict[str, Any]]) -> str:
    """Check shared E012 config/input identity and return the common source hash."""

    _require(set(observations) == set(ROLES), "all three E012 observation roles are required")
    source_hashes = {observations[role]["source_sha256"] for role in ROLES}
    _require(len(source_hashes) == 1, "observation source hashes disagree")
    source_hash = next(iter(source_hashes))
    reference_config = observations["reference-control"]["config"]
    for role in ROLES:
        config = observations[role]["config"]
        _require(config["shape"] == reference_config["shape"], f"{role} config shape disagrees across roles")
        _require(config["dtype"] == reference_config["dtype"], f"{role} config dtype disagrees across roles")
        _require(config["seed"] == reference_config["seed"], f"{role} config seed disagrees across roles")
        _require(config["tolerance"] == reference_config["tolerance"], f"{role} reference tolerance disagrees across roles")
        _require(config["model_state_sha256"] == reference_config["model_state_sha256"], f"{role} model state hash disagrees across roles")
    by_mask = {row["mask_kind"]: row["input"] for row in observations["reference-control"]["rows"]}
    for role in ROLES:
        for row in observations[role]["rows"]:
            mask_kind = row["mask_kind"]
            _require(mask_kind in by_mask, f"{role} has an input mask absent from reference-control: {mask_kind}")
            reference_input = by_mask[mask_kind]
            for field in ("shape", "dtype", "mask", "fully_masked_rows", "cpu_input_sha256", "model_state_sha256"):
                _require(row["input"][field] == reference_input[field], f"{role}/{mask_kind} input identity disagrees with reference-control ({field})")
    return source_hash


@contextmanager
def _staged_source(source: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Copy/extract *source* into a fresh temp directory and scrub PYTHONPATH."""

    source = Path(source)
    _require(not source.is_symlink(), f"fixture source symlink is not allowed: {source}")
    _require(source.exists(), f"fixture source does not exist: {source}")
    old_pythonpath = os.environ.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="stage2a-e012-replay-") as temp_name:
            staged = Path(temp_name) / "fixture"
            package_info: dict[str, Any] = {"source_kind": "directory" if source.is_dir() else "zip", "extracted": False}
            if source.is_dir():
                _ensure_no_symlinks(source)
                shutil.copytree(source, staged, symlinks=False)
                package_info["copied"] = True
            elif source.is_file():
                _require(zipfile.is_zipfile(source), f"fixture source is neither a directory nor a ZIP: {source}")
                # Reuse the Stage 1 payload/hash/coverage verifier before staged
                # extraction.  It never opens original_path values.
                try:
                    if __package__:
                        from .verify_evidence_zip import verify_archive
                    else:  # direct ``python experiments/replay_e012.py`` execution
                        from verify_evidence_zip import verify_archive
                    verify_archive(source)
                except ImportError as exc:
                    raise ReplayFailure("verify_evidence_zip.py is required beside replay_e012.py") from exc
                staged.mkdir(parents=True, exist_ok=False)
                try:
                    with zipfile.ZipFile(source, "r") as archive:
                        names = [info.filename for info in archive.infolist()]
                        _require(len(names) == len(set(names)), "duplicate ZIP member name")
                        for info in archive.infolist():
                            _validate_member_name(info.filename)
                            mode = (info.external_attr >> 16) & 0o170000
                            _require(mode != 0o120000, f"symlink ZIP member is not allowed: {info.filename}")
                            target = staged.joinpath(*PurePosixPath(info.filename).parts)
                            target_parent = target.parent
                            target_parent.mkdir(parents=True, exist_ok=True)
                            if info.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                            else:
                                with archive.open(info, "r") as src, target.open("xb") as dst:
                                    shutil.copyfileobj(src, dst, CHUNK_SIZE)
                except (OSError, zipfile.BadZipFile) as exc:
                    raise ReplayFailure(f"failed to extract fixture ZIP: {source}") from exc
                _ensure_no_symlinks(staged)
                package_info["extracted"] = True
            else:
                raise ReplayFailure(f"fixture source is not a regular directory/ZIP: {source}")
            yield staged, package_info
    finally:
        if old_pythonpath is not None:
            os.environ["PYTHONPATH"] = old_pythonpath


def audit_fixture(source: Path | str, output: Path | str | None = None) -> dict[str, Any]:
    """Audit a public E012 fixture directory or ZIP in an isolated temp copy."""

    source = Path(source)
    with _staged_source(source) as (root, package_info):
        manifest, role_paths, role_members = _manifest(root)
        provenance = _provenance(root, _relative_path(root, PROVENANCE))
        expected = _expected_score(_relative_path(root, EXPECTED_SCORE))
        _scan_for_private_text(manifest, "manifest")
        manifest_expected = manifest.get("expected_score")
        _require(
            isinstance(manifest_expected, dict)
            and manifest_expected.get("score") == expected["expected"]["score"]
            and manifest_expected.get("max_score") == expected["expected"]["max_score"],
            "manifest expected_score disagrees with standalone expected-score.json",
        )
        observations: dict[str, dict[str, Any]] = {}
        for role in ROLES:
            value = strict_json_file(role_paths[role])
            _scan_for_private_text(value, role)
            parsed = _observation(value, role, role_members[role])
            observations[role] = parsed
        source_hash = _validate_cross_role_identity(observations)
        _require(source_hash == provenance["worker_source_sha256"], "observation/provenance worker source hash mismatch")
        score = score_case_evidence(observations, expected)
        result = {
            "status": "PASS_OFFLINE_SCIENTIFIC_AUDIT" if score["status"] == "PASS_DEVELOPMENT_EXAMPLE" else score["status"],
            "scope": "offline numeric readback in a fresh copied/extracted directory; no GPU/model/Agent semantic replay",
            "case": "E012",
            "fixture_schema": FIXTURE_SCHEMA,
            "source_sha256": source_hash,
            "source_kind": package_info["source_kind"],
            "checks": {
                "manifest_payload_hashes": "PASS",
                "coverage_mapping": "PASS",
                "strict_json": "PASS",
                "copied_or_extracted_to_fresh_temp": True,
                "pythonpath_removed_during_audit": True,
                "original_paths_reopened": False,
                "gpu_reexecution": False,
                "model_reexecution": False,
                "agent_semantics_replayed": False,
                "reference_policy_metadata": "DECLARED_ONLY_NOT_RECOMPUTED",
                "comparison_summary_recompute": "PASS; max rel=1e-7 abs=1e-12, mean rel=1e-5 abs=1e-12 for float32 summary accumulation",
            },
            "mapping": {role: {"original_path": f"case://E012/{role}", "archived_as": role_members[role]} for role in ROLES},
            "score": score,
            "expected_score_audit": score.get("expected_score_audit"),
            "limitations": [
                "values_flat statistics and comparisons are independently recomputed",
                "stored source/input/model hashes identify observations but are not rehashed against hidden inputs",
                "all_masked_weights_zero is a declared reference-policy metadata field, not an independent weight replay",
                "this does not replay private Agent trajectories or the Stage 1F semantic 2/2 rubric",
                "E012 is already exposed; this fixture is a development example, not held-out gold",
            ],
        }
    if output is not None:
        output_path = Path(output)
        _require(not output_path.exists(), f"refuse to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _reference_expected(source: Path | str) -> dict[str, Any]:
    """Audit then read the expected-score contract from a public fixture copy."""

    # Recompute the reference fixture before using its expected-score file.
    # A manifest/expected-score pair with matching hashes is still not a
    # scientific oracle if the full values_flat evidence no longer supports it.
    reference_result = audit_fixture(source)
    with _staged_source(Path(source)) as (root, _package_info):
        manifest, _role_paths, _role_members = _manifest(root)
        expected = _expected_score(_relative_path(root, EXPECTED_SCORE))
        manifest_expected = manifest.get("expected_score")
        _require(
            isinstance(manifest_expected, dict)
            and manifest_expected.get("score") == expected["expected"]["score"]
            and manifest_expected.get("max_score") == expected["expected"]["max_score"],
            "reference fixture expected score is internally inconsistent",
        )
        _require(reference_result["score"]["score"] == expected["expected"]["score"], "reference fixture computed score disagrees with expected-score.json")
        _require(reference_result["score"]["max_score"] == expected["expected"]["max_score"], "reference fixture computed max score disagrees with expected-score.json")
        return expected


def audit_raw_records(
    baseline: Path | str,
    path_control: Path | str,
    reference_control: Path | str,
    *,
    reference_fixture: Path | str | None = None,
    output: Path | str | None = None,
) -> dict[str, Any]:
    """Audit three first-party raw E012 run records in memory.

    This is an optional bridge for a fresh GPU/contract run.  It performs no
    GPU or model work and emits no raw arrays.  ``reference_fixture`` supplies
    an independently audited development expected-score contract for an
    informational comparison; a mismatch becomes an explicit field and does
    not convert a valid ``NOT_REPRODUCED`` observation into malformed data.
    """

    raw_paths = {
        "baseline": Path(baseline),
        "path-control": Path(path_control),
        "reference-control": Path(reference_control),
    }
    for role, path in raw_paths.items():
        _require(not path.is_symlink() and path.is_file(), f"raw {role} must be a regular non-symlink file")
    old_pythonpath = os.environ.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="stage2a-e012-raw-") as temp_name:
            staged_root = Path(temp_name)
            staged_paths: dict[str, Path] = {}
            for role, path in raw_paths.items():
                target = staged_root / f"{role}.json"
                shutil.copyfile(path, target)
                staged_paths[role] = target
            projected = {
                role: _project_raw_observation(strict_json_file(path), role)
                for role, path in staged_paths.items()
            }
    finally:
        if old_pythonpath is not None:
            os.environ["PYTHONPATH"] = old_pythonpath
    observations = {
        role: _observation(value, role, role)
        for role, value in projected.items()
    }
    source_hash = _validate_cross_role_identity(observations)
    score = score_case_evidence(observations)
    result: dict[str, Any] = {
        "status": "PASS_OFFLINE_RAW_AUDIT" if score["status"] == "PASS_DEVELOPMENT_EXAMPLE" else score["status"],
        "scope": "in-memory projection of three first-party raw records; no GPU/model/Agent replay and no raw arrays returned",
        "case": "E012",
        "source_sha256": source_hash,
        "raw_input_roles": list(ROLES),
        "checks": {
            "strict_json": "PASS",
            "role_contract": "PASS",
            "cross_role_identity": "PASS",
            "gpu_reexecution": False,
            "model_reexecution": False,
            "agent_semantics_replayed": False,
            "reference_policy_metadata": "DECLARED_ONLY_NOT_RECOMPUTED",
            "comparison_summary_recompute": "PASS; max rel=1e-7 abs=1e-12, mean rel=1e-5 abs=1e-12",
        },
        "score": score,
        "raw_observations_returned": False,
    }
    if reference_fixture is not None:
        expected = _reference_expected(reference_fixture)
        reference_criteria = {key: item["passed"] for key, item in expected["criteria"].items()}
        actual_criteria = {key: item["passed"] for key, item in score["criteria"].items()}
        result["reference_expected_score"] = {
            "source": "public development fixture expected-score.json",
            "score": expected["expected"],
            "criteria_match": actual_criteria == reference_criteria,
            "comparison_only": True,
        }
    if output is not None:
        output_path = Path(output)
        _require(not output_path.exists(), f"refuse to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _coverage_from_zip(archive_path: Path) -> tuple[zipfile.ZipFile, list[dict[str, Any]]]:
    try:
        archive = zipfile.ZipFile(archive_path, "r")
        coverage = strict_json_bytes(archive.read(FROZEN_COVERAGE), f"{archive_path}!{FROZEN_COVERAGE}")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ReplayFailure(f"cannot read legacy coverage from {archive_path}") from exc
    _require(isinstance(coverage, list), "legacy frozen-coverage.json must be a list")
    return archive, coverage


def audit_legacy_selection(archive: Path | str, selection: dict[str, str]) -> dict[str, Any]:
    """Audit three legacy ZIP members through coverage, without Path(original_path).

    This helper is intended for a private migration receipt.  It reads the
    selected bytes from ``archived_as`` members and projects them in memory to
    the same public observation shape.  It never writes or returns raw arrays.
    """

    archive_path = Path(archive)
    _require(isinstance(selection, dict) and set(selection) == set(ROLES), "legacy selection must name baseline/path-control/reference-control")
    try:
        from .verify_evidence_zip import verify_archive
    except ImportError:
        from verify_evidence_zip import verify_archive
    verify_archive(archive_path)
    bundle, coverage = _coverage_from_zip(archive_path)
    try:
        by_origin: dict[str, dict[str, Any]] = {}
        for item in coverage:
            if not isinstance(item, dict):
                raise ReplayFailure("legacy coverage entry is not an object")
            origin = item.get("original_path")
            if not isinstance(origin, str):
                raise ReplayFailure("legacy coverage original_path is not text")
            by_origin[origin] = item
        projected: dict[str, dict[str, Any]] = {}
        members: dict[str, str] = {}
        for role in ROLES:
            origin = selection[role]
            _require(isinstance(origin, str) and origin in by_origin, f"legacy selection is not a coverage original_path: {role}")
            item = by_origin[origin]
            member = item.get("archived_as")
            _require(isinstance(member, str) and member, f"legacy selected mapping is external/unmapped: {role}")
            _validate_member_name(member)
            raw = strict_json_bytes(bundle.read(member), f"{archive_path}!{member}")
            _require(isinstance(raw, dict), f"legacy selected observation is not an object: {role}")
            # Selection is known to be E012 raw run.json.  Projection is local
            # and is intentionally not serialized or returned.
            projected[role] = _project_raw_observation(raw, role, legacy=True)
            members[role] = member
        observations = {role: _observation(projected[role], role, members[role]) for role in ROLES}
        _validate_cross_role_identity(observations)
        score = score_case_evidence(observations)
        return {
            "status": "PASS_LEGACY_MAPPING_OFFLINE_REPLAY" if score["score"] == score["max_score"] else score["status"],
            "scope": "private migration receipt; selected ZIP bytes resolved through frozen-coverage only",
            "archive_sha256": sha_file(archive_path),
            "selected_original_paths": dict(selection),
            "selected_archived_as": members,
            "original_paths_reopened": False,
            "raw_observations_returned": False,
            "score": score,
        }
    finally:
        bundle.close()


def _project_raw_observation(raw: dict[str, Any], role: str, *, legacy: bool = False) -> dict[str, Any]:
    """Project a first-party E012 run JSON to the narrow public allowlist.

    Kept private so public audit callers cannot accidentally treat raw Agent or
    runtime metadata as part of the fixture contract.
    """

    _require(raw.get("schema") == "stage1f-e012-workload-v1", "raw observation schema mismatch")
    _require(raw.get("case") == "E012", "raw observation case mismatch")
    _require(raw.get("mode") == role, f"raw observation mode does not match {role}")
    _require(raw.get("status") == "completed", "raw observation is not completed")
    _require(type(raw.get("errors")) is list and raw["errors"] == [], "raw observation has errors")
    finalization = raw.get("finalization")
    _require(isinstance(finalization, dict) and finalization.get("status") == "completed", "raw observation finalization is not completed")
    if "issues" in finalization:
        _require(type(finalization["issues"]) is list and finalization["issues"] == [], "raw observation finalization has issues")
    else:
        _require(legacy, "raw observation finalization.issues is missing")
    _require(isinstance(raw.get("environment"), dict) and raw["environment"].get("cuda_available") is True, "raw observation lacks observed CUDA runtime")
    config = raw.get("config")
    _require(isinstance(config, dict), "raw observation config is missing")
    expected_config = {"d_model": 32, "n_head": 4, "ff_dim": 64, "batch": 2, "seq_len": 6, "n_layers": 2, "seed": 0}
    for key, expected_value in expected_config.items():
        _require(type(config.get(key)) is int and config[key] == expected_value, f"raw config.{key} must be the fixed E012 integer contract")
    tolerance = config.get("reference_tolerance")
    _require(isinstance(tolerance, dict), "raw config.reference_tolerance is missing")
    _require(tolerance.get("atol") == 2e-4 and tolerance.get("rtol") == 2e-4, "raw config.reference_tolerance must be fixed at 2e-4/2e-4")
    source_hash = _hex_digest(raw.get("source_sha256"), "raw source_sha256")
    keep_config = {
        "shape": [config["batch"], config["seq_len"], config["d_model"]],
        "dtype": None,
        "reference_tolerance": {"atol": tolerance["atol"], "rtol": tolerance["rtol"]},
        "seed": config["seed"],
        "model_state_sha256": "",
    }
    raw_rows = raw.get("rows")
    _require(isinstance(raw_rows, list) and raw_rows, "raw observation rows are missing")
    _require(type(finalization.get("rows_observed")) is int and finalization["rows_observed"] == len(raw_rows), "raw finalization rows_observed mismatch")
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ReplayFailure("raw row is not an object")
        input_data = row.get("input")
        _require(isinstance(input_data, dict), "raw row input is missing")
        _require(input_data.get("shape") == keep_config["shape"], "raw input shape disagrees with fixed E012 shape")
        _require(input_data.get("dtype") == "torch.float32", "raw input dtype must be supported torch.float32")
        if keep_config["dtype"] is None:
            keep_config["dtype"] = input_data["dtype"]
        _require(keep_config["dtype"] == input_data["dtype"], "raw input dtypes disagree")
        model_hash = _hex_digest(input_data.get("model_state_sha256"), "raw model_state_sha256")
        if not keep_config["model_state_sha256"]:
            keep_config["model_state_sha256"] = model_hash
        _require(keep_config["model_state_sha256"] == model_hash, "raw model state hashes disagree")
        for key in ("cpu_input_sha256", "key_padding_mask", "fully_masked_rows"):
            _require(key in input_data, f"raw row input.{key} is missing")
        keep_input = {key: input_data[key] for key in ("shape", "dtype", "cpu_input_sha256", "model_state_sha256", "key_padding_mask", "fully_masked_rows")}
        _require(row.get("mask_kind") in {"all", "partial", "none"}, "raw row mask_kind is invalid")
        keep_row: dict[str, Any] = {"mask_kind": row["mask_kind"], "input": keep_input}
        for key in ("fastpath", "slowpath_intervention", "independent_reference_cpu", "independent_reference_gpu"):
            item = row.get(key)
            if item is None:
                keep_row[key] = None
                continue
            if key == "fastpath":
                _require(isinstance(item, dict) and "output" in item, f"raw row {key} output is missing")
                keep_row[key] = {"output": item["output"]}
            elif key == "slowpath_intervention":
                _require(isinstance(item, dict) and "fastpath_enabled" in item and "output" in item, f"raw row {key} output is missing")
                keep_row[key] = {"fastpath_enabled": item["fastpath_enabled"], "output": item["output"]}
            else:
                comparison_key = "slow_vs_cpu_reference" if key.endswith("cpu") else "slow_vs_gpu_reference"
                _require(isinstance(item, dict) and "output" in item and comparison_key in item and "details" in item, f"raw row {key} evidence is incomplete")
                keep_row[key] = {"output": item["output"], comparison_key: item.get(comparison_key), "details": item.get("details")}
        rows.append(keep_row)
    _require(keep_config["model_state_sha256"], "raw model state hash is missing")
    return {
        "schema": OBSERVATION_SCHEMA,
        "case": "E012",
        "role": role,
        "observation_status": "GPU_OBSERVED",
        "source_sha256": source_hash,
        "config": keep_config,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline, stdlib-only E012 scientific evidence audit")
    parser.add_argument("fixture", type=Path, nargs="?", help="public fixture directory or deterministic ZIP")
    parser.add_argument("--raw-baseline", type=Path, help="first-party raw baseline run.json for in-memory projection")
    parser.add_argument("--raw-path-control", type=Path, help="first-party raw path-control run.json for in-memory projection")
    parser.add_argument("--raw-reference-control", type=Path, help="first-party raw reference-control run.json for in-memory projection")
    parser.add_argument("--reference", type=Path, help="public fixture used only for expected-score comparison in raw mode")
    parser.add_argument("--output", type=Path, help="write a new JSON receipt; existing output is never overwritten")
    args = parser.parse_args(argv)
    try:
        raw_paths = (args.raw_baseline, args.raw_path_control, args.raw_reference_control)
        if any(path is not None for path in raw_paths):
            _require(all(path is not None for path in raw_paths), "raw mode requires all three --raw-* paths")
            _require(args.fixture is None, "fixture positional input cannot be combined with --raw-* mode")
            result = audit_raw_records(
                args.raw_baseline,
                args.raw_path_control,
                args.raw_reference_control,
                reference_fixture=args.reference,
                output=args.output,
            )
        else:
            _require(args.fixture is not None, "provide a fixture directory/ZIP or all three --raw-* paths")
            _require(args.reference is None, "--reference is only valid with --raw-* mode")
            result = audit_fixture(args.fixture, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] in {"PASS_OFFLINE_SCIENTIFIC_AUDIT", "PASS_OFFLINE_RAW_AUDIT", "NOT_REPRODUCED", "PARTIAL_CASE_EVIDENCE"} else 1
    except ReplayFailure as exc:
        result = {
            "status": "FAIL_OFFLINE_SCIENTIFIC_AUDIT",
            "scope": "no scientific conclusion; malformed or unsafe fixture",
            "fixture": str(args.fixture) if args.fixture is not None else None,
            "error": str(exc),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
