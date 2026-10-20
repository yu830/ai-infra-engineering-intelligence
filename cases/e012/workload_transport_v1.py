"""Self-contained GPU qualification for PyTorch issue #199054 (E012).

The upstream report describes NaNs from the Transformer/MultiheadAttention
fastpath when a key-padding-mask fully masks one row.  The worker runs the
reported dimensions without shrinking them, compares the fastpath with the
slowpath intervention, and computes an independent safe-softmax reference in
plain Python/Torch operations.  It also runs a partial-mask negative control.

No project-local module, evaluator file, or Agent proposal is imported.  The
worker writes only to the fresh absolute output directory passed by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


CASE = "E012"
SCHEMA = "stage1f-e012-workload-v1"
D_MODEL = 32
N_HEAD = 4
FF_DIM = 64
BATCH = 2
SEQ_LEN = 6
N_LAYERS = 2
SEED = 0
VALID_MODES = ("baseline", "path-control", "reference-control", "qualification")
REFERENCE_ATOL = 2e-4
REFERENCE_RTOL = 2e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_output(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("--output must be an absolute path")
    if path == Path(path.anchor):
        raise ValueError("--output must name a task directory")
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")
    return path


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _error_record(exc: BaseException) -> dict[str, Any]:
    return {"error": repr(exc), "traceback": traceback.format_exc()}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        return {"argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:  # pragma: no cover - host dependent
        return {"argv": argv, "error": repr(exc), "traceback": traceback.format_exc()}


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        result.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "device_total_memory_bytes": int(props.total_memory),
                "mem_get_info": {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)},
                "nvidia_smi": _run_command(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total,memory.free",
                        "--format=csv,noheader",
                    ]
                ),
            }
        )
    return result


def _build_encoder(device: torch.device | str) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=N_HEAD,
        dim_feedforward=FF_DIM,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=N_LAYERS, enable_nested_tensor=False).to(device).eval()


def _mask(device: torch.device | str, mask_kind: str) -> torch.Tensor:
    key_padding_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool, device=device)
    if mask_kind == "all":
        key_padding_mask[1, :] = True
    elif mask_kind == "partial":
        key_padding_mask[1, SEQ_LEN // 2 :] = True
    elif mask_kind == "none":
        pass
    else:
        raise ValueError(f"unknown mask_kind: {mask_kind}")
    return key_padding_mask


def _describe(tensor: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    flattened = tensor.detach().cpu().reshape(-1).tolist()
    actual_values: list[float | str] = []
    for value in flattened:
        value = float(value)
        if math.isnan(value):
            actual_values.append("NaN")
        elif math.isinf(value):
            actual_values.append("Inf" if value > 0 else "-Inf")
        else:
            actual_values.append(value)
    return {
        "shape": [int(v) for v in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
        "finite": bool(finite.all().item()),
        "row1_finite": bool(finite[1].all().item()),
        "row1_absmax_finite_values": float(torch.nan_to_num(tensor[1]).abs().max().item()),
        "values_flat": actual_values,
    }


def _run_encoder(
    encoder: nn.TransformerEncoder,
    x: torch.Tensor,
    key_padding_mask: torch.Tensor,
    fastpath: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(fastpath)
    try:
        with torch.no_grad():
            output = encoder(x, src_key_padding_mask=key_padding_mask)
        torch.cuda.synchronize()
        return output, {"fastpath_enabled": bool(fastpath), "output": _describe(output)}
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def _manual_mha(
    module: nn.MultiheadAttention,
    values: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Independent self-attention with an explicit safe all-masked rule."""

    batch, length, embed_dim = values.shape
    head_dim = embed_dim // module.num_heads
    qkv = F.linear(values, module.in_proj_weight, module.in_proj_bias)
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.view(batch, length, module.num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, length, module.num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, length, module.num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) * (head_dim**-0.5)
    masked = key_padding_mask[:, None, None, :]
    scores = scores.masked_fill(masked, float("-inf"))
    all_masked = masked.all(dim=-1, keepdim=True)
    weights = torch.softmax(scores.float(), dim=-1).masked_fill(all_masked, 0.0).to(scores.dtype)
    attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, embed_dim)
    output = F.linear(attended, module.out_proj.weight, module.out_proj.bias)
    return output, {
        "weights_shape": [int(v) for v in weights.shape],
        "all_masked_weight_absmax": float(weights[1].abs().max().item()),
        "all_masked_weights_zero": bool(torch.equal(weights[1], torch.zeros_like(weights[1]))),
    }


def _manual_encoder(
    encoder: nn.TransformerEncoder,
    x: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    current = x
    attention_details: list[dict[str, Any]] = []
    for layer in encoder.layers:
        normalized = layer.norm1(current)
        attended, details = _manual_mha(layer.self_attn, normalized, key_padding_mask)
        current = current + layer.dropout1(attended)
        normalized = layer.norm2(current)
        feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized))))
        current = current + layer.dropout2(feedforward)
        attention_details.append(details)
    return current, {"attention_layers": attention_details}


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = (left.float() - right.float()).abs()
    passed = bool(torch.allclose(left, right, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL))
    return {
        "passed": passed,
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "atol": REFERENCE_ATOL,
        "rtol": REFERENCE_RTOL,
    }


def _case(
    encoder_gpu: nn.TransformerEncoder,
    x_cpu: torch.Tensor,
    mask_kind: str,
) -> dict[str, Any]:
    x_gpu = x_cpu.to(device="cuda")
    mask_gpu = _mask("cuda", mask_kind)
    fast, fast_details = _run_encoder(encoder_gpu, x_gpu, mask_gpu, True)
    slow, slow_details = _run_encoder(encoder_gpu, x_gpu, mask_gpu, False)

    encoder_cpu = _build_encoder("cpu")
    encoder_cpu.load_state_dict({key: value.detach().cpu() for key, value in encoder_gpu.state_dict().items()})
    mask_cpu = mask_gpu.detach().cpu()
    reference_cpu, reference_details = _manual_encoder(encoder_cpu, x_cpu, mask_cpu)
    reference_gpu, reference_gpu_details = _manual_encoder(encoder_gpu, x_gpu, mask_gpu)
    torch.cuda.synchronize()

    return {
        "mask_kind": mask_kind,
        "input": {
            "shape": [int(v) for v in x_gpu.shape],
            "dtype": str(x_gpu.dtype),
            "cpu_input_sha256": _tensor_sha256(x_cpu),
            "model_state_sha256": _state_dict_sha256(encoder_gpu),
            "key_padding_mask": mask_gpu.detach().cpu().tolist(),
            "fully_masked_rows": [int(i) for i in torch.where(mask_gpu.all(dim=-1))[0].tolist()],
        },
        "fastpath": fast_details,
        "slowpath_intervention": slow_details,
        "independent_reference_cpu": {
            "output": _describe(reference_cpu),
            "slow_vs_cpu_reference": _compare(slow.detach().cpu(), reference_cpu),
            "details": reference_details,
        },
        "independent_reference_gpu": {
            "output": _describe(reference_gpu),
            "slow_vs_gpu_reference": _compare(slow, reference_gpu),
            "details": reference_gpu_details,
        },
        "fast_vs_slow_finite_only": (
            _compare(torch.nan_to_num(fast), slow)
            if bool(torch.isfinite(fast).all().item())
            else {"passed": False, "reason": "fastpath output contains non-finite values"}
        ),
    }


def _case_scoped(
    encoder_gpu: nn.TransformerEncoder,
    x_cpu: torch.Tensor,
    mask_kind: str,
    *,
    include_fast: bool,
    include_slow: bool,
    include_reference: bool,
) -> dict[str, Any]:
    """Run only the observations allowed by a fixed public mode.

    The full qualification mode uses ``_case``.  Public control modes keep the
    baseline from receiving the later control/reference observations while
    still writing every value that was actually measured.
    """

    x_gpu = x_cpu.to(device="cuda")
    mask_gpu = _mask("cuda", mask_kind)
    fast = slow = None
    fast_details: dict[str, Any] | None = None
    slow_details: dict[str, Any] | None = None
    if include_fast:
        fast, fast_details = _run_encoder(encoder_gpu, x_gpu, mask_gpu, True)
    if include_slow:
        slow, slow_details = _run_encoder(encoder_gpu, x_gpu, mask_gpu, False)

    reference_cpu = reference_gpu = None
    reference_details: dict[str, Any] | None = None
    reference_gpu_details: dict[str, Any] | None = None
    slow_vs_cpu: dict[str, Any] | None = None
    slow_vs_gpu: dict[str, Any] | None = None
    if include_reference:
        encoder_cpu = _build_encoder("cpu")
        encoder_cpu.load_state_dict({key: value.detach().cpu() for key, value in encoder_gpu.state_dict().items()})
        mask_cpu = mask_gpu.detach().cpu()
        reference_cpu, reference_details = _manual_encoder(encoder_cpu, x_cpu, mask_cpu)
        reference_gpu, reference_gpu_details = _manual_encoder(encoder_gpu, x_gpu, mask_gpu)
        torch.cuda.synchronize()
        if slow is not None:
            slow_vs_cpu = _compare(slow.detach().cpu(), reference_cpu)
            slow_vs_gpu = _compare(slow, reference_gpu)

    return {
        "mask_kind": mask_kind,
        "input": {
            "shape": [int(v) for v in x_gpu.shape],
            "dtype": str(x_gpu.dtype),
            "cpu_input_sha256": _tensor_sha256(x_cpu),
            "model_state_sha256": _state_dict_sha256(encoder_gpu),
            "key_padding_mask": mask_gpu.detach().cpu().tolist(),
            "fully_masked_rows": [int(i) for i in torch.where(mask_gpu.all(dim=-1))[0].tolist()],
        },
        "fastpath": fast_details,
        "slowpath_intervention": slow_details,
        "independent_reference_cpu": (
            {
                "output": _describe(reference_cpu),
                "slow_vs_cpu_reference": slow_vs_cpu,
                "details": reference_details,
            }
            if reference_cpu is not None
            else None
        ),
        "independent_reference_gpu": (
            {
                "output": _describe(reference_gpu),
                "slow_vs_gpu_reference": slow_vs_gpu,
                "details": reference_gpu_details,
            }
            if reference_gpu is not None
            else None
        ),
        "fast_vs_slow_finite_only": (
            _compare(torch.nan_to_num(fast), slow)
            if fast is not None and slow is not None and bool(torch.isfinite(fast).all().item())
            else (
                {"passed": False, "reason": "fastpath output contains non-finite values"}
                if fast is not None and slow is not None
                else None
            )
        ),
    }


def _issues_from_data(errors: list[dict[str, Any]], failed_observation_checks: list[str] | None = None) -> list[str]:
    issues = [f"error:{item.get('stage', 'unknown')}" for item in errors]
    issues.extend(f"observation:{name}" for name in (failed_observation_checks or []))
    return issues


def _finalize(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    all_masked = next(row for row in rows if row["mask_kind"] == "all")
    partial = next(row for row in rows if row["mask_kind"] == "partial")
    none = next(row for row in rows if row["mask_kind"] == "none")
    fast_nan = all_masked["fastpath"]["output"]["nan_count"] > 0
    slow_finite = all_masked["slowpath_intervention"]["output"]["finite"]
    reference_finite = all_masked["independent_reference_cpu"]["output"]["finite"]
    all_masked_reference = all_masked["independent_reference_cpu"]["slow_vs_cpu_reference"]["passed"]
    partial_clean = (
        partial["fastpath"]["output"]["finite"]
        and partial["slowpath_intervention"]["output"]["finite"]
        and partial["independent_reference_cpu"]["slow_vs_cpu_reference"]["passed"]
    )
    none_clean = none["fastpath"]["output"]["finite"] and none["slowpath_intervention"]["output"]["finite"]
    passed = bool(
        fast_nan
        and slow_finite
        and reference_finite
        and all_masked_reference
        and partial_clean
        and none_clean
        and not errors
    )
    failed_observation_checks = [] if passed else [
        name
        for name, value in {
            "fastpath_all_masked_nan_count_positive": fast_nan,
            "slowpath_all_masked_finite": slow_finite,
            "cpu_reference_all_masked_finite": reference_finite,
            "slowpath_cpu_reference_comparison": all_masked_reference,
            "partial_mask_observations": partial_clean,
            "no_mask_observations": none_clean,
        }.items()
        if not value
    ]
    return {
        "status": "completed" if passed else "failed",
        "rows_observed": len(rows),
        "failed_observation_checks": failed_observation_checks,
        "issues": _issues_from_data(errors, failed_observation_checks),
    }


def _run(mode: str, output: Path, repeat_id: str, argv: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "case": CASE,
        "mode": mode,
        "repeat_id": repeat_id,
        "status": "running",
        "argv": argv,
        "source_path": str(Path(__file__).absolute()),
        "source_sha256": _sha256(Path(__file__)),
        "config": {
            "d_model": D_MODEL,
            "n_head": N_HEAD,
            "ff_dim": FF_DIM,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "n_layers": N_LAYERS,
            "seed": SEED,
            "rng_construction": "manual_seed(0); initialize CPU encoder; generate CPU torch.randn input; copy model and input to CUDA",
            "reference_tolerance": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
        },
        "environment": _environment(),
        "rows": [],
        "errors": [],
        "events_file": "events.jsonl",
    }
    event_path = output / "events.jsonl"
    _append_event(event_path, {"event": "start", "case": CASE, "mode": mode, "argv": argv})
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        # Match the upstream reproducer's RNG order: initialize the CPU
        # module, then generate CPU inputs, and only then copy both to CUDA.
        torch.manual_seed(SEED)
        encoder_cpu = _build_encoder("cpu")
        x_cpu = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        encoder_gpu = _build_encoder("cuda")
        encoder_gpu.load_state_dict({key: value.detach().cpu() for key, value in encoder_cpu.state_dict().items()})
        if mode == "baseline":
            mode_contract = "exact all-masked input; fastpath observation only"
            row_specs = [("all", True, False, False)]
        elif mode == "path-control":
            mode_contract = "exact all-masked input; fastpath and disabled-fastpath path control"
            row_specs = [("all", True, True, False)]
        elif mode == "reference-control":
            mode_contract = "all/partial/no-mask inputs; disabled-fastpath and independent CPU/GPU reference"
            row_specs = [
                ("all", False, True, True),
                ("partial", True, True, True),
                ("none", True, True, True),
            ]
        else:
            mode_contract = "all observations and negative controls for qualification"
            row_specs = [(kind, True, True, True) for kind in ("all", "partial", "none")]
        data["mode_contract"] = mode_contract
        for mask_kind, include_fast, include_slow, include_reference in row_specs:
            row = (
                _case(encoder_gpu, x_cpu, mask_kind)
                if include_fast and include_slow and include_reference
                else _case_scoped(
                    encoder_gpu,
                    x_cpu,
                    mask_kind,
                    include_fast=include_fast,
                    include_slow=include_slow,
                    include_reference=include_reference,
                )
            )
            data["rows"].append(row)
            _append_event(
                event_path,
                {
                    "event": "row",
                    "mask_kind": mask_kind,
                    "fast_nan_count": (
                        row["fastpath"]["output"]["nan_count"] if row["fastpath"] is not None else None
                    ),
                    "slow_finite": (
                        row["slowpath_intervention"]["output"]["finite"]
                        if row["slowpath_intervention"] is not None
                        else None
                    ),
                },
            )
        if mode == "qualification":
            data["finalization"] = _finalize(data["rows"], data["errors"])
            data["status"] = data["finalization"]["status"]
        else:
            issues = _issues_from_data(data["errors"])
            data["finalization"] = {
                "status": "completed" if not issues else "failed",
                "rows_observed": len(data["rows"]),
                "issues": issues,
            }
            data["status"] = data["finalization"]["status"]
    except Exception as exc:
        data["errors"].append({"stage": "top_level", **_error_record(exc)})
        data["finalization"] = {
            "status": "failed",
            "qualification": "EXECUTION_FAILED",
            "issues": _issues_from_data(data["errors"]),
        }
        data["status"] = "failed"
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    _append_event(event_path, {"event": "finish", "status": data["status"], "finalization": data["finalization"]})
    _atomic_json(output / "run.json", data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="E012 PyTorch fully-masked TransformerEncoder qualification")
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-id", default="qualification-r01")
    args = parser.parse_args()
    output: Path | None = None
    try:
        output = _safe_output(args.output)
        result = _run(args.mode, output, args.repeat_id, sys.argv)
        print(json.dumps({"status": result["status"], "mode": args.mode, "finalization": result["finalization"]}, indent=2))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        error = {"schema": SCHEMA, "case": CASE, "status": "failed", "failure_signal": "driver_exception", **_error_record(exc)}
        if output is not None:
            _atomic_json(output / "run.json", error)
        print(json.dumps(error, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

