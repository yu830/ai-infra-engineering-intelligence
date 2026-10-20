"""Packet-local control client; raw observations remain complete and read-only."""
from __future__ import annotations
import os
import sys
if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", __file__, *sys.argv[1:]])

import argparse
import json
from pathlib import Path
import re
import time
import uuid

PROTOCOL = "stage1e-fixed-source-v1"
OUTPUT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def create_request(work: Path, mode: str, output: str):
    config = json.loads((work / "measurement.json").read_text())
    if config.get("protocol") != PROTOCOL or mode not in config["modes"]:
        raise ValueError("unknown fixed mode/protocol")
    if not OUTPUT.fullmatch(output) or output in (".", ".."):
        raise ValueError("output must be a safe basename")
    request_id = uuid.uuid4().hex
    value = {"protocol": PROTOCOL, "request_id": request_id, "mode": mode, "output": output}
    path = work / "requests" / f"request-{request_id}.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream)
    temporary.rename(path)
    return request_id, work / "observations" / f"receipt-{request_id}.json", config


def validate_receipt(receipt: object, request_id: str, mode: str, output: str, config: dict) -> dict:
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    if receipt.get("request_id") != request_id or receipt.get("protocol") != PROTOCOL:
        raise ValueError("receipt id/protocol mismatch")
    if receipt.get("status") not in {"completed", "failed", "rejected", "not_run_resource"}:
        raise ValueError("receipt lacks a terminal status")
    if receipt["status"] == "completed":
        required = {"mode", "error", "driver_status", "worker_launched", "returncode", "source_sha256",
                    "raw_result", "raw_directory", "evidence_seal", "evaluation_scope", "patch_validation"}
        if not required.issubset(receipt):
            raise ValueError("completed receipt is incomplete")
        if (receipt["mode"] != mode or receipt["error"] is not None
                or receipt["driver_status"] != "completed" or receipt["worker_launched"] is not True
                or type(receipt["returncode"]) is not int or receipt["returncode"] != 0
                or receipt["source_sha256"] != config["source_sha256"]
                or receipt["raw_result"] != f"observations/{output}/run.json"
                or receipt["raw_directory"] != f"observations/{output}"
                or receipt["evidence_seal"] != "complete"
                or receipt["evaluation_scope"] != "DIAGNOSIS_ONLY"
                or receipt["patch_validation"] != "NOT_EVALUATED"):
            raise ValueError("completed receipt contradicts the fixed-source contract")
    elif not isinstance(receipt.get("error"), str) or not receipt["error"]:
        raise ValueError("noncompleted receipt requires its error")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_id = None
    path = None
    try:
        request_id, path, config = create_request(Path(__file__).resolve().parent, args.mode, args.output)
        wait = config["wait_seconds"]
        if type(wait) not in (int, float) or not 0 < wait <= 600:
            raise ValueError("invalid receipt wait budget")
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if path.is_file():
                receipt = validate_receipt(json.loads(path.read_text()), request_id, args.mode, args.output, config)
                print(json.dumps(receipt, indent=2))
                return 0 if receipt.get("status") == "completed" else 2
            time.sleep(0.05)
        raise TimeoutError("measurement receipt unavailable within wait budget")
    except Exception as exc:
        print(json.dumps({"status": "client_failed", "error": repr(exc),
                          "request_id": request_id, "receipt_path": str(path) if path else None,
                          "mode": args.mode, "output": args.output}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
