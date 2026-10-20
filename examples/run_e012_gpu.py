"""Run the real E012 Stage 2A transport preflight on an approved POSIX GPU.

This entry point starts no Agent and makes no model/API call.  It requires the
existing PyTorch CUDA environment and sends the fixed transport-only worker
through the existing Stage 1E MeasurementServer and client.  The output root
must be new; raw, receipt and public publication bytes are retained there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.stage2a.contracts import preflight_case


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_python() -> str:
    """Keep the virtualenv launcher spelling when Python resolves its target."""
    if sys.prefix != sys.base_prefix:
        candidate = Path(sys.prefix) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if candidate.is_file():
            return str(candidate)
    return str(Path(sys.executable).absolute())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--valid-for-seconds", type=float, default=24 * 60 * 60)
    args = parser.parse_args()
    python = _current_python()
    source = ROOT / "cases" / "e012" / "workload_transport_v1.py"
    transport = ROOT / "evals" / "stage1e" / "execution_transport.py"
    client = ROOT / "evals" / "stage1e" / "measure.py"
    case = {
        "case_id": "E012",
        "schema": "stage1f-e012-workload-v1",
        "modes": ["baseline", "path-control", "reference-control"],
        "mode_row_counts": {"baseline": 1, "path-control": 1, "reference-control": 3},
    }
    worker = {
        "python": python,
        "source": str(source),
        "sha256": _sha(source),
        "modes": {
            "baseline": ["--mode", "baseline", "--repeat-id", "stage2a-e012-baseline"],
            "path-control": ["--mode", "path-control", "--repeat-id", "stage2a-e012-path-control"],
            "reference-control": ["--mode", "reference-control", "--repeat-id", "stage2a-e012-reference-control"],
        },
        "max_jobs": 3,
        "timeout_seconds": 120,
        "output_argument": "absolute",
    }
    proof = preflight_case(
        case,
        worker,
        {"path": str(transport), "module": "evals.stage1e.execution_transport", "protocol": "stage1e-fixed-source-v1"},
        {"path": str(client), "module": "evals.stage1e.measure", "protocol": "stage1e-fixed-source-v1"},
        {"python": python},
        args.output_root,
        valid_for_seconds=args.valid_for_seconds,
    )
    value = json.loads(proof.read_text(encoding="utf-8"))
    print(json.dumps({
        "proof": str(proof),
        "status": value.get("status"),
        "scope": value.get("scope"),
        "source_sha256": value.get("worker", {}).get("sha256"),
        "rows": [
            {
                "mode": row.get("mode"),
                "receipt": row.get("receipt_path"),
                "raw": row.get("raw_path"),
                "publication": row.get("publication_path"),
                "validation": row.get("validation"),
            }
            for row in value.get("rows", [])
        ],
    }, indent=2, ensure_ascii=False))
    return 0 if value.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
