from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from evals.stage1e.execution_transport import measurement_view
from evals.stage2a import contracts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict, Path]:
    root = tmp_path / "proof-root"
    root.mkdir(parents=True)
    packet = root / "packet"
    (packet / "requests").mkdir(parents=True)
    (packet / "observations" / ".staging").mkdir(parents=True)
    evidence = root / "private-evidence" / "evaluator-tools"
    source = tmp_path / "approved-worker.py"
    source.write_text("# fixed worker fixture\n", encoding="utf-8")
    source_hash = _sha(source)
    request_id = "0123456789abcdef0123456789abcdef"
    output = "tcase-01-baseline"
    job = evidence / request_id
    (job / "work-output").mkdir(parents=True)
    (packet / "observations" / output).mkdir(parents=True)
    raw = {
        "schema": "stage2a-test-worker-v1",
        "case": "TCASE",
        "mode": "baseline",
        "status": "completed",
        "source_path": str(source),
        "source_sha256": source_hash,
        "rows": [{"observation": "finite"}],
        "errors": [],
        "finalization": {"status": "completed", "rows_observed": 1, "issues": []},
    }
    public = measurement_view(raw)
    raw_path = job / "work-output" / "run.json"
    public_path = packet / "observations" / output / "run.json"
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    public_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    events = packet / "observations" / output / "events.jsonl"
    events.write_text(json.dumps({"event": "finish", "status": "completed"}) + "\n", encoding="utf-8")
    receipt = {
        "protocol": contracts.TRANSPORT_PROTOCOL,
        "request_id": request_id,
        "mode": "baseline",
        "status": "completed",
        "error": None,
        "driver_status": "completed",
        "worker_launched": True,
        "returncode": 0,
        "source_sha256": source_hash,
        "raw_result": f"observations/{output}/run.json",
        "raw_directory": f"observations/{output}",
        "evidence_seal": "complete",
        "evaluation_scope": "DIAGNOSIS_ONLY",
        "patch_validation": "NOT_EVALUATED",
    }
    receipt_path = packet / "observations" / f"receipt-{request_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    process = {
        "request_id": request_id,
        "source_sha256_expected": source_hash,
        "source_sha256_after": source_hash,
        "status": "completed",
        "returncode": 0,
        "worker_launched": True,
    }
    process_path = job / "process.json"
    process_path.write_text(json.dumps(process, indent=2) + "\n", encoding="utf-8")
    worker = {
        "python": str(contracts._current_launcher_path()),
        "source": str(source),
        "sha256": source_hash,
        "modes": {"baseline": []},
        "max_jobs": 1,
        "timeout_seconds": 10,
        "output_argument": "absolute",
    }
    worker_value, _ = contracts._normalise_worker(worker)
    transport_value = contracts._normalise_tool(
        {"module": "evals.stage1e.execution_transport", "protocol": contracts.TRANSPORT_PROTOCOL},
        "transport",
        "evals.stage1e.execution_transport",
    )
    client_value = contracts._normalise_tool(
        {"module": "evals.stage1e.measure", "protocol": contracts.TRANSPORT_PROTOCOL},
        "client",
        "evals.stage1e.measure",
    )
    runtime_value = contracts._collect_runtime_observations(contracts._runtime_record({}, worker_value["python"]))
    row = {
        "mode": "baseline",
        "output": output,
        "request_id": request_id,
        "receipt_path": "packet/observations/receipt-0123456789abcdef0123456789abcdef.json",
        "receipt_sha256": _sha(receipt_path),
        "receipt": receipt,
        "process_path": f"private-evidence/evaluator-tools/{request_id}/process.json",
        "process_sha256": _sha(process_path),
        "publication_path": f"packet/observations/{output}/run.json",
        "raw_path": f"private-evidence/evaluator-tools/{request_id}/work-output/run.json",
        "private_raw_path": f"private-evidence/evaluator-tools/{request_id}/work-output/run.json",
        "publication_sha256": _sha(public_path),
        "raw_sha256": _sha(raw_path),
        "private_raw_sha256": _sha(raw_path),
        "events_path": f"packet/observations/{output}/events.jsonl",
        "events_sha256": _sha(events),
        "view_check": {"passed": True},
        "validation": {"passed": True, "errors": []},
    }
    proof = {
        "proof_version": contracts.PROOF_VERSION,
        "contract_version": contracts.CONTRACT_VERSION,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "case": {"case_id": "TCASE", "schema": "stage2a-test-worker-v1", "modes": ["baseline"]},
        "worker": worker_value,
        "transport": transport_value,
        "client": client_value,
        "runtime": runtime_value,
        "packet_root": "packet",
        "private_evidence_root": "private-evidence",
        "modes": ["baseline"],
        "rows": [row],
        "binding_errors": [],
        "scope": "CPU synthetic contract probe",
    }
    proof_path = root / "preflight-proof.json"
    contracts._write_synthetic_proof(proof_path, proof)
    return proof_path, worker, root


def _rewrite(proof_path: Path, mutate) -> tuple[Path, dict, Path]:
    proof, worker, root = _fixture(proof_path.parent.parent)
    value = json.loads(proof.read_text(encoding="utf-8"))
    mutate(value, root)
    contracts._write_synthetic_proof(proof, value)
    return proof, worker, root


def test_require_preflight_rereads_complete_receipt_process_raw_and_view(tmp_path: Path) -> None:
    proof, worker, _ = _fixture(tmp_path)
    accepted = contracts.require_preflight(proof, worker)
    assert accepted["status"] == "PASS"
    assert accepted["rows"][0]["view_check"]["passed"] is True


def test_proof_snapshot_or_receipt_mismatch_is_rejected(tmp_path: Path) -> None:
    proof, worker, _ = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    value["rows"][0]["receipt"]["returncode"] = 1
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError):
        contracts.require_preflight(proof, worker)


@pytest.mark.parametrize("field", ["schema", "rows", "errors", "finalization"])
def test_missing_required_worker_fields_never_defaults_to_pass(tmp_path: Path, field: str) -> None:
    proof, worker, root = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    raw_path = root / value["rows"][0]["raw_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw.pop(field)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    # The sidecar is updated so the test exercises evidence validation rather
    # than only the proof-byte seal.
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError):
        contracts.require_preflight(proof, worker)


def test_nonempty_issues_and_failed_observations_are_rejected(tmp_path: Path) -> None:
    proof, worker, root = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    raw_path = root / value["rows"][0]["raw_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["finalization"]["issues"] = ["observation:row"]
    raw["finalization"]["failed_observation_checks"] = ["row"]
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError):
        contracts.require_preflight(proof, worker)


def test_boolean_rows_observed_cannot_masquerade_as_integer(tmp_path: Path) -> None:
    proof, worker, root = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    row = value["rows"][0]
    raw_path = root / row["raw_path"]
    publication_path = root / row["publication_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["finalization"]["rows_observed"] = True
    public = measurement_view(raw)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    publication_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    row["raw_sha256"] = _sha(raw_path)
    row["private_raw_sha256"] = _sha(raw_path)
    row["publication_sha256"] = _sha(publication_path)
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError, match="rows_observed"):
        contracts.require_preflight(proof, worker)


def test_nonfinite_observation_values_are_not_instrument_failure(tmp_path: Path) -> None:
    proof, worker, root = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    row = value["rows"][0]
    raw_path = root / row["raw_path"]
    publication_path = root / row["publication_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["rows"][0].update({"finite": False, "values_flat": ["NaN"]})
    public = measurement_view(raw)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    publication_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    row["raw_sha256"] = _sha(raw_path)
    row["private_raw_sha256"] = _sha(raw_path)
    row["publication_sha256"] = _sha(publication_path)
    contracts._write_synthetic_proof(proof, value)
    assert contracts.require_preflight(proof, worker)["status"] == "PASS"


def test_illegal_evidence_path_and_expiry_are_rejected(tmp_path: Path) -> None:
    proof, worker, _ = _fixture(tmp_path)
    value = json.loads(proof.read_text(encoding="utf-8"))
    value["rows"][0]["publication_path"] = "../outside/run.json"
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError):
        contracts.require_preflight(proof, worker)

    expiry_root = tmp_path / "expiry"
    expiry_root.mkdir()
    proof, worker, _ = _fixture(expiry_root)
    value = json.loads(proof.read_text(encoding="utf-8"))
    value["expires_at"] = "2020-01-01T00:00:00Z"
    contracts._write_synthetic_proof(proof, value)
    with pytest.raises(contracts.ContractError):
        contracts.require_preflight(proof, worker)


def test_freeze_case_gates_delegate_before_any_call(tmp_path: Path) -> None:
    proof, worker, _ = _fixture(tmp_path)
    called: list[bool] = []

    def prepare() -> str:
        called.append(True)
        return "prepared"

    with pytest.raises(contracts.ContractError):
        contracts.freeze_case(tmp_path / "missing-proof.json", worker, tmp_path / "blocked", prepare=prepare)
    assert called == []
    manifest = contracts.freeze_case(proof, worker, tmp_path / "frozen", prepare=prepare)
    assert manifest.is_file()
    assert called == [True]


def _cpu_worker(tmp_path: Path, kind: str) -> tuple[dict, dict]:
    source = tmp_path / f"cpu-{kind}.py"
    source.write_text(
        f'''from pathlib import Path
import argparse
import hashlib
import json
import sys

KIND = {kind!r}
parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--repeat-id", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
if KIND == "nonzero":
    raise SystemExit(7)
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
source_path = Path(__file__).absolute()
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
if KIND == "malformed":
    run = {{
        "case": "TCPU",
        "mode": args.mode,
        "status": "completed",
        "source_sha256": source_hash,
        "finalization": {{"status": "completed", "rows_observed": 1, "issues": []}},
    }}
elif KIND == "issues":
    run = {{
        "schema": "stage2a-test-worker-v1",
        "case": "TCPU",
        "mode": args.mode,
        "status": "failed",
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "rows": [{{"observation": "failed"}}],
        "errors": [{{"stage": "row"}}],
        "finalization": {{"status": "failed", "rows_observed": 1, "issues": ["error:row"]}},
    }}
elif KIND == "row-failed":
    run = {{
        "schema": "stage2a-test-worker-v1",
        "case": "TCPU",
        "mode": args.mode,
        "status": "completed",
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "rows": [{{"status": "failed", "error": "synthetic row error"}}],
        "errors": [],
        "finalization": {{"status": "completed", "rows_observed": 1, "issues": []}},
    }}
else:
    run = {{
        "schema": "stage2a-test-worker-v1",
        "case": "TCPU",
        "mode": args.mode,
        "status": "completed",
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "rows": [{{"observation": "finite"}}],
        "errors": [],
        "finalization": {{"status": "completed", "rows_observed": 1, "issues": []}},
    }}
(output / "run.json").write_text(json.dumps(run, indent=2) + "\\n", encoding="utf-8")
(output / "events.jsonl").write_text(json.dumps({{"event": "finish", "status": run["status"], "finalization": run["finalization"]}}) + "\\n", encoding="utf-8")
''',
        encoding="utf-8",
    )
    worker = {
        "python": str(contracts._current_launcher_path()),
        "source": str(source),
        "sha256": _sha(source),
        "modes": {"probe": ["--mode", "probe", "--repeat-id", f"cpu-{kind}"]},
        "max_jobs": 1,
        "timeout_seconds": 10,
        "output_argument": "absolute",
    }
    case = {"case_id": "TCPU", "schema": "stage2a-test-worker-v1", "modes": ["probe"], "mode_row_counts": {"probe": 1}}
    return case, worker


@pytest.mark.posix
@pytest.mark.parametrize("kind", ["normal", "nonzero", "malformed", "issues", "row-failed"])
def test_preflight_case_runs_real_cpu_worker_and_fails_closed(tmp_path: Path, kind: str) -> None:
    if os.name != "posix":
        pytest.skip("MeasurementServer requires the verified POSIX transport")
    case, worker = _cpu_worker(tmp_path, kind)
    proof = contracts.preflight_case(case, worker, output_root=tmp_path / "preflight")
    value = json.loads(proof.read_text(encoding="utf-8"))
    if kind == "normal":
        assert value["status"] == "PASS"
        assert contracts.require_preflight(proof, worker)["status"] == "PASS"
        assert value["rows"][0]["receipt"]["status"] == "completed"
    else:
        assert value["status"] == "FAIL"
        assert value["rows"][0]["validation"]["passed"] is False
        with pytest.raises(contracts.ContractError):
            contracts.require_preflight(proof, worker)


@pytest.mark.posix
def test_preflight_rejects_tool_path_that_is_not_the_imported_transport(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("MeasurementServer requires the verified POSIX transport")
    case, worker = _cpu_worker(tmp_path, "normal")
    actual = Path(contracts._resolve_module_path("evals.stage1e.execution_transport"))
    wrong = tmp_path / "renamed-transport.py"
    shutil.copyfile(actual, wrong)
    with pytest.raises(contracts.ContractError, match="transport declaration"):
        contracts.preflight_case(
            case,
            worker,
            {"path": str(wrong), "module": "evals.stage1e.execution_transport", "protocol": contracts.TRANSPORT_PROTOCOL},
            output_root=tmp_path / "wrong-tool",
        )
