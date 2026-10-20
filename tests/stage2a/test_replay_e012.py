from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from experiments.replay_e012 import ReplayFailure, audit_fixture, audit_raw_records, strict_json_bytes


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples/e012"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _refresh_manifest(root: Path) -> None:
    coverage_path = root / "frozen-coverage.json"
    coverage = _read(coverage_path)
    for item in coverage:
        path = root / Path(*item["archived_as"].split("/"))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(coverage_path, coverage)
    manifest_path = root / "DELIVERY-MANIFEST.json"
    manifest = _read(manifest_path)
    for entry in manifest["files"]:
        path = root / Path(*entry["path"].split("/"))
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(manifest_path, manifest)


class ReplayE012Tests(unittest.TestCase):
    def test_directory_and_zip_are_independently_audited(self):
        directory_result = audit_fixture(FIXTURE)
        self.assertEqual(directory_result["status"], "PASS_OFFLINE_SCIENTIFIC_AUDIT")
        self.assertEqual(directory_result["score"]["score"], 5)
        self.assertFalse(directory_result["checks"]["original_paths_reopened"])
        self.assertFalse(directory_result["checks"]["gpu_reexecution"])

        with tempfile.TemporaryDirectory(prefix="stage2a-zip-") as temp:
            fixture_zip = Path(temp) / "e012-public-fixture.zip"
            with zipfile.ZipFile(fixture_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(FIXTURE.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(FIXTURE).as_posix())
            zip_result = audit_fixture(fixture_zip)
        self.assertEqual(zip_result["status"], "PASS_OFFLINE_SCIENTIFIC_AUDIT")
        self.assertEqual(zip_result["source_kind"], "zip")
        self.assertEqual(zip_result["score"]["max_score"], 5)

    def test_raw_bridge_projects_records_without_returning_raw_arrays(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-raw-") as temp:
            raw_root = Path(temp)
            for role in ("baseline", "path-control", "reference-control"):
                observation = _read(FIXTURE / f"e012/{role}.json")
                shape = observation["config"]["shape"]
                raw = {
                    "schema": "stage1f-e012-workload-v1",
                    "case": "E012",
                    "mode": role,
                    "status": "completed",
                    "errors": [],
                    "finalization": {"status": "completed", "rows_observed": len(observation["rows"]), "issues": []},
                    "environment": {"cuda_available": True},
                    "source_sha256": observation["source_sha256"],
                    "config": {
                        "d_model": shape[2], "n_head": 4, "ff_dim": 64, "batch": shape[0],
                        "seq_len": shape[1], "n_layers": 2, "seed": observation["config"]["seed"],
                        "reference_tolerance": observation["config"]["reference_tolerance"],
                    },
                    "rows": observation["rows"],
                }
                _write(raw_root / f"{role}.json", raw)
            result = audit_raw_records(
                raw_root / "baseline.json",
                raw_root / "path-control.json",
                raw_root / "reference-control.json",
                reference_fixture=FIXTURE,
            )
        self.assertEqual(result["status"], "PASS_OFFLINE_RAW_AUDIT")
        self.assertEqual(result["score"]["score"], 5)
        self.assertTrue(result["reference_expected_score"]["criteria_match"])
        self.assertFalse(result["raw_observations_returned"])

    def test_nonstandard_json_number_is_rejected(self):
        with self.assertRaises(ReplayFailure):
            strict_json_bytes(b'{"value": NaN}', "negative.json")
        with self.assertRaises(ReplayFailure):
            strict_json_bytes(b'{"value": 1e999}', "negative.json")

    def test_corrupt_declared_counter_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            baseline_path = root / "e012/baseline.json"
            baseline = _read(baseline_path)
            baseline["rows"][0]["fastpath"]["output"]["nan_count"] += 1
            _write(baseline_path, baseline)
            _refresh_manifest(root)
            with self.assertRaises(ReplayFailure):
                audit_fixture(root)

    def test_wrong_coverage_mapping_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            coverage_path = root / "frozen-coverage.json"
            coverage = _read(coverage_path)
            coverage[1]["archived_as"] = coverage[0]["archived_as"]
            _write(coverage_path, coverage)
            _refresh_manifest(root)
            with self.assertRaises(ReplayFailure):
                audit_fixture(root)

    def test_duplicate_mask_row_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            reference_path = root / "e012/reference-control.json"
            reference = _read(reference_path)
            reference["rows"].append(reference["rows"][0])
            _write(reference_path, reference)
            _refresh_manifest(root)
            with self.assertRaises(ReplayFailure):
                audit_fixture(root)

    def test_cross_role_input_identity_and_tolerance_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            path_control_path = root / "e012/path-control.json"
            path_control = _read(path_control_path)
            path_control["config"]["reference_tolerance"]["atol"] = 1e-4
            path_control["rows"][0]["input"]["model_state_sha256"] = "0" * 64
            _write(path_control_path, path_control)
            _refresh_manifest(root)
            with self.assertRaises(ReplayFailure):
                audit_fixture(root)

    def test_expected_score_is_checked_against_computed_score(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            expected_path = root / "expected-score.json"
            expected = _read(expected_path)
            expected["expected"]["score"] = 4
            _write(expected_path, expected)
            manifest = _read(root / "DELIVERY-MANIFEST.json")
            manifest["expected_score"]["score"] = 4
            _write(root / "DELIVERY-MANIFEST.json", manifest)
            _refresh_manifest(root)
            with self.assertRaises(ReplayFailure):
                audit_fixture(root)

    def test_valid_finite_fastpath_is_reported_as_not_reproduced(self):
        with tempfile.TemporaryDirectory(prefix="stage2a-test-") as temp:
            root = Path(temp) / "fixture"
            shutil.copytree(FIXTURE, root)
            baseline_path = root / "e012/baseline.json"
            baseline = _read(baseline_path)
            descriptor = baseline["rows"][0]["fastpath"]["output"]
            descriptor["values_flat"] = [0.0 if isinstance(value, str) else value for value in descriptor["values_flat"]]
            values = descriptor["values_flat"]
            row_size = len(values) // descriptor["shape"][0]
            descriptor["nan_count"] = 0
            descriptor["inf_count"] = 0
            descriptor["finite"] = True
            descriptor["row1_finite"] = True
            descriptor["row1_absmax_finite_values"] = max(abs(float(value)) for value in values[row_size:])
            _write(baseline_path, baseline)

            expected_path = root / "expected-score.json"
            expected = _read(expected_path)
            expected["criteria"]["fully_masked_fastpath_nonfinite_region"]["passed"] = False
            expected["expected"]["score"] = 4
            expected["expected"]["status"] = "NOT_REPRODUCED_DEVELOPMENT_EXAMPLE"
            _write(expected_path, expected)
            manifest = _read(root / "DELIVERY-MANIFEST.json")
            manifest["expected_score"]["score"] = 4
            _write(root / "DELIVERY-MANIFEST.json", manifest)
            _refresh_manifest(root)

            result = audit_fixture(root)
            self.assertEqual(result["status"], "NOT_REPRODUCED")
            self.assertEqual(result["score"]["score"], 4)
            self.assertFalse(result["score"]["criteria"]["fully_masked_fastpath_nonfinite_region"]["passed"])

    def test_public_fixture_has_no_private_payload_fields(self):
        for path in FIXTURE.rglob("*.json"):
            value = _read(path)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/mnt/", text)
            self.assertNotIn("/home/", text)
            self.assertNotIn("/tmp/", text)
            self.assertNotIn('"argv"', text)
            self.assertNotIn('"source_path"', text)
            self.assertNotIn('"environment"', text)
            self.assertIsNotNone(value)


if __name__ == "__main__":
    unittest.main()
