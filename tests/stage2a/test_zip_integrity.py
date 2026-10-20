import hashlib
import json
import zipfile

import pytest

from experiments.verify_evidence_zip import VerificationFailure, verify_archive


def make_archive(path, mutation=None):
    payload = b"x"
    coverage = [{"original_path": "case://demo/observation", "archived_as": "observation.txt",
                 "sha256": hashlib.sha256(payload).hexdigest()}]
    files = {"observation.txt": payload, "frozen-coverage.json": json.dumps(coverage).encode()}
    manifest = {"frozen_paths": 1, "external_binary_fingerprints": 0,
                "files": [{"path": name, "size_bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()} for name,data in files.items()]}
    if mutation:
        mutation(manifest)
    with zipfile.ZipFile(path, "x") as archive:
        for name,data in files.items():
            archive.writestr(name,data)
        archive.writestr("DELIVERY-MANIFEST.json", json.dumps(manifest))
    return path


def test_integrity_replays_without_original_machine(tmp_path):
    result = verify_archive(make_archive(tmp_path/"valid.zip"))
    assert result["status"] == "PASS_PACKAGE_CONSISTENCY"
    assert result["checks"]["original_paths_reopened"] is False


@pytest.mark.parametrize("mutation", [
    lambda value: value["files"][0].update(size_bytes=True),
    lambda value: value.update(frozen_paths=True),
    lambda value: value.update(external_binary_fingerprints=False),
    lambda value: value.update(untrusted_metadata=float("nan")),
])
def test_malformed_manifest_is_not_an_integrity_pass(tmp_path, mutation):
    with pytest.raises(VerificationFailure):
        verify_archive(make_archive(tmp_path/"bad.zip",mutation))
