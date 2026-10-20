"""Offline consistency verifier for a selected evidence ZIP.

This small stdlib-only reader retains the historical payload/hash checks.
It reads the ZIP central directory,
``DELIVERY-MANIFEST.json`` and ``frozen-coverage.json``.  It never extracts the
archive and never opens ``original_path`` values from frozen coverage: those
values are compared as mapping strings only.

The PASS result means package consistency only.  It does not establish source
execution, scientific reproduction, Agent behavior, or delivery synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile
from typing import Any, Iterable


DELIVERY_MANIFEST = "DELIVERY-MANIFEST.json"
FROZEN_COVERAGE = "frozen-coverage.json"
CHUNK_SIZE = 1024 * 1024


class VerificationFailure(ValueError):
    """A package consistency check failed."""


def sha_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest used by Stage 1E packaging."""

    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationFailure(message)


def _read_json(bundle: zipfile.ZipFile, name: str) -> Any:
    try:
        raw = bundle.read(name)
    except KeyError as exc:
        raise VerificationFailure(f"missing required member: {name}") from exc
    try:
        def reject_constant(value: str) -> None:
            raise VerificationFailure(f"non-finite JSON literal in {name}: {value}")
        return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationFailure(f"invalid JSON member: {name}") from exc


def _validate_member_name(name: str) -> None:
    _require(bool(name), "empty ZIP member name")
    _require("\\" not in name, f"non-POSIX ZIP member name: {name!r}")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), f"absolute ZIP member name: {name!r}")
    _require(".." not in path.parts, f"parent traversal ZIP member name: {name!r}")


def _validate_manifest_shape(manifest: Any) -> list[dict[str, Any]]:
    _require(isinstance(manifest, dict), "DELIVERY-MANIFEST.json must be an object")
    files = manifest.get("files")
    _require(isinstance(files, list), "manifest files must be a list")
    entries: list[dict[str, Any]] = []
    names: list[str] = []
    for index, entry in enumerate(files):
        _require(isinstance(entry, dict), f"manifest files[{index}] must be an object")
        name = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        _require(isinstance(name, str), f"manifest files[{index}].path must be a string")
        _validate_member_name(name)
        _require(name != DELIVERY_MANIFEST, "DELIVERY-MANIFEST.json cannot be its own payload")
        _require(type(size) is int and size >= 0, f"invalid size for manifest member: {name}")
        _require(isinstance(digest, str) and len(digest) == 64, f"invalid SHA-256 for manifest member: {name}")
        _require(digest == digest.lower() and all(c in "0123456789abcdef" for c in digest), f"non-hex SHA-256 for: {name}")
        names.append(name)
        entries.append({"path": name, "size_bytes": size, "sha256": digest})
    _require(len(names) == len(set(names)), "duplicate payload path in DELIVERY-MANIFEST.json")
    _require(FROZEN_COVERAGE in set(names), "frozen-coverage.json is not listed as a payload")
    return entries


def _validate_coverage_shape(coverage: Any, payload_by_name: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> dict[str, int]:
    _require(isinstance(coverage, list), "frozen-coverage.json must be a list")
    _require(type(manifest.get("frozen_paths")) is int and manifest["frozen_paths"] >= 0,
             "manifest frozen_paths must be a non-negative integer")
    _require(type(manifest.get("external_binary_fingerprints")) is int and manifest["external_binary_fingerprints"] >= 0,
             "manifest external_binary_fingerprints must be a non-negative integer")
    original_paths: list[str] = []
    archived_paths: list[str] = []
    mapped = 0
    external = 0
    for index, entry in enumerate(coverage):
        _require(isinstance(entry, dict), f"coverage[{index}] must be an object")
        original = entry.get("original_path")
        digest = entry.get("sha256")
        archived = entry.get("archived_as")
        _require(isinstance(original, str) and bool(original), f"coverage[{index}].original_path must be a string")
        # Deliberately do not turn original into Path or access it.  A frozen
        # absolute path is provenance metadata, not a verifier input.
        _require(isinstance(digest, str) and len(digest) == 64, f"invalid coverage SHA-256 at index {index}")
        _require(digest == digest.lower() and all(c in "0123456789abcdef" for c in digest), f"non-hex coverage SHA-256 at index {index}")
        _require(original not in original_paths, f"duplicate original_path in coverage: {original!r}")
        original_paths.append(original)
        if archived is None:
            external += 1
            _require(entry.get("scope"), f"external fingerprint lacks scope at index {index}")
            continue
        _require(isinstance(archived, str) and bool(archived), f"coverage[{index}].archived_as must be a string or null")
        _validate_member_name(archived)
        archived_paths.append(archived)
        _require(archived in payload_by_name, f"coverage archived_as is not a payload: {archived!r}")
        _require(payload_by_name[archived]["sha256"] == digest, f"coverage/payload SHA mismatch: {archived!r}")
        mapped += 1
    _require(manifest.get("frozen_paths") == len(coverage), "manifest frozen_paths disagrees with coverage length")
    _require(manifest.get("external_binary_fingerprints") == external, "manifest external fingerprint count disagrees with coverage")
    return {
        "entries": len(coverage),
        "mapped": mapped,
        "external_binary_fingerprints": external,
        "unique_archived_as": len(set(archived_paths)),
        "archived_as_alias_entries": len(archived_paths) - len(set(archived_paths)),
    }


def _check_payloads(bundle: zipfile.ZipFile, entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    checked = 0
    total_bytes = 0
    for entry in entries:
        name = entry["path"]
        try:
            info = bundle.getinfo(name)
        except KeyError as exc:
            raise VerificationFailure(f"manifest payload missing from ZIP: {name}") from exc
        digest = hashlib.sha256()
        size = 0
        try:
            with bundle.open(info, "r") as stream:
                for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
                    size += len(block)
                    digest.update(block)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise VerificationFailure(f"payload read/CRC failure: {name}") from exc
        actual_digest = digest.hexdigest()
        _require(size == entry["size_bytes"], f"payload byte count mismatch: {name}")
        _require(actual_digest == entry["sha256"], f"payload SHA-256 mismatch: {name}")
        checked += 1
        total_bytes += size
    return {"payloads": checked, "payload_bytes": total_bytes}


def verify_archive(archive: Path) -> dict[str, Any]:
    """Verify one archive without extracting or reopening frozen original paths."""

    _require(archive.is_file(), f"archive does not exist: {archive}")
    try:
        archive_size = archive.stat().st_size
        archive_digest = sha_file(archive)
        with zipfile.ZipFile(archive, "r") as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            for name in names:
                _validate_member_name(name)
            _require(len(names) == len(set(names)), "duplicate ZIP member name")
            _require(DELIVERY_MANIFEST in names, "missing DELIVERY-MANIFEST.json")
            _require(len(names) == len(set(names)), "ZIP member names are not unique")

            manifest = _read_json(bundle, DELIVERY_MANIFEST)
            entries = _validate_manifest_shape(manifest)
            payload_names = {entry["path"] for entry in entries}
            expected_names = payload_names | {DELIVERY_MANIFEST}
            _require(set(names) == expected_names, "ZIP members differ from manifest payload set")
            payload_by_name = {entry["path"]: entry for entry in entries}

            coverage = _read_json(bundle, FROZEN_COVERAGE)
            coverage_counts = _validate_coverage_shape(coverage, payload_by_name, manifest)

            # testzip checks central-directory CRCs.  Payload reads above also
            # exercise CRC checks for every manifest-listed member.
            bad_crc = bundle.testzip()
            _require(bad_crc is None, f"CRC failure in ZIP member: {bad_crc}")
            payload_counts = _check_payloads(bundle, entries)

        return {
            "status": "PASS_PACKAGE_CONSISTENCY",
            "scope": "package consistency only; does not establish scientific reproduction, Agent behavior, or delivery synchronization",
            "archive": str(archive),
            "archive_bytes": archive_size,
            "archive_sha256": archive_digest,
            "zip_members": len(names),
            "manifest_payloads": payload_counts["payloads"],
            "payload_bytes": payload_counts["payload_bytes"],
            "coverage": coverage_counts,
            "checks": {
                "unique_zip_members": "PASS",
                "manifest_member_set": "PASS",
                "payload_sha256_and_bytes": "PASS",
                "crc": "PASS",
                "coverage_original_to_archived_mapping": "PASS",
                "external_fingerprint_count": "PASS",
                "original_paths_reopened": False,
                "extracted": False,
            },
            "scientific_reproduction": "NOT_EVALUATED",
        }
    except (OSError, zipfile.BadZipFile, VerificationFailure) as exc:
        raise VerificationFailure(str(exc)) from exc


def _write_optional_output(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    if path.exists():
        raise VerificationFailure(f"refuse to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a selected evidence ZIP without extraction or source reopening")
    parser.add_argument("archive", type=Path, help="evidence ZIP containing DELIVERY-MANIFEST.json")
    parser.add_argument("--output", type=Path, help="optional new JSON receipt path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        result = verify_archive(args.archive)
        _write_optional_output(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except VerificationFailure as exc:
        result = {
            "status": "FAIL_PACKAGE_CONSISTENCY",
            "scope": "package consistency only; no scientific conclusion",
            "archive": str(args.archive),
            "error": str(exc),
        }
        try:
            _write_optional_output(args.output, result)
        except VerificationFailure as output_exc:
            result["output_error"] = str(output_exc)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

