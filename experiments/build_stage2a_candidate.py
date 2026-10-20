"""Build a selected, unpublished candidate; never copy a complete repository."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

from experiments.verify_evidence_zip import verify_archive

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "pyproject.toml", "uv.lock", "MANIFEST.in", "README.md", "LICENSE",
    ".github/workflows/offline-candidate.yml",
    "docs/stage2a-support.md", "docs/stage2a-rights.md", "docs/stage2b-rights.md", "docs/stage2b-public-files.json",
    "experiments/verify_evidence_zip.py", "experiments/replay_e012.py",
    "experiments/build_stage2a_candidate.py",
    "evals/stage1e/execution_transport.py", "evals/stage1e/measure.py",
    "evals/stage1d/execution_transport.py", "evals/stage1d/run_one.py",
    "cases/e012/workload_transport_v1.py",
    "examples/run_e012_gpu.py",
)
TREES = ("evals/stage2a", "tests/stage2a", "tests/stage2b", "examples/e012")
SUFFIXES = {".py", ".json", ".md", ".txt", ".zip"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def _clean_commit() -> str:
    try:
        checkout = Path(_git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        if checkout != ROOT.resolve():
            raise ValueError("release gate requires its own Git checkout root")
        commit = _git("rev-parse", "HEAD").decode().strip()
        if _git("status", "--porcelain", "--untracked-files=all").strip():
            raise ValueError("release gate requires a clean Git commit")
        return commit
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("release gate requires an available Git checkout") from exc


def _release_paths() -> list[Path]:
    config = json.loads((ROOT / "docs/stage2b-public-files.json").read_text(encoding="utf-8"))
    names = config["files"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("release allowlist must be a nonempty list of paths")
    if len(names) != len(set(names)):
        raise ValueError("duplicate release allowlist path")
    for name in names:
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name or ":" in name):
            raise ValueError("unsafe release allowlist path")
    return [ROOT / name for name in names]


def build(output: Path, *, release_gate: bool = False) -> dict:
    if output.exists():
        raise ValueError("candidate output must be a new directory")
    source_commit = _clean_commit() if release_gate else None
    paths = _release_paths() if release_gate else [ROOT / name for name in FILES]
    for tree in (() if release_gate else TREES):
        directory = ROOT / tree
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"required candidate input directory missing: {tree}")
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and p.suffix in SUFFIXES and "__pycache__" not in p.parts)
    payloads = {}
    # Actual identities are computed at runtime, not embedded in public source.
    private_roots = (str(ROOT), str(Path.home()))
    patterns = (
        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        rb"\b(?:sk-proj-|ghp_|github_pat_)[A-Za-z0-9_\-]{20,}",
        rb"Authorization[\"\s:]+Bearer\s+[A-Za-z0-9._\-]{20,}",
        rb"https://(?:app\.)?notion\.(?:com|so)/p/",
        rb"https://drive\.google\.com/(?:file/d|drive/folders)/",
        rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        rb"\bAIza[A-Za-z0-9_-]{35}\b",
        rb"\bxox[baprs]-[A-Za-z0-9-]{20,}",
        rb"\b[A-Za-z]:[\\/](?:Users|80_Dev)[\\/]",
        rb"/(?:home|Users)/[A-Za-z0-9_.-]+/",
        rb"/mnt/[a-z]/(?:80_Dev|Users)/",
    )
    def scan(data: bytes, name: str) -> None:
        if any(re.search(p, data) for p in patterns):
            raise ValueError(f"candidate content requires private-content review: {name}")
        if any(root.encode() in data for root in private_roots):
            raise ValueError(f"candidate contains producer path: {name}")

    for path in sorted(set(paths)):
        if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents if p != ROOT.parent):
            raise ValueError(f"candidate input missing or linked: {path.relative_to(ROOT)}")
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        if release_gate and data != _git("show", f"{source_commit}:{name}"):
            raise ValueError(f"release payload differs from Git blob: {name}")
        if path.suffix == ".zip":
            verify_archive(path)
            with zipfile.ZipFile(path) as nested:
                for item in nested.infolist():
                    if item.filename.endswith(".zip"):
                        raise ValueError("nested fixture archives are outside this candidate contract")
                    scan(nested.read(item), name + "!" + item.filename)
        else:
            scan(data, name)
        payloads[name] = data
    if not payloads:
        raise ValueError("empty candidate")
    coverage = [{"original_path": "source://" + name, "sha256": digest(data), "archived_as": name}
                for name, data in sorted(payloads.items())]
    payloads["frozen-coverage.json"] = (json.dumps(coverage, indent=2) + "\n").encode()
    if release_gate:
        if _clean_commit() != source_commit:
            raise ValueError("Git HEAD changed during candidate build")
        source_tree_dirty = False
    else:
        try:
            source_commit = _git("rev-parse", "HEAD").decode().strip()
            source_tree_dirty = bool(_git("status", "--porcelain").strip())
        except (OSError, subprocess.CalledProcessError):
            prior = ROOT / "DELIVERY-MANIFEST.json"
            prior_manifest = json.loads(prior.read_text()) if prior.is_file() else {}
            source_commit = prior_manifest.get("source_commit", "UNVERSIONED")
            source_tree_dirty = prior_manifest.get("source_tree_dirty")
    approved_for_private_ci = False
    owner_approval = "PENDING"
    first_party_license = "PENDING"
    if release_gate:
        release_config = json.loads((ROOT / "docs/stage2b-public-files.json").read_text(encoding="utf-8"))
        owner_approval = release_config.get("owner_approval", "PENDING")
        first_party_license = release_config.get("license", "PENDING")
        approved_for_private_ci = owner_approval == "APPROVED_PRIVATE_CI"
        if approved_for_private_ci:
            if (first_party_license != "Apache-2.0" or "LICENSE" not in payloads
                    or b"Apache License" not in payloads["LICENSE"][:250]
                    or len(payloads["LICENSE"]) < 11000):
                raise ValueError("owner-approved private CI requires the complete Apache-2.0 LICENSE")
    manifest = {
        "scope": ("Licensed private CI staging candidate; public release not authorized; not held-out data"
                  if approved_for_private_ci else "Unpublished development candidate; license approval pending; not held-out data"),
        "source_commit": source_commit,
        "source_tree_dirty": source_tree_dirty,
        "source_identity_scope": "commit is repository base; manifest payload hashes identify the selected bytes",
        "candidate_status": ("LICENSED_PRIVATE_CI_NOT_PUBLIC" if approved_for_private_ci else "PARTIAL_LICENSE_PENDING"),
        "clean_commit_gate": release_gate,
        "public_allowlist": "docs/stage2b-public-files.json" if release_gate else None,
        "owner_approval": owner_approval,
        "license": first_party_license,
        "frozen_paths": len(coverage), "external_binary_fingerprints": 0,
        "files": [{"path": name, "size_bytes": len(data), "sha256": digest(data)}
                  for name, data in sorted(payloads.items())],
    }
    if release_gate:
        manifest["source_identity_scope"] = "every selected payload equals its clean source_commit Git blob"
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for name, data in payloads.items():
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (source / "DELIVERY-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output / ("AI-Infra-Stage2B-Candidate.zip" if release_gate else "AI-Infra-Stage2A-Candidate.zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(source).as_posix(), (2026, 10, 19, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, path.read_bytes())
    verified = verify_archive(archive)
    result = {"status": "PASS_SELECTED_CANDIDATE_BUILD", "source_commit": manifest["source_commit"],
              "source_tree_dirty": source_tree_dirty,
              "source": str(source), "archive": str(archive), "archive_sha256": digest(archive.read_bytes()),
              "files": len(payloads), "package_check": verified["status"],
              "license": first_party_license, "owner_approval": owner_approval, "published": False,
              "scope": "selected source/fixture integrity only; subsequent execution and independent rights review required"}
    (output / "build-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-gate", action="store_true", help="require clean Git and exact Stage 2B allowlist")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output.resolve(), release_gate=args.release_gate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
