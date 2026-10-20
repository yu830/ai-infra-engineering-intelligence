"""Release source identity and accidental-disclosure regressions use real Git."""
import json
import subprocess
import zipfile

import pytest

from experiments import build_stage2a_candidate as builder


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE).decode().strip()


def _commit(root):
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Release Test", "-c", "user.email=test@example.invalid",
         "commit", "-m", "Synthetic release source")


def _source(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "core.autocrlf", "false")
    config = root / "docs/stage2b-public-files.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"files": ["README.md", "docs/stage2b-public-files.json"]}))
    (root / "README.md").write_text("first-party development candidate\n")
    _commit(root)
    monkeypatch.setattr(builder, "ROOT", root)
    return root


def test_clean_source_commit_and_exact_allowlist(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    # An ignored private tree is permitted locally, never exported.
    (root / ".git/info/exclude").write_text("private/\n")
    (root / "private").mkdir()
    (root / "private/trace.txt").write_text("do not export")
    result = builder.build(tmp_path / "delivery", release_gate=True)
    assert result["source_tree_dirty"] is False
    assert result["source_commit"] == _git(root, "rev-parse", "HEAD")
    with zipfile.ZipFile(result["archive"]) as archive:
        assert set(archive.namelist()) == {
            "README.md", "docs/stage2b-public-files.json",
            "DELIVERY-MANIFEST.json", "frozen-coverage.json"}
        manifest = json.loads(archive.read("DELIVERY-MANIFEST.json"))
        assert manifest["owner_approval"] == "PENDING"
        assert manifest["clean_commit_gate"] is True


@pytest.mark.parametrize("kind", ["tracked", "untracked", "staged"])
def test_dirty_sources_rejected_before_output(tmp_path, monkeypatch, kind):
    root = _source(tmp_path, monkeypatch)
    (root / ("new.txt" if kind == "untracked" else "README.md")).write_text("changed")
    if kind == "staged":
        _git(root, "add", ".")
    with pytest.raises(ValueError, match="clean Git"):
        builder.build(tmp_path / "delivery", release_gate=True)
    assert not (tmp_path / "delivery").exists()


def test_unversioned_source_cannot_inherit_a_prior_manifest(tmp_path, monkeypatch):
    root = tmp_path / "unversioned"
    root.mkdir()
    (root / "DELIVERY-MANIFEST.json").write_text('{"source_commit":"pretend"}')
    monkeypatch.setattr(builder, "ROOT", root)
    with pytest.raises(ValueError, match="available Git checkout"):
        builder.build(tmp_path / "delivery", release_gate=True)


def test_export_directory_cannot_borrow_parent_checkout(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    (root / ".git/info/exclude").write_text("export/\n")
    exported = root / "export"
    exported.mkdir()
    monkeypatch.setattr(builder, "ROOT", exported)
    with pytest.raises(ValueError, match="own Git checkout root"):
        builder.build(tmp_path / "delivery", release_gate=True)


def test_clean_git_status_does_not_replace_blob_byte_identity(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    (root / ".gitattributes").write_text("README.md text eol=crlf\n")
    _commit(root)
    # Git may retain an unchanged working file despite checkout-index --force.
    # Remove only this disposable test file so checkout applies the EOL rule.
    (root / "README.md").unlink()
    _git(root, "checkout-index", "--force", "--", "README.md")
    _git(root, "add", "--renormalize", "README.md")
    assert not _git(root, "status", "--porcelain")
    assert b"\r\n" in (root / "README.md").read_bytes()
    with pytest.raises(ValueError, match="differs from Git blob"):
        builder.build(tmp_path / "delivery", release_gate=True)


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "a/../b", "a\\b", "C:/outside.py"])
def test_allowlist_cannot_escape_root(tmp_path, monkeypatch, name):
    root = _source(tmp_path, monkeypatch)
    (root / "docs/stage2b-public-files.json").write_text(json.dumps({"files": [name]}))
    _commit(root)
    with pytest.raises(ValueError, match="unsafe release allowlist"):
        builder.build(tmp_path / "delivery", release_gate=True)


def test_committed_secret_still_rejected(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    synthetic_key = "ghp_" + "A" * 30
    (root / "README.md").write_text(synthetic_key)
    _commit(root)
    with pytest.raises(ValueError, match="private-content review"):
        builder.build(tmp_path / "delivery", release_gate=True)
    assert not (tmp_path / "delivery").exists()
