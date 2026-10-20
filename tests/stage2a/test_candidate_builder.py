"""The export boundary must survive accidental private files and broken inputs."""
from pathlib import Path
import zipfile
import pytest
from experiments import build_stage2a_candidate as builder
from experiments.verify_evidence_zip import verify_archive


def _source(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / 'producer'
    root.mkdir()
    for name in builder.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('first-party development file\n')
    for tree in builder.TREES:
        (root / tree).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(builder, 'ROOT', root)
    return root


def test_export_is_selected_and_rejects_overwrite(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    (root / 'private').mkdir()
    (root / 'private/trace.txt').write_text('private trace must stay local')
    result = builder.build(tmp_path / 'delivery')
    archive = Path(result['archive'])
    assert verify_archive(archive)['status'] == 'PASS_PACKAGE_CONSISTENCY'
    with zipfile.ZipFile(archive) as bundle:
        assert not any('private/' in name for name in bundle.namelist())
    with pytest.raises(ValueError, match='new directory'):
        builder.build(tmp_path / 'delivery')


def test_allowlisted_file_still_rejects_private_content(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    # Construct the synthetic forbidden URL at runtime so the distributable
    # test source itself contains no private-Drive URL.
    forbidden_url = 'https://' + 'drive.google.com/file/d/' + 'private-fixture/view'
    (root / 'README.md').write_text(forbidden_url)
    with pytest.raises(ValueError, match='private-content review'):
        builder.build(tmp_path / 'delivery')
    assert not (tmp_path / 'delivery').exists()


def test_missing_input_fails_before_export(tmp_path, monkeypatch):
    root = _source(tmp_path, monkeypatch)
    (root / 'pyproject.toml').unlink()
    with pytest.raises(ValueError, match='missing or linked'):
        builder.build(tmp_path / 'delivery')


@pytest.mark.posix
def test_linked_input_cannot_smuggle_other_content(tmp_path, monkeypatch):
    import os
    if os.name != 'posix':
        pytest.skip('symlink export boundary tested on POSIX')
    root = _source(tmp_path, monkeypatch)
    outside = tmp_path / 'outside.py'
    outside.write_text('private outside source')
    path = root / 'evals/stage2a/link.py'
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='missing or linked'):
        builder.build(tmp_path / 'delivery')
