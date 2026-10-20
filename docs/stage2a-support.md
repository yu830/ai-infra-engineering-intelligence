# Supported development slice

This candidate is unpublished and license approval is pending. It provides
DIAGNOSIS_ONLY evidence inspection. E001–E012 are exposed development cases;
none is a fresh held-out effect sample. Stage 1F remains PARTIAL.

## Environment matrix

| Operation | Target | Evidence / limit |
| --- | --- | --- |
| Integrity and numerical replay | Python 3.12, standard library | No torch or GPU needed; recorded observations only |
| Offline contract tests | POSIX Linux x86-64, Python 3.12, pinned test group | Uses real child processes and filesystem publication; CPU synthetic probes |
| Stage 2B release-source tests/build | Same POSIX target plus Git on PATH | Tests create disposable local repositories; strict build requires its own clean checkout. Installed replay does not require Git. |
| Optional E012 CUDA reproduction | Existing PyTorch 2.9.1+cu128, Python 3.12, WSL2 Linux x86-64, RTX 5070 Laptop 8GB | Current locally tested configuration; different versions can legitimately not reproduce |
| Native Windows Transport | Unsupported | POSIX directory/no-follow and process contract has not been ported |
| Other GPUs, operating systems, Python versions | Unverified | No compatibility claim |
| GitHub Actions | Configuration supplied | Remote workflow NOT RUN until an authorized push/run |

The retained Stage 2A GPU run reports driver **617.42**, device capability
**12.0**, device total memory **8,518,041,600 bytes**, and
`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.
The PyTorch binary reports git version
`5811a8d7da873dd699ff6687092c225caffcf1bb`. These are recorded environment
facts, not a claim that all other driver/kernel combinations are unsupported.

The lock records the optional CUDA stack but the default dependency set has no
torch. Installing the test group must not select the `gpu` extra. Optional GPU
installation is a separate operator choice, not required for evidence replay.

## Default offline setup

From the extracted candidate root on the supported POSIX target:

```sh
uv sync --locked --no-default-groups --group test
uv run --locked --no-sync python -m pytest tests/stage2a -m 'not gpu'
```

`uv lock --check` checks lock freshness. A CUDA test reported SKIP is not a GPU
execution PASS. Hash verification establishes consistency of packaged bytes;
it is not a signature, proof of physical execution, or approval to release.

## Optional GPU reproduction

Use an already available compatible CUDA Python environment. The commands
below generate new development observations through the actual Transport;
they do not start an Agent, execute a patch, or install anything:

```sh
python examples/run_e012_gpu.py --output-root ./local-e012
python - <<'PY'
import json
from pathlib import Path
from experiments.replay_e012 import audit_raw_records
root = Path('local-e012')
proof = json.loads((root / 'preflight-proof.json').read_text())
assert proof['status'] == 'PASS', 'Transport preflight failed; inspect retained evidence'
paths = {row['mode']: root / row['private_raw_path'] for row in proof['rows']}
result = audit_raw_records(
    paths['baseline'], paths['path-control'], paths['reference-control'],
    reference_fixture=Path('examples/e012'), output=root / 'numeric-replay.json')
print(json.dumps(result, indent=2, allow_nan=False))
PY
```

Choose a new output directory for each attempt; existing evidence must not be
overwritten. A newer backend may legitimately eliminate the bug, yielding a
valid `NOT_REPRODUCED` result. A malformed, missing or contradictory record
is an audit failure. The comparison reports reference criterion differences;
it must not force a different environment to match the example's score.
Live preflight proofs bind the current worker, interpreter, tools and retained
publication files, and expire. They are local consistency gates, not portable
execution certificates. A new Case × Worker × Transport combination must run
its own preflight before using `freeze_case`; historical freeze APIs are
preserved and are not retroactively guarded or revalidated by this new API.

The driver is **upstream-reproduction-based**, with provenance linked to
[PyTorch #199054](https://github.com/pytorch/pytorch/issues/199054). It preserves
the original CPU RNG/input construction and then transfers to current CUDA;
this is not exact replay of the original PyTorch 2.10 CPU environment.

## Evidence interpretation

The scientific replay must derive supported counts, mask regions and numerical
comparisons from archived full output arrays, rather than trust stored summary
booleans. Identity strings for inputs/model state remain declared identities
where the underlying tensors are absent. The CPU reference implements an
explicit all-masked zero-attention convention; an encoder output can remain
nonzero because of residual and feed-forward operations.

The mechanical development score is distinct from private Agent diagnosis
scores. It does not measure Skill effect, GPU-active time, speedup, patch
correctness or end-to-end benefit. No new model efficacy study runs here.

## Rights and publication

Read [rights review](stage2a-rights.md), [license options](license-options.md)
and [license status](../LICENSE-PENDING.md). Candidate license files are drafts
only. No public release is authorized. The selected bundle omits private Agent
traces, historical hidden answers, Issue bodies/attachments and framework
source/binaries. A future held-out suite needs separately controlled inputs.
