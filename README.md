# Infra Diagnostic Benchmark

This is an Apache-2.0-licensed, PRIVATE CI staging slice of AI Infra Engineering
Intelligence. Copyright 2026 Yu Jiacheng.
It checks archived numerical evidence and a Case × Worker × Transport contract.
It has not been publicly released, and it is NOT a held-out evaluation.

The example is exposed E012 development data, not a new held-out evaluation.
It contains locally generated numerical observations and expected development
checks, not private Agent traces or future formal answers. Offline replay does
not execute CUDA, establish hardware performance, or evaluate Skill benefit.

The tested starting point is POSIX Linux x86-64 / Python 3.12. Offline evidence
commands use only the standard library. Project tests use the pinned test
group; the Stage 2B release-source tests additionally require Git on PATH.
Git is not needed for installed replay or the original 36 Stage 2A tests.
Optional GPU reproduction requires a separately available, compatible
CUDA PyTorch environment; install or synchronize GPU dependencies only when
that is intended.

Run the offline development slice from this extracted source directory:

```sh
uv sync --locked --no-default-groups --group test
uv run --locked --no-sync python -m experiments.replay_e012 examples/e012
uv run --locked --no-sync python -m pytest tests/stage2a -m 'not gpu'
uv run --locked --no-sync python -m pytest tests/stage2b
```

The replay command also accepts the packaged E012 fixture ZIP. Use
`python -m experiments.verify_evidence_zip <archive.zip>` for integrity alone.
Integrity PASS and numerical audit PASS have different scopes; neither reruns
CUDA. A full-array mechanical case-evidence score is not an Agent diagnosis
score or evidence of a Skill effect. See [support and limitations](docs/stage2a-support.md)
for the platform matrix, and [rights review](docs/stage2a-rights.md) for content
boundaries. The lock includes optional CUDA dependencies; these commands do
not install that extra.

The historical Stage 1F
comparison remains PARTIAL: two sessions started before a common worker-output
contract fault stopped the study; six sessions did not run. Repair-only tests
do not restore that study or establish a Skill effect.

First installation may require network access for locked build/test packages.
Replay itself uses only the standard library. To test a normal wheel install,
run `uv build`, install its wheel into a fresh Python 3.12 environment, and run
`python -m experiments.replay_e012 <absolute-path-to-extracted-examples/e012>`
from a directory outside this checkout. The wheel contains the selected
Python modules; the source distribution carries the separate numerical fixture,
tests, protocol and documentation. No framework binary is bundled.

The owner-approved private-CI file list is [the allowlist](docs/stage2b-public-files.json).
[Protocol](docs/stage2b-protocol.md) and [rights decision](docs/stage2b-rights.md)
define the release gates. Build a private candidate from clean Git with
`python -m experiments.build_stage2a_candidate --release-gate --output <new-directory>`.
This rejects dirty/unversioned sources and verifies selected bytes against Git.
The builder intentionally requires Git; replay and installed runtime commands
do not. Comparison license drafts and historical private evidence are excluded.

Owner confirmed first-party rights and approved private CI validation.
A real GitHub Actions run and separate public-release approval are still required. Cross-device GPU reproduction is unverified. No general
platform, GPU, Agent or Skill improvement is claimed.
