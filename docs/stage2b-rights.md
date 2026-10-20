# Stage 2B rights and distribution decision

Status: **PRIVATE CI APPROVED — PUBLIC RELEASE STILL NOT AUTHORIZED**.

The exact proposed source distribution is `stage2b-public-files.json`. It is a
file-by-file allowlist, not a directory glob. Build receipts bind these bytes
to a clean Git commit; the manifest and coverage file are generated metadata.
The owner approved this exact first-party scope for an Apache-2.0-licensed private CI candidate. Public release is NOT authorized.

| Material | Proposed treatment | Rights boundary |
| --- | --- | --- |
| Selected first-party Python, tests and build/CI configuration | Included in private CI staging only | Apache-2.0 confirmed; Copyright 2026 Yu Jiacheng; owner confirmed no employer, school or collaborator restrictions |
| Protocol, support and rights documentation | Included in private CI staging only | Same proposed first-party license |
| E012 arrays, configuration, provenance and expected mechanical development checks | Included in private CI staging only | Newly generated first-party development observations; exposed, never held-out answers |
| Upstream Issue reproduction method | Cite, with driver labelled upstream-reproduction-based | Public accessibility does not license Issue prose, attachments or nontrivial copied expression |
| Torch and other dependencies | Resolve separately via pinned metadata | No framework/vendor source or binaries bundled; dependency licenses remain separate |
| Private execution records, Agent traces, hidden answers, original host paths, credentials | Exclude | Remain outside all public source/install artifacts |
| Historical reports and license comparison draft texts | Exclude | Remain in private repository/history; historical failures are summarized without repackaging private records |

The Stage 2B source comparison reread PyTorch #199054 and the current worker.
The driver preserves API choices, dimensions, masks and CPU RNG construction
for a reproduction-based test. No Issue prose or attachment is bundled. The
driver contains similar API setup and reproduction choices, which remain part
of the owner's rights review. This source comparison is not a clean-room
claim or a legal determination that all similarity is unprotected expression.
The first-party license cannot grant rights the owner does not hold.

Owner confirmed the legal first-party copyright holder (Yu Jiacheng), year 2026,
no employer/school/collaborator restrictions, Apache-2.0, and this exact first-party scope.
An effective Apache-2.0 LICENSE is included. This grants rights only to owned
material; upstream reproductions retain their independently documented boundaries.
The repository is private, and no public release or PR merge is authorized.
Owner authorized a new private yu830/ai-infra-engineering-intelligence repository
and writes to ci/stage2b-release-gate only. Hosted CI must not incur extra charges. Publishing the historical repository would expose
excluded history; an approved public repository must contain only reviewed
distribution history. Never push this private historical branch as a public
benchmark repository.

The historical Stage 1F study remains PARTIAL, with two started sessions and
six NOT RUN after a common transport fault; Skill benefit is UNKNOWN. E012 is
only a development/regression example. Neither release engineering nor the
five mechanical evidence checks constitute a new Agent efficacy experiment.

Official sources: [Apache text](https://www.apache.org/licenses/LICENSE-2.0.txt),
[MIT text](https://spdx.org/licenses/MIT.html),
[upstream Issue](https://github.com/pytorch/pytorch/issues/199054),
[PyTorch version license](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE).
See `license-options.md` for the recommendation and `stage2a-rights.md` for
the earlier provenance review.
