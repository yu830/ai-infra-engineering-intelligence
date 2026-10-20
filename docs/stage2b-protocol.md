# Minimum benchmark and release-check protocol

This slice measures evidence consistency for the exposed E012 development
case. It has no held-out task, Agent-effect endpoint, performance endpoint or
patch-correctness endpoint. All conclusions are DIAGNOSIS_ONLY.

1. Validate exact ZIP members, CRC, SHA256, coverage mapping and schema before
   using observations. Never reopen an `original_path` as a filesystem input.
2. Recompute full-array NaN/Inf counts, mask-region finite values, error and
   allclose with the fixed absolute and relative tolerances `2e-4`.
3. Report the five development criteria individually: fully masked fastpath
   nonfinite region; disabled fastpath finite; agreement with the reference;
   partial/no-mask negative controls; finite, nonzero fallback encoder output.
   Missing input tensors/model state leave their hashes as declared identity.
4. For a new live case/worker/runtime combination, run baseline, path-control
   and reference-control through the real transport and call `require_preflight`
   before `freeze_case`. Completed envelopes with failed observation rows must
   be rejected. Legitimate diagnostic NaNs are not instrument execution errors.
5. A valid newer backend may return NOT_REPRODUCED. A contradictory or corrupt
   record fails; do not change thresholds to make it match the example.
6. Build with `python -m experiments.build_stage2a_candidate --release-gate
   --output <new-directory>` only from clean Git. Compare every selected file
   against its commit blob, scan content and verify the resulting archive.

Install dependencies normally with the locked test group; first installation
can need network. Then run replay and tests without torch. An isolated runtime
is reported separately from another host or GPU. Missing GPU checks are NOT
RUN or SKIP, never GPU PASS. Remote Actions is NOT RUN until explicit remote
write/run authorization and actual logs/results are obtained.

Retain failing commands, stderr, JUnit, exit status, environment, artifact
hashes and any SKIP reason. A hash match is byte consistency, not a signature
or proof of physical execution. First-party ownership and license are approved for this private CI candidate.
Actual hosted CI (without extra charges), independent physical-machine replay,
and eventual public release remain separate gates.
