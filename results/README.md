# Experiment outputs

[`offline-validation/`](offline-validation/) records a mock run of E1–E4 at
40 scenes and 24 turns, using the calibrated offline rule judge. These outputs
validate the harness; they are **not live-model findings** and must not be used
as evidence for the paper's behavioural predictions.

See [the validation report](../docs/OFFLINE-VALIDATION.md) for the source commit,
setup correction, verified outcomes, limitations and commands to reproduce it.
Each result JSON records its backend, judge fingerprint and draft lock status.
`judge_checks/` contains the corresponding rule-judge calibration record.

Future live results should retain separate directories per model family, as
produced by `experiments/run_cross_family.sh`.
