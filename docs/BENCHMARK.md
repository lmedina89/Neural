# v0.1.1 Benchmark

Release benchmark seed: `424242`.

The benchmark trains on the Motor Nursery domain and evaluates on a separate fixed `heldout:v1` domain. This is a smoke/learning-stability experiment, not a claim of broad general intelligence.

## 60k release smoke

A representative release run reached approximately:

- initial held-out return: **-9.674**
- initial food: **0.094**
- initial survival: **0%**
- latest at ~60k return: **+2.93**
- latest food: **2.78**
- latest survival: **96.9%**
- protected Balanced at ~50k return: **+2.64**

## 150k stress observation

The longer deterministic training experiment intentionally exposed the instability that motivated this milestone. A strong ~50k policy later forgot earlier competence. v0.1.1 detected the loss in the fixed all-skills suite and restored the protected Balanced policy while keeping the experience counter monotonic and reducing the optimizer learning rate.

Follow-up validations are shortened to 25k-step intervals during recovery. Catastrophic-forgetting alarms can trigger the same guard even when the aggregate balanced-score regression alone has not yet crossed its threshold.

This is an important distinction: **v0.1.1 does not claim PPO can no longer regress. It detects and contains regressions much earlier and preserves usable policies while we study the root continual-learning problem.**
