# v0.1.1.1 Benchmark

Release training seed: `424242`.

Benchmark evaluation uses `benchmark:all-skills:v2`. It is separate from training, validation, checkpoint-comparison and the final Unseen Test domain.

## 60k release smoke

Observed run:

- initial all-skills score: **4.3%** (approx. range 1.0–7.6%)
- latest at 60,160 steps: **37.7%** (26.4–48.9%)
- protected Balanced at 50,176 steps: **36.9%** (25.5–48.3%)
- initial mean return: **-9.448**
- latest mean return: **-0.097**
- initial food: **0.188**
- latest food: **1.313**
- initial survival: **0%**
- latest survival: **50%**

The purpose is to prove that the revised evaluator and guarded learner still learn; this is not a claim of general intelligence.

## 150k destructive-regression stress run

The longer run reproduced the type of policy collapse seen during physical testing:

- Best Balanced established around **50,176** steps;
- at **100,096** steps, balanced regression and a catastrophic Motor Nursery drop appeared as **WATCH x1**;
- no automatic rollback occurred on that first observation;
- the next check was accelerated;
- at **125,184** steps, the aggregate regression and Motor Nursery failure repeated and became **CONFIRMED x2**;
- Best Balanced was then restored automatically without rewinding the 125,184-step experience age;
- LR was reduced to **1.2e-4**;
- by ~150k, the live policy had begun recovering while the protected 50k policy remained available.

This is the intended behavior: calibration first, confirmation second, recovery only after repeated destructive evidence.
