# MicroMind v0.1.1.1 — Validation Calibration & Generalization Diagnostics

**Build:** `VALCAL-0111`

MicroMind is a browser-based reinforcement-learning laboratory built around a deliberately tiny, inspectable recurrent actor-critic. v0.1.1.1 is a calibration milestone built directly from v0.1.1 `STABRET-011` after physical iPhone testing revealed an important research problem: the fixed skill-retention validator could rank an older protected brain above a newer policy even when a separate unseen-world test showed the newer policy outperforming it on every major observed metric.

This build does **not** make the brain larger and does **not** add curiosity, a world model, language, or planning. It makes the measurement system more trustworthy before more cognitive complexity is added.

## What changed

### Validation protocol v3

Automatic validation still tests every curriculum stage, but now uses **12 fixed episodes per skill** instead of 8 and reports a 95%-style confidence range for each skill score and the balanced score.

The protocol seed domain is now:

`validation:v3`

Each per-stage skill score is computed from per-episode competence samples, so uncertainty can be estimated rather than hidden behind one percentage.

### Repeat-confirmed forgetting

A single bad validation no longer causes an immediate automatic rollback.

Per-skill behavior now progresses through:

- no alert;
- **WATCH** after one statistically meaningful regression;
- **CONFIRMED** after the regression repeats on the next scheduled check.

A severe single-skill regression can block curriculum promotion while it is being confirmed, but one narrow skill drop alone does not automatically destroy a newer policy by restoring an old brain.

Automatic Best-Balanced recovery now requires stronger evidence: a repeat-confirmed balanced regression or multiple confirmed catastrophic skill regressions.

When a watch is active, the next validation is brought forward to the existing 25k-step recovery interval.

### v0.1.1 migration ordering fix

The first v0.1.1 physical migration exposed a subtle ordering problem: the old protected brain was re-evaluated after the first Latest forgetting decision, which could temporarily display `retention OK` even when the newly recalibrated historical skill best was much higher.

v0.1.1.1 recalibrates the preserved archive **before** assessing Latest against it.

### Protected archive recalibration

v0.1.1 used `validation:v2`. Those numeric scores are not compared directly with v0.1.1.1 `validation:v3` scores.

When a schema-3 v0.1.1 save is loaded:

1. Latest, total experience, optimizer, curriculum, milestones and all protected specialist models are preserved;
2. the existing protected models remain inspectable while paused;
3. restoring an old archive brain is disabled until recalibration completes;
4. Resume/Learn triggers immediate v3 evaluation before training advances;
5. every unique preserved specialist model is re-evaluated under validation:v3;
6. the new archive is rebuilt only from compatible v3 measurements;
7. the current Latest is then assessed against that recalibrated baseline.

### Final holdout is now truly all-skills

The **Unseen Test** no longer tests only the current curriculum stage.

It now evaluates all four skills on a separate final domain:

`heldout:final:v2`

with 8 episodes per skill, for 32 total held-out episodes.

It reports:

- all-skills generalization score + range;
- return;
- food;
- survival;
- remaining energy;
- episode length;
- each individual held-out skill score + range.

The same current Latest weights also receive a **read-only validation:v3 measurement** during Unseen Test so validation and final-holdout rankings are compared on the exact same current policy.

Critically, Unseen Test is diagnostic only. It does **not**:

- update model weights;
- update optimizer state;
- replace a protected archive brain;
- update curriculum;
- append validation history;
- trigger automatic rollback.

If validation and the final holdout rank Latest vs Protected Balanced differently, the UI reports a **VALIDATION / HELD-OUT CONFLICT** instead of automatically choosing one.

Repeatedly consulting any final holdout can still bias human development decisions, so the UI explicitly recommends using it sparingly.

### Comparison domain separated from final holdout

**Compare Brains** now uses:

`heldout:compare:v2`

rather than consuming the final Unseen Test domain.

Historical comparisons are also all-skills, with a lighter 4 episodes per skill to keep the mobile comparison cost bounded.

### Save schema 4

v0.1.1.1 writes checkpoint schema 4 and reads schemas 1–4.

Schema 4 adds calibrated skill-best records, confidence ranges, per-skill confirmation streaks, balanced-regression confirmation state, archive-recalibration state, and preserves incompatible older validation records separately as legacy history.

The recovery-safe Manual Save / Validation Autosave split and lower-step overwrite guard remain intact.

## Recommended upgrade from v0.1.1

Before replacing the deployed files, press **Save Manual** in the currently running v0.1.1 page so the newest multi-million-step Latest and all protected specialists are preserved.

After deploying v0.1.1.1:

1. refresh the page;
2. verify the Manual Save still shows the expected multi-million-step run;
3. press **Load Manual**;
4. confirm training is paused and the experience counter is correct;
5. do **not** press Restore Selected while calibration is pending;
6. press **Resume**;
7. wait for the one-time v3 archive recalibration to complete;
8. inspect the Skill Retention ranges and WATCH/CONFIRMED labels;
9. press **Save Manual** after calibration if you want an immediate schema-4 checkpoint;
10. use Unseen Test sparingly as a final diagnostic.

Do not clear Safari website data between versions; checkpoints live in IndexedDB.

## Run locally

```bash
npm test
npm run build
python3 -m http.server 8080 --directory dist
```

Then open `http://localhost:8080/`.

## Benchmark

```bash
npm run benchmark
STEPS=150000 npm run benchmark
```

The release benchmark uses a separate `benchmark:all-skills:v2` domain, not the final Unseen Test domain. See `docs/BENCHMARK.md` and `BUILD_REPORT.md`.

## Scope boundary

v0.1.1.1 deliberately keeps the exact same **1,040-parameter** policy architecture and the same environment/reward mechanics as v0.1.1. The next cognitive milestone should not begin until this measurement system is physically accepted on iPhone.
