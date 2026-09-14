# MicroMind v0.1.1.1 — Validation Calibration & Generalization Diagnostics

**Build marker:** `VALCAL-0111`

## Exact baseline

Built directly from the packaged MicroMind v0.1.1 `STABRET-011` repository. The extracted v0.1.1 directory was byte-for-byte compared with the delivered v0.1.1 ZIP before modification.

The following core systems were intentionally left unchanged:

- 10→24 recurrent actor-critic architecture / 1,040 parameters;
- PPO optimizer implementation and KL safety controller;
- world physics;
- reward decomposition;
- sensor semantics;
- curriculum world definitions;
- neural/world renderers;
- checkpoint IndexedDB storage implementation.

## Why this milestone exists

Physical iPhone testing exposed two measurement problems rather than a lack of raw learning:

1. a first migration could display `retention OK` even after the recalibrated historical Motor Nursery best was much higher, because forgetting was assessed before the old protected brain updated the baseline;
2. at ~3.84M steps, the final unseen test showed Latest substantially outperforming Protected Balanced while the fixed skill validator still preferred the older protected policy.

The correct response was to improve measurement and evidence handling, not enlarge the network.

## Implemented

- release identity `v0.1.1.1 / VALCAL-0111`;
- checkpoint schema 4 with schema 1–4 reader support;
- `validation:v3` protocol;
- 12 fixed evaluation episodes per skill;
- per-episode competence scoring;
- 95%-style skill and balanced uncertainty ranges;
- archive-protocol recalibration before Latest forgetting assessment;
- preservation/re-evaluation of all unique v0.1.1 specialist archive models;
- Restore Selected disabled while archive recalibration is pending;
- first observation = WATCH, repeated evidence = CONFIRMED;
- accelerated 25k follow-up validation while a watch is active;
- automatic rollback only after repeat-confirmed balanced regression or multiple confirmed catastrophic skills;
- single-skill specialization can block promotion without immediately destroying Latest;
- separate `heldout:compare:v2` checkpoint-comparison domain;
- separate `heldout:final:v2` final Unseen Test domain;
- Unseen Test upgraded from current-stage-only to all four skills;
- read-only same-weights validation measurement during final Unseen Test;
- explicit validation/heldout conflict diagnostic;
- final holdout cannot update archives, rollback state, optimizer, curriculum or validation history;
- mobile Skill Retention panel now shows point estimate, range, WATCH/CONFIRMED severity and streak.

## Automated verification

- core tests: **34/34 PASS**
- validation determinism: PASS
- validation/compare/final seed-domain separation: PASS
- confidence-bound integrity: PASS
- WATCH→CONFIRMED repeated-evidence test: PASS
- repeat-confirmed balanced recovery test: PASS
- schema-4 roundtrip including legacy validation-history preservation: PASS
- schema-3 v0.1.1 archive recalibration migration: PASS
- schema-2 and schema-1 legacy migration: PASS
- final Unseen Test non-mutating path audit: PASS
- all-skills heldout evaluation: PASS
- JavaScript syntax checks: PASS

## Benchmark

60k all-skills smoke on `benchmark:all-skills:v2`:

- initial score: **4.3%**
- latest ~60k: **37.7%**
- protected ~50k: **36.9%**

150k regression stress:

- first destructive evidence at ~100k: WATCH only, no rollback;
- repeated evidence at 125,184: CONFIRMED;
- automatic recovery restored the protected ~50k policy while keeping total experience at 125,184;
- recovery LR reduced to 1.2e-4.

See `docs/BENCHMARK.md` for details.

## Physical iPhone acceptance gate

1. Save the current v0.1.1 long-run Manual checkpoint before deployment.
2. Deploy v0.1.1.1 and refresh.
3. Load Manual and verify the exact multi-million-step age.
4. Confirm old protected specialist options remain visible but Restore Selected is disabled while calibration is pending.
5. Resume once and wait for v3 archive recalibration.
6. Verify ranges and retention status populate.
7. Save Manual again after calibration to create a schema-4 checkpoint.
8. Train through at least two validation events and verify a one-off WATCH does not auto-restore.
9. Run one final Unseen Test and inspect any validation/heldout conflict message.
10. Confirm Safari responsiveness and thermal behavior remain acceptable.

## Known limitations

- confidence ranges are approximate normal intervals over seeded stochastic episode competence;
- validation is still finite and can mis-rank policies;
- repeated human inspection of the final holdout can leak information into development decisions;
- recurrence still lacks sequence BPTT/GRU credit assignment;
- the system preserves/recovers policies externally rather than consolidating old skills internally;
- no curiosity, world model, planning or language is added here.
