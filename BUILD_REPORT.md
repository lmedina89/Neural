# MicroMind v0.1.1 — Learning Stability & Skill Retention

**Build marker:** `STABRET-011`

## Baseline

Built directly from MicroMind v0.1.0.1.2 `SAVEREC-01012`. The 10→24 recurrent actor-critic, 7-action space, world physics, reward decomposition, live neural instrumentation, historical checkpoints, explicit Manual/Autosave recovery slots and GitHub Pages deployment structure are preserved.

## Reason for milestone

Physical iPhone training demonstrated non-monotonic policy behavior: strong balanced policies could later specialize into low-thrust/high-survival behavior or otherwise lose foraging competence. The old Best protection saved good weights but did not measure earlier skills consistently or constrain the optimizer enough.

## Implemented

- fixed all-four-stage `validation:v2` protocol;
- bounded per-stage skill retention scores;
- catastrophic-forgetting detection;
- curriculum promotion gate requiring retained prior skills;
- separate protected Balanced / Overall / Forager / Survivor / Efficiency archives;
- schema-3 persistence for archive/retention state;
- schema-1 and schema-2 migration;
- re-evaluation of legacy v0.1.0.1.x Best on the new protocol before comparison;
- PPO clip reduction to 0.12;
- conservative 5e-5…2e-4 adaptive learning-rate range;
- target-KL epoch early stopping;
- hard-KL whole-update rollback including Adam state;
- training-step entropy schedule;
- low-entropy rescue multiplier;
- automatic Best-Balanced recovery after severe regression or catastrophic forgetting;
- recovery learning-rate cooldown;
- 25k-step follow-up validation during recovery;
- UI metrics for LR, KL, PPO epochs and skill retention;
- Observe/Probe access to protected specialist brains.

## Automated verification

- core tests: **28/28 PASS**
- JavaScript syntax checks: PASS
- clean static production build: PASS
- checkpoint schema-2 → schema-3 migration test: PASS
- hard-KL rejected-update exact model rollback test: PASS
- fixed full-skill deterministic validation test: PASS
- catastrophic-forgetting / automatic recovery test: PASS
- recovery-safe lower-step Manual Save protection remains tested.

## Benchmark evidence

The release smoke run begins around -9.674 held-out return and learns a strongly positive held-out policy. A 150k stress run reproduces later policy forgetting; the new retention guard detects it, restores the protected policy without rewinding experience, reduces LR, and arms more frequent validation.

The stress result is intentionally reported rather than hidden: rollback protection is a containment mechanism, not a final solution to continual learning.

## Physical iPhone acceptance gate

After deploying:

1. Load the current v0.1.0.1.2 Manual Save explicitly.
2. Confirm the expected multi-million step count remains intact.
3. Resume and allow the immediate v0.1.1 rebaseline validation to complete.
4. Verify the old protected Best remains available during migration and gets re-evaluated under the new protocol.
5. Verify Skill Retention populates all four stages.
6. Verify protected specialist options populate.
7. Train through at least two 250k historical milestones while watching LR/KL/retention.
8. Confirm any automatic recovery leaves the total experience counter monotonic.
9. Run Unseen Test and Compare Brains.
10. Check Safari responsiveness/heat in Balanced mode.

## Known limitations

- recurrence remains stop-gradient through time rather than sequence BPTT/GRU;
- validation/recovery preserves good behavior but does not itself consolidate skills inside the network;
- a single 1,040-parameter policy may still face interference between objectives;
- fixed validation adds periodic CPU work, though it is infrequent outside recovery;
- no curiosity/world model/planning is included in this milestone.
