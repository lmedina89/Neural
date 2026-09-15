# MicroMind v0.1.3 — Build Report

**Milestone:** Intrinsic Curiosity Foundation  
**Build marker:** `CURIOUS-013`  
**Baseline:** exact packaged v0.1.2.1 `CHAMPEFF-0121`  
**Baseline SHA-256:** `c330d629686c755617d226f627b1e1b19832b36f70da2c4205720beefc389049`  
**Save schema:** 7 (loads schemas 1–7)

## Scope containment
The accepted 1,040-parameter policy/PPO learner remains unchanged. Byte-for-byte unchanged baseline modules include policy model, PPO, GAE rollout math, world physics/rewards, curriculum, and evaluator.

## New intelligence component
- Added `src/ai/curiosity.js`: a **441-parameter learned forward predictor**.
- Input: current 10-value observation + one-hot 7-action choice.
- Hidden: 16 tanh units.
- Output: nine predicted dynamic next-observation values (constant bias feature excluded).
- Prediction surprise is normalized against a learned recent-error baseline.
- Curiosity is sampled every fourth transition for mobile efficiency.
- Intrinsic reward is positive-only, terminal-safe, capped per sampled step, and capped to **0.25 per episode**.
- Predictor trains after surprise is measured so a transition is not made artificially familiar before its novelty is scored.

## Evaluation isolation
Curiosity reward is never used by:
- `validation:v3` Champion selection;
- `heldout:compare:v2` historical comparison;
- `heldout:final:v2` final Unseen Test.

External world reward/episode return remains separately recorded and displayed.

## Visualization
- Added real Curiosity / Prediction neural display.
- Real prediction vs actual sensory outputs and error rings.
- Live prediction error, novelty, intrinsic bonus, predictor loss, episode budget, and predictor parameter count.
- Training-world novelty trail uses actual prediction surprise samples; no random decorative activity.

## Persistence / migration
- schema 7 saves/restores curiosity weights, Adam state, and error normalization state;
- schema-6 v0.1.2.1 policy/Champion saves migrate without altering those policy weights; curiosity starts fresh;
- v0.1.3 Champion/Hall/frozen-lineage snapshots carry curiosity state when available;
- pre-curiosity Champions without predictor state remain valid and receive a fresh predictor if explicitly used as a branch parent.

## Predictor evidence
80,128-step release sanity run:
- prediction error: **0.3188 → 0.0331** (sampled checkpoints);
- predictor loss: **0.15938 → 0.01656**;
- random initial all-skills generalization: **0.5%**;
- final Learner: **11.4%**;
- protected Champion: **15.5%**.

The intrinsic bonus remained deliberately small. This run proves the predictor learns; it is not a claim of long-horizon curiosity superiority.

## Performance evidence
Five-process development benchmark, identical workload:
- v0.1.2.1 median: **31,884 wall-clock steps/sec**;
- v0.1.3 median: **29,965 wall-clock steps/sec**;
- median overhead: **~6.0%**.

Physical iPhone Safari remains the performance authority.

## Automated QA
- `npm test`: **57/57 PASS** before final package verification.
- Curiosity predictor finite/parameter count: PASS.
- Deterministic-mapping learning test: PASS.
- Intrinsic reward cap/terminal-zero test: PASS.
- External-vs-intrinsic reward separation: PASS.
- Evaluation curiosity isolation: PASS.
- schema-6 migration + schema-7 curiosity roundtrip: PASS.
- Existing Hall/branch/rehearsal/Champion/heldout/PPO safety suites retained.

## Physical iPhone gate
1. On current v0.1.2.1, **Save Manual** first (subject to its existing higher-step protection).
2. Deploy v0.1.3 and verify `v0.1.3 • CURIOUS-013`.
3. Load the intended Manual/Autosave and confirm policy age, Champions, lineage and Hall survived.
4. Stay paused first: curiosity should show as fresh/untrained for a migrated schema-6 run.
5. Resume on Balanced. Watch prediction error/loss over ~50k–200k steps; familiar dynamics should trend lower while novelty still spikes.
6. Confirm the intrinsic bonus remains tiny and episode budget never exceeds 0.25.
7. Watch for reward-hacking behavior (intentional spinning, wall impacts, death seeking). Stop and report if observed.
8. Check simulation/training throughput and phone heat versus v0.1.2.1.
9. Use normal validation/Compare Brains for routine evidence. Use final Unseen Test sparingly and only after a meaningful curiosity-training interval.
