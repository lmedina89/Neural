# MicroMind v0.1.4.1.2 — Build Report

**Version:** `0.1.4.1.2`  
**Build marker:** `OCCSPIN-01412`  
**Parent baseline:** v0.1.4.1.1 `ROTAUD-01411`

## Objective

Measure the user-observed live-world spin specifically around loss of a clean/direct target route or line-of-sight before changing the learner.

The v0.1.4.1.1 rear-target audit showed that the accepted long-running Learner and Balanced Champion successfully handle clean rear targets in an empty arena. Therefore this build moves from “rear bearing” to the interaction among food direction, wall geometry, danger rays, recurrent state, angular motion and action selection.

## Implementation

1. Added `src/evaluation/occlusionSpinTelemetry.js` as a read-only diagnostic module.
2. Added exact segment/rectangle line-of-sight measurement from agent center to nearest-food center.
3. Added direct-path measurement using the **actual agent collision radius** to expand walls. Spawn-only `wallMargin` is intentionally excluded.
4. Added a diagnostic ±70° forward-cone flag. It is never added to the policy observation.
5. Added a bounded `LiveSpinRecorder` used only by normal **OBSERVE** stepping.
6. The live recorder captures food bearing/distance, LOS/path status, danger rays, speed/omega, sampled action/probabilities/value, recurrent-state summary, reward parts, target identity and food progress.
7. Potential spin events require rolling angular travel >=0.90 rotations while food-distance progress is <=0.035 over the diagnostic window. Food-collection windows are excluded from triggering.
8. Each event stores bounded pre-event context plus a short recovery window and receives a descriptive context classification. At most 12 completed events are kept in memory.
9. Added a controlled paired audit with OPEN, CLEARANCE (path blocked / centerline clear) and OCCLUDED (centerline blocked) geometry. Eight seeded stochastic trials are run per case by default.
10. Learner and Balanced Champion controlled audits are guarded by serialize-before/after policy equality checks.
11. LIVE world metadata exposes `diag DIRECT`, `diag PATH BLOCKED`, or `diag LOS BLOCKED` so physical testing can correlate behavior with the external geometry measurement.
12. All new event/audit data is session-only. Save schema remains **10**.

## Learning safety

The following parent files are intentionally unchanged byte-for-byte:

- `src/ai/model.js`
- `src/ai/ppo.js`
- `src/ai/rollout.js`
- `src/ai/session.js`
- `src/ai/curiosity.js`
- `src/sim/world.js`
- `src/sim/curriculum.js`
- `src/evaluation/evaluator.js`
- `src/storage/checkpoints.js`
- `src/utils/prng.js`
- `src/app/runtimeDiagnostics.js`
- all visualization renderer modules

The only behavior-path integration is read-only sampling around `stepObserved()` in the app layer. LEARN/training rollouts are not instrumented by this recorder.

## Verification

Final verification results are appended after packaging.

## Final verification results

- Full Node test suite: **100/100 passed**.
- Static `dist/` rebuild completed successfully.
- JavaScript syntax check passed for every source, script and test module.
- Protected policy/learning/simulation/storage/runtime files and every visualization renderer are **byte-for-byte identical** to v0.1.4.1.1.
- Deterministic parent/candidate continuation parity: same seed, 8 environments and **3,456 training steps** with ordinary validation disabled produced byte-identical policy, optimizer, curiosity, curriculum, environment, recurrent hidden state, action RNG, rehearsal, episode and learner-experience state.
- Geometry contract tests verify three distinct diagnostic states: OPEN, collision-corridor blocked while raw LOS stays clear, and raw LOS blocked. The policy observation shape remains 10 and food X/Y/distance are not replaced by the diagnostic.
- Controlled occlusion audit is fixed-seed deterministic and exact session/model state is unchanged before versus after execution.
- Live recorder test verifies sustained poor-progress rotation is captured and wall/food conflict context can be classified without policy feedback.
- Rear-target audit, validation-confidence tests, curiosity A/B tests, spawn-clearance tests, Hall/branching tests, runtime diagnostics and final-heldout isolation tests all continue to pass.
- Save schema remains **10**; no telemetry or controlled-audit result is serialized.
