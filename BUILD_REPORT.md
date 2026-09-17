# MicroMind v0.1.4.1.1 — Build Report

**Version:** `0.1.4.1.1`  
**Build marker:** `ROTAUD-01411`  
**Parent baseline:** v0.1.4.1 `VALCONF-0141`

## Objective

Measure the observed rear-target spinning failure before changing learning, sensors, rewards, memory, or physics.

## Evidence from the accepted code

- `World.observe()` always provides nearest-food relative X, relative Y, and distance. There is no food field-of-view or wall occlusion in the policy observation.
- The three limited rays are danger sensors.
- `BRAKE` multiplies linear `vx/vy` by `0.72`, but it does not apply a special angular brake. Angular velocity receives the same `angularDrag` used after every action.

Therefore the immediate question is whether the learned policy has a rear-bearing orientation / angular-control failure, not whether hidden-state memory loses an unseen food target.

## Implementation

1. Added `src/evaluation/orientationAudit.js` as a read-only diagnostic module.
2. Added fixed target bearings at ±179°, ±135°, ±90°, ±45° and 0°.
3. Added 8 seeded stochastic-policy trials per bearing, 180 steps maximum per trial.
4. Each trial starts from rest with zero recurrent state in an empty arena while preserving the real observation and angular physics paths.
5. Captured facing success, reach success, spin incidence, steps-to-face, cumulative angular travel, BRAKE-while-turning usage, turn-action usage, and initial policy preference.
6. Added a Research-panel control that pauses training, audits the exact Learner, then audits Balanced Champion on the same protocol when available.
7. Added runtime serialize-before/after guards to fail safely if an audit ever mutates policy weights.
8. Audit results are runtime-only; save schema remains **10**.

## Learning safety

The following parent files are intentionally unchanged byte-for-byte:

- `src/ai/model.js`
- `src/ai/ppo.js`
- `src/ai/rollout.js`
- `src/ai/session.js`
- `src/ai/curiosity.js`
- `src/sim/world.js`
- `src/sim/curriculum.js`
- `src/storage/checkpoints.js`
- `src/utils/prng.js`
- `src/app/runtimeDiagnostics.js`
- all visualization renderer modules

Only release identity, UI/docs/tests, styling, and the new read-only evaluation module were added/changed.

## Verification

Final verification results are appended after packaging.

## Final verification results

- Full Node test suite: **96/96 passed**.
- Static `dist/` rebuild completed successfully.
- JavaScript syntax check passed for every source/script/test module.
- Protected learning/simulation/storage/runtime files and all visualization renderer modules are byte-for-byte identical to parent v0.1.4.1.
- Deterministic parent/candidate continuation parity: same seed, 8 environments and **3,456 training steps** with ordinary validation disabled produced exactly identical policy, optimizer, curiosity, curriculum, environment, recurrent hidden-state, action-RNG, rehearsal and episode state.
- Diagnostic contract tests verify that rear food remains encoded in the existing policy observation and that BRAKE applies no special angular braking beyond the common angular drag.
- Orientation-audit tests verify fixed-seed determinism, bounded metrics and exact session/model immutability before vs. after audit execution.
- Save schema remains **10**; existing schema-9 → schema-10 migration and schema-10 roundtrip tests continue to pass.
