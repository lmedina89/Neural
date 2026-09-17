# MicroMind v0.1.4.1.3 — Build Report

**Version:** `0.1.4.1.3`  
**Build marker:** `OCCCHAR-01413`  
**Parent baseline:** v0.1.4.1.2 `OCCSPIN-01412`

## Objective

Characterize the user-observed failure that appears when the target loses a clean/direct route, after v0.1.4.1.2 showed 0% food reach in the LOS-blocked controlled case despite successful OPEN and LOS-clear/path-blocked cases.

This build remains read-only. It broadens the previous continuous-spin detector into a rotation-trap detector and expands the controlled audit into multiple symmetric/asymmetric detour geometries before any learning change is attempted.

## Implementation

1. Kept exact wall/food LOS and agent-radius direct-path diagnostics outside the neural observation.
2. Replaced the narrow continuous-spin event threshold with `live-occlusion-rotation-trap:v2`.
3. Rotation traps can now be detected by either large cumulative turning with poor food progress or smaller oscillatory loops with repeated turn reversals / food-bearing crossings.
4. Live events now report cumulative/net turning, reversals, bearing crossings, turn/thrust/brake rates, away-from-food thrust, blocked-path/LOS rates, context classification, and recovery.
5. Increased the bounded OBSERVE history window so turn–hesitate–reverse loops can be characterized without instrumenting the LEARN hot loop.
6. Replaced the 3-case controlled check with `occlusion-failure-characterization:v2`: OPEN, CLEARANCE, NARROW CENTER, WIDE CENTER, LEFT-HEAVY, RIGHT-HEAVY, and LONG DETOUR.
7. Controlled trials remain paired by start pose, food position, zero recurrent state, and stochastic-action RNG stream.
8. Added path-clear rate/time, successful-detour rate, temporary retreat, lateral excursion, cumulative/net turns, turn reversals, food-bearing crossings, action mix, wall hits, and food reach.
9. Learner and Balanced Champion audits retain serialize-before/after equality guards.
10. All diagnostic state remains session-only; save schema remains **10**.

## Learning safety

Compared with v0.1.4.1.2, the following learning/simulation/storage/runtime files remain byte-for-byte unchanged:

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

Changed runtime code is limited to release identity, the read-only occlusion/rotation diagnostic module, Research/LIVE diagnostic presentation, and tests/docs.

## Verification

- Full Node test suite: **100/100 passed**.
- Static `dist/` rebuild completed successfully.
- Release identity test confirms `0.1.4.1.3 / OCCCHAR-01413`.
- Controlled failure-characterization audit is deterministic, paired, and model-read-only.
- Live rotation-trap test verifies blocked wall/food conflict capture without policy feedback.
- Policy observation size and existing food X/Y/distance behavior remain unchanged.
- Deterministic parent/candidate continuation over **3,456 training steps** produced equal policy weights, PPO optimizer state, curiosity state, curriculum, RNG/cursor state, episodes, learner experience, rehearsal history, and other learning state. Only wall-clock throughput/profile fields differed, as expected from separate process timing.
- Save schema remains **10**.

The final ZIP is accompanied by a `.sha256` verification file.
