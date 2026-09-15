# MicroMind v0.1.4.0.4 — Build Report

**Version:** `0.1.4.0.4`  
**Build marker:** `DECHUD-01404`  
**Parent baseline:** v0.1.4.0.3 `RUNTIME-01403`

## Objective

Fix the iPhone-visible Decision card freeze introduced by the off-screen visualization performance optimization, without waking hidden canvases or changing learning behavior.

## Root cause

`updateDecision()` was called from the LIVE canvas render branch. v0.1.4.0.1 correctly stopped World/Cognitive Flow rendering while those canvases were off-screen, but that also stopped the Decision card from receiving fresh action probabilities, value, energy and reward-component text. Training itself continued normally.

## Implementation

1. `updateDecision()` now also runs from the normal lightweight `updateUI()` path.
2. LEARN UI cadence remains 5 Hz, so the Decision card stays responsive without restoring expensive 12 Hz off-screen canvas work.
3. Existing on-screen LIVE rendering is unchanged; the canvas render path may still refresh Decision opportunistically while visible.
4. No hidden canvas is awakened by the Decision refresh.
5. v0.1.4.0.3 rolling 5s/30s train-rate and foreground/visual-state diagnostics remain intact.

## Safety / parity

The learning/simulation files are intentionally unchanged from v0.1.4.0.3:

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

All visualization renderer modules and `styles.css` are also unchanged. Save schema remains **9**.

## Verification

Verification is completed before packaging and recorded below in the final report update.

## Final verification results

- Full Node test suite: **87/87 passed**.
- Static `dist/` rebuild completed successfully.
- JavaScript syntax check passed for every source/script/test module.
- Protected learning/simulation files, all visualization renderer modules, `styles.css`, and runtime throughput diagnostics are **byte-for-byte identical** to v0.1.4.0.3; the configuration body is identical after the release identity lines.
- Deterministic parent/candidate continuation parity: same seed and **3,456 training steps** produced exactly equal policy weights, PPO optimizer state, curiosity predictor, curriculum, rehearsal history, action RNG, environment cursor, global steps and Learner experience.
- Development performance harness: **42.3k wall-clock training steps/sec**; last inner simulation phase **156.2k raw simulation steps/sec**; PPO ~**3.6 ms** in the container environment. Browser/iPhone results remain device-dependent.
- New regression contract test verifies the Decision HUD refresh is present in the lightweight UI path while the off-screen canvas sleeping condition remains intact.
