# MicroMind v0.1.4.0.1 — Build Report

**Version:** `0.1.4.0.1`  
**Build marker:** `VISPERF-01401`  
**Parent baseline:** v0.1.4.0 `COGOBS-0140`

## Objective

Recover mobile training throughput lost to the richer Cognitive Observatory visuals without reducing visual fidelity or changing learning behavior.

## Implementation

1. Added IntersectionObserver-backed viewport visibility tracking for all expensive observatory canvases.
2. LIVE World and Cognitive Flow now sleep independently when scrolled off-screen.
3. Predict World, Curiosity, Memory, History, and the return chart also sleep when off-screen.
4. Cognitive-FX computation is deferred until a visible canvas actually needs it.
5. Cognitive Flow now reuses stable node/curve geometry and edge objects, while refreshing real weights/activations every visible frame.
6. No render-rate, connection-count, glow, canvas-resolution, or visual-style reductions were made.

## Learning safety

The learning modules are intended to remain identical to v0.1.4.0. Save schema remains 9. Final parity and test results are recorded below after verification.

## Verification

- Unit/integration suite: **79/79 passed**.
- Deterministic learning parity: v0.1.4.0 and v0.1.4.0.1 were trained from the same seed for **3,840 steps**; policy weights, PPO optimizer state, curiosity predictor, curriculum state, rehearsal history, global steps/episodes and learner experience were exactly equal.
- Protected learning modules are byte-for-byte identical to v0.1.4.0: `model.js`, `ppo.js`, `rollout.js`, `session.js`, `curiosity.js`, `world.js`, `curriculum.js`, `evaluator.js`, `checkpoints.js`, and `prng.js`.
- `CONFIG` learning/runtime body is unchanged after the version/build identity lines.
- `styles.css` is byte-for-byte identical to v0.1.4.0, so the hotfix introduces no CSS/visual-style downgrade.
- Build completed successfully.
- Development performance smoke remained healthy; latest core profile reported about **73.6k simulation steps/sec** for the simulation phase and **25.9k training steps/sec** inside the Node performance harness. This harness does not render Safari canvases; the real-device win being targeted is specifically removal of off-screen rendering work.

## Expected physical-device behavior

The largest improvement should occur when LIVE remains selected but the World / Cognitive Flow canvases have been scrolled completely out of view (for example, while reading Training, PPO Stability, or lower cards). Those canvases now stop repainting and cognitive-FX preparation is skipped until they re-enter the viewport. When visible, the same render-rate settings, connection counts, glow effects, Cognitive Halo, Attention/Echo effects and canvas resolution remain in use.
