# MicroMind v0.1.4.0.4 — Decision HUD Refresh Hotfix

**Build:** `DECHUD-01404`  
**Parent:** v0.1.4.0.3 `RUNTIME-01403`

This is a narrow UI refresh hotfix. The v0.1.4.0.1 off-screen rendering optimization correctly lets expensive World/Cognitive Flow canvases sleep when they leave the viewport, but the Decision card's action probabilities/value/energy/reward text was still being refreshed from the same visual-render branch. On iPhone this could leave the Decision card frozen at an old snapshot while training continued normally.

## What changed

- The lightweight Decision HUD now refreshes from the normal UI update cadence.
- In LEARN mode that cadence remains `5 Hz` (`CONFIG.runtime.learnUiHz`).
- World, Cognitive Flow, Predict, Memory and History canvases still sleep independently when off-screen.
- The Decision refresh does **not** wake any sleeping canvas.
- v0.1.4.0.3's 5-second / 30-second wall-clock train-rate diagnostics remain intact.

## What did not change

No learning/simulation behavior was changed. Policy, PPO, rollout/session, curiosity, world/spawn clearance, curriculum, evaluator, save/checkpoint logic, PRNG, visual renderers, styles, rewards, sensors, Champion/Hall behavior and save schema are unchanged from the parent build. Save schema remains **9**.

## iPhone acceptance test

1. Deploy and confirm `v0.1.4.0.4 • DECHUD-01404`.
2. Load the intended Manual Save and verify lineage/Champion/step count before training.
3. Run LEARN with LIVE visible; confirm Decision probabilities/value move.
4. Scroll World/Cognitive Flow completely off-screen while leaving the Decision card visible.
5. Wait a few seconds. Decision probabilities/value/energy should continue updating while VISUALS reports `LIVE OFFSCREEN-IDLE` when applicable.
6. Scroll back to LIVE; the canvases should wake normally and look unchanged.
7. Compare 5s/30s train rates only after the rolling meter is ready.
