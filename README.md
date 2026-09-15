# MicroMind v0.1.4.0.1 — Mobile Visualization Performance Hotfix

**Build:** `VISPERF-01401`  
**Parent:** v0.1.4.0 `COGOBS-0140`

This is a rendering-performance hotfix for the Cognitive Observatory. It does **not** change how MicroMind learns or what the existing visualizations look like while they are on screen.

## What changed

- Heavy canvases now use viewport visibility tracking. When a canvas is scrolled fully off-screen, it keeps its last rendered frame but stops consuming repeated draw work until it comes back into view.
- LIVE world and Cognitive Flow are gated independently, so scrolling to the Training card no longer keeps both large canvases repainting in the background.
- Predict World, curiosity network, Memory Constellation, History, and the return chart receive the same off-screen sleep behavior.
- Cognitive Flow caches its stable node/curve geometry and reuses edge objects. Each visible frame still refreshes the **real current weights and activations**, sorts them the same way, and draws the same visual paths/effects.
- Expensive cognitive-FX preparation is skipped entirely when the active visual canvases are off-screen.

## Deliberately unchanged

- policy architecture and parameters
- PPO implementation and hyperparameters
- curiosity predictor/reward behavior
- world physics and rewards
- curriculum/rehearsal
- validation and Champion/Hall rules
- save schema (`9`)
- render-rate settings and visual effect settings
- Cognitive Observatory layout, colors, glow, connection count, halo, attention field, prediction echo, Memory Constellation and History appearance

## iPhone test

1. Load the real Manual Save and verify lineage / Champion / step count.
2. In LIVE, leave World + Cognitive Flow visible and note FPS/throughput. Visual appearance should match v0.1.4.0.
3. Scroll down until both large canvases are fully off-screen while remaining in LIVE. Training throughput should recover because their drawing and cognitive-FX preparation are asleep.
4. Scroll back up. Rendering should resume automatically with no button press and no visual downgrade.
5. Repeat with PREDICT / MEMORY / HISTORY. Only the visible active canvas should perform heavy drawing.

The scientific roadmap remains v0.1.4.1 Validation Confidence & True Regression Audit after this hotfix.
