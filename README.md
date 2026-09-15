# MicroMind v0.1.4.0 — Cognitive Observatory

**Build:** `COGOBS-0140`  
**Parent:** v0.1.3.4 `PREDATTN-0134`  
**Save schema:** 9 (unchanged)

MicroMind v0.1.4.0 is a visualization/navigation milestone. It does **not** change how the agent learns. The policy, PPO implementation, curiosity learner, curriculum, world physics, evaluator, checkpoint system, Champion/Hall behavior, and PRNG are preserved from v0.1.3.4.

## What changed

### Five-view Cognitive Observatory
The long single-page laboratory is split into compact views:

- **LIVE** — World, Cognitive Flow brain, training metrics, decisions.
- **PREDICT** — Prediction Echo/Attention stage plus curiosity predictor diagnostics.
- **MEMORY** — real recurrent-state Memory Constellation.
- **HISTORY** — Learning Timeline plus Brain Lineage map.
- **RESEARCH** — PPO stability, retention, evaluation, save/research details.

Only the active heavy canvas view is rendered. This keeps the page shorter and avoids animating hidden canvases on mobile.

### Memory Constellation
The active policy's real 24-value recurrent hidden state is sampled into a bounded **320-point runtime ring buffer**. A fixed deterministic projection maps those states into a 2D state-space view; similar internal states tend to occupy nearby regions without training another model or consuming RNG state.

The constellation shows:

- recent recurrent-state trajectory,
- activation/activity intensity,
- action-colored state points,
- current-state halo,
- real novelty/reward/danger events,
- **Experience Ripples** emitted by those events.

The runtime constellation intentionally is **not added to checkpoint schema 9**. It is observational telemetry for the current browser session and resets on reload or lineage replacement. That keeps existing saves byte-compatible and prevents visualization history from bloating checkpoints.

### Learning Timeline + Brain Lineage
The History view reconstructs the agent's training story from data already saved by MicroMind:

- validation history,
- current balanced score,
- Champion archive positions,
- Hall-of-Fame entries,
- learner lineage history,
- frozen branches / A/B descendants.

Tapping a timeline or lineage node shows its step, score (when available), and lineage information. The renderer is read-only.

## Safety / scientific constraints

v0.1.4.0 does **not** modify:

- 1,040-parameter recurrent actor-critic architecture,
- 441-parameter curiosity predictor architecture,
- PPO clip / learning rate / gradient guard,
- curiosity reward scale or 0.25 episode budget,
- continual rehearsal mix,
- validation protocol or Champion promotion rules,
- save schema (still 9),
- autonomous learner weights through any visualization action.

The AI/simulation/evaluation/storage source files are byte-for-byte identical to v0.1.3.4. Only release identity plus app/visualization/UI files changed.

## QA

- `npm test`: **77/77 pass**
- `npm run check`: pass
- `npm run benchmark`: pass
- `npm run performance`: pass
- exact deterministic continuation parity against v0.1.3.4 after 3,840 training steps: **policy, optimizer, curiosity predictor, curriculum, rehearsal state all identical**
- protected AI/sim/evaluation/storage/PRNG source hash parity against v0.1.3.4: **all exact**
- JS syntax check: pass
- root/dist key-file parity: pass

## Recommended physical iPhone test

1. Deploy the repo-root ZIP.
2. Verify `v0.1.4.0 • COGOBS-0140`.
3. **Load the real Manual Save** before training if a disposable visual-test brain is currently active.
4. Verify learner lineage, Champion, Hall, and step count.
5. Tap through LIVE → PREDICT → MEMORY → HISTORY → RESEARCH.
6. Let LEARN run with MEMORY visible for 30–60 seconds and confirm the constellation fills, trails move, and occasional real event ripples appear.
7. Open HISTORY and confirm validation/Champion/branch nodes reflect the loaded save.
8. Return to LIVE and compare training throughput with heavy views hidden.

The next scientific milestone remains validation-confidence work; v0.1.4.0 deliberately does not attempt to fix the learning instability yet.
