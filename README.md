# MicroMind v0.1.3 — Intrinsic Curiosity Foundation

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in deterministic procedural worlds while its activations, decisions, memory, rewards, rehearsal, Champions, and generalization are inspectable.

v0.1.3 adds a separate **441-parameter learned forward-prediction model** that creates a small, bounded intrinsic curiosity signal during training.

**Build:** `CURIOUS-013`

## What changed in v0.1.3
The accepted v0.1.2.1 policy learner remains the control:

- policy architecture remains **1,040 parameters**;
- PPO implementation and hyperparameters are unchanged;
- observations/actions are unchanged;
- external reward coefficients are unchanged;
- physics/world generation are unchanged;
- curriculum and rehearsal targets are unchanged;
- Champion selection and every evaluation protocol are unchanged.

New capability:

`current observation + chosen action → curiosity predictor → predicted next observation`

The predictor learns the nine dynamic sensory values (the policy's tenth observation is a constant bias feature). Prediction error becomes **novelty**. Novel nonterminal transitions can add a tiny intrinsic bonus to PPO's training reward.

Safety/containment:

- curiosity is sampled every fourth transition to protect mobile throughput;
- intrinsic reward is positive-only and tightly capped per step;
- each episode has an absolute curiosity budget of **0.25 reward**;
- terminal/death transitions receive **zero intrinsic bonus** even though the predictor still learns them;
- curriculum promotion, training charts, validation, Compare Brains, and Unseen Test continue to use external task performance;
- **all evaluations run with curiosity reward OFF**.

## Curiosity visualization
The new Curiosity / Prediction screen is real instrumentation:

- live prediction error;
- normalized novelty;
- actual intrinsic reward;
- remaining episode curiosity budget;
- predictor training loss;
- predictor parameter count;
- a live predictor graph showing real state/action inputs, hidden activity, predicted sensory outputs, actual next sensory values, and error rings;
- a subtle world novelty trail driven by real prediction surprise from the current training environment.

As familiar transitions are learned, prediction error should generally fall. Unusual transitions can temporarily become more novel again.

## Save migration
v0.1.3 uses **checkpoint schema 7** and loads schemas 1–7.

Loading a v0.1.2.1/schema-6 checkpoint preserves the policy, optimizer, Champions, lineages, Hall, rehearsal state, and experiment age. Because older versions had no curiosity model, the predictor begins fresh after migration. If the persistent Hall is empty, the existing Balanced Champion is pinned as a **pre-curiosity reference**.

## Run
```bash
npm test
npm run benchmark
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

Then open `http://localhost:8080/`.

GitHub Pages can serve the repository root directly.
