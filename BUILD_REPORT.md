# MicroMind v0.1.3.3 — Build Report

**Milestone:** Neural Flow & Cognitive FX  
**Build marker:** `NEURAFX-0133`  
**Baseline:** exact packaged v0.1.3.2 `STABOBS-0132`  
**Baseline archive SHA-256:** `5f9e6eaea59027b87a54bec10fb6c75778c4b1374621e88b7455c0362a34a745`  
**Save schema:** 9 (unchanged; loads schemas 1–9)

## Purpose

The project started as a learning laboratory, but the primary experience is also supposed to make the AI itself visible. v0.1.3.3 therefore upgrades the live visualization without retuning the learner.

The design rule is strict: the new effects are driven by real model state and existing telemetry. They do not invent a second decorative “brain” that can disagree with the actual policy.

## Cognitive Flow mode

The Live Brain selector now defaults to **Cognitive Flow** while retaining Active, Weights, and Strongest modes.

Cognitive Flow uses the real policy forward pass to render:

- sensory input activations;
- sensory-to-hidden influence through the actual `wx` weights;
- recurrent memory links through the actual `wh` matrix and previous hidden state;
- hidden-to-policy influence through the actual `wp` weights;
- hidden-to-value influence through the actual `wv` weights;
- policy probabilities and value estimate;
- moving light pulses along the strongest currently active pathways;
- a decision beam from the highest-probability policy output;
- brighter node glow for strongly active/influential units.

The hidden layer is arranged as a compact “cognitive core” only for presentation. Node placement does not change the network itself.

## Signature light effect: Cognitive Halo

A multi-ring **Cognitive Halo** surrounds the recurrent core. Its motion and intensity are data-driven:

- decision-confidence gap changes halo emphasis;
- curiosity novelty contributes magenta activity;
- current external reward contributes warm/gold activity;
- real novelty/reward events can generate small star-like sparks.

This is a read-only visual mapping of existing signals, not an additional reward or learning mechanism.

## World-linked cognition

The world view now connects the visible brain to what the agent is actually sensing and choosing:

- danger-ray brightness is modulated by real normalized sensory influence;
- the nearest food can receive a salience ring/path when food sensors are influential;
- the agent receives a confidence/novelty halo;
- the dominant policy action receives a short directional decision-light cue.

The underlying world, observations, rewards, physics, actions, and agent behavior are unchanged.

## Curiosity / prediction visualization

The existing 441-parameter learned forward model now has a more literal visual explanation:

- real state/action → predictor → next-sensory pathways carry moving activity pulses;
- output labels explicitly show **PREDICTED → ACTUAL**;
- each sensory output has a ghost marker displaced according to real prediction mismatch;
- prediction-error rings/halo remain tied to actual predictor error and novelty.

Curiosity reward math, predictor training, evaluation isolation, budget, and coefficient are unchanged.

## Learning invariants

The following v0.1.3.2 core files are byte-for-byte identical in v0.1.3.3:

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

Only release identity, UI/app read-only telemetry plumbing, visualization renderers, CSS, tests, and documentation changed.

Therefore this release does **not** change:

- the 1,040-parameter recurrent policy;
- the 441-parameter curiosity predictor;
- PPO learning/optimizer rules;
- rewards or curiosity strength;
- curriculum/rehearsal distribution;
- observations/actions/world physics;
- Champion/Hall/branch behavior;
- stability-observatory semantics;
- evaluation protocols;
- save schema.

## Mobile/performance strategy

The richer visualization remains throttled by the existing render schedule rather than running on every simulation step. Cognitive Flow limits the number of drawn edges and moving pulses, caps DPR at 2, and keeps the long research panels collapsible. The curiosity canvas still avoids redraw while its panel is collapsed.

No attempt was made to increase training render frequency.

## QA recovered and completed

The interrupted first pass was recovered before release. The incomplete draft had a stale root HTML version tag/missing Cognitive Flow selector and stale v0.1.3.2 documentation; those were corrected before final packaging.

Final checks:

- `npm test`: **71/71 PASS**.
- `npm run check`: PASS, including fresh static `dist/` build.
- development performance smoke: ~39.6k wall-clock steps/sec; latest profiled simulation ~133.8k steps/sec; PPO ~4.03 ms; curiosity update ~0.18 ms. These are development-machine measurements, not iPhone guarantees.
- v0.1.3.2 learning-critical files: byte-for-byte parity PASS.
- Cognitive Flow static wiring checks for recurrent memory, value paths, pulses, halo, world influence and prediction ghosts: PASS.
- schema remains 9: PASS.

## Physical acceptance test

1. Keep the current v0.1.3.2 Manual Save before deployment.
2. Deploy and verify `v0.1.3.3 • NEURAFX-0133`.
3. Load the intended learner and verify the same lineage, global/branch steps, Champion, Hall, frozen branches, curiosity mode, and stability history.
4. Keep training paused initially and set **Live Brain → Cognitive Flow** (it is the new default).
5. In Probe or Observe, watch the network while moving food/hazards or while the agent acts. Pulses, input salience and the decision beam should change with the real policy.
6. Open Curiosity / Prediction briefly during Learn. Prediction ghosts and error effects should track the displayed predicted-versus-actual values.
7. Resume the same stability-observation run. v0.1.3.3 is visual-only and should not invalidate the v0.1.3.2 PPO evidence already being collected.
8. On iPhone, watch browser FPS/heat for several minutes. If the richer brain view is too expensive, switch the Live Brain selector back to Active/Strongest without affecting training.
