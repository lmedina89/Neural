# MicroMind v0.1.3.4 — Build Report

**Milestone:** Prediction Echo & Attention Fields  
**Build marker:** `PREDATTN-0134`  
**Baseline:** exact packaged v0.1.3.3 `NEURAFX-0133`  
**Baseline archive SHA-256:** `91625797de1f148716fc8bae2cea47bc61288b0e951377fb8214a161346ddc3f`  
**Save schema:** 9 (unchanged; loads schemas 1–9)

## Purpose

The v0.1.3.3 Cognitive Flow view made the real recurrent policy visually alive. v0.1.3.4 adds the two other visual systems requested for the simulator itself: a world-linked **Attention Field** and a learned-model **Prediction Echo**.

The design rule remains strict: every effect must be driven by real MicroMind state or real predictor telemetry. No decorative effect is allowed to feed back into learning.

## Attention Field

The running policy's normalized real input influence is projected back into the world:

- food x/y/distance influence controls the nearest-food field intensity and salience line;
- danger L/F/R influence controls each actual sensor ray and contact-region glow independently;
- energy influence controls a subtle pressure ring around the agent;
- dominant action/confidence and novelty continue to drive the existing agent cognitive FX.

This provides a direct visual chain from **world → sensors → influence → decision** without altering the world or agent.

## Prediction Echo / Echo Lens

The 441-parameter curiosity model already predicts nine dynamic sensory features. v0.1.3.4 turns one genuine sampled env-0 transition into an agent-local visual lens inside the World panel:

- gold hollow marks are **predicted next sensory values**;
- cyan filled marks are **actual next sensory values**;
- the line between them is literal prediction mismatch;
- food x/y become a paired local vector;
- danger L/F/R become paired points on the three local ray directions;
- speed, turn rate and energy errors become compact orbit arcs;
- the surrounding pulse is driven by actual prediction error and novelty.

The lens is deliberately sensor-local. Training can advance thousands of environment transitions between browser frames; projecting a sampled historical prediction onto the current physical object positions would imply false precision. The lens therefore shows exactly what the predictor knew: sensory prediction versus sensory outcome.

## Compact control

The World header now includes one compact selector:

- Attention + Echo
- Attention Field
- Prediction Echo
- Clean World

No new long diagnostic section was added. This keeps the page-height work from v0.1.3.2 intact.

## Learning invariants

The learning-critical model/PPO/session/curiosity/world/curriculum/evaluation/storage/PRNG files are byte-for-byte identical to v0.1.3.3.

Therefore v0.1.3.4 does **not** change:

- 1,040-parameter recurrent policy architecture;
- 441-parameter curiosity predictor architecture or training;
- PPO coefficients, optimizer, KL guards or gradient limits;
- external/intrinsic reward math;
- curriculum and rehearsal mix;
- observations, actions or world physics;
- Champion, Hall-of-Fame and branch rules;
- Stability Observatory semantics;
- validation/final-holdout protocols;
- save schema 9.

## QA

- `npm test`: **73/73 PASS**.
- `npm run check`: PASS; static `dist/` rebuilt successfully.
- JavaScript syntax checks for the modified app/renderer: PASS.
- local HTTP resource smoke for config, world renderer and app wiring: PASS.
- v0.1.3.3 learning-critical parity: PASS for all ten protected files.
- development performance smoke: ~44.7k wall-clock steps/sec; latest profiled simulation ~102.3k steps/sec; PPO ~4.58 ms; curiosity update ~0.54 ms. Development-machine measurements only, not iPhone guarantees.

## Physical acceptance test

1. Keep a Manual Save of the real long-running learner before deployment.
2. Deploy and verify `v0.1.3.4 • PREDATTN-0134`.
3. If the page starts with a throwaway fresh brain, **do not save it**; load the intended Manual Save first.
4. Verify lineage, total steps, Champion, Hall, branches, curiosity mode and stability history survived unchanged.
5. In the World header leave **Attention + Echo** selected initially.
6. In Observe/Probe, move food/hazards and confirm the Attention Field changes with the real sensor/policy state.
7. During Learn, the Prediction Echo lens should appear when curiosity has a sampled env-0 transition. Gold is prediction; cyan is actual.
8. Switch among Attention / Echo / Clean World to confirm the overlays are visual-only and can be disabled instantly.
9. Watch iPhone FPS and heat for several minutes. If needed, use Clean World and a simpler Live Brain mode during unattended training without affecting the learner.
