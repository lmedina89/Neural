# MicroMind v0.1.3.2 — Build Report

**Milestone:** Continual Learning Stability Observatory + Compact Mobile Research UI  
**Build marker:** `STABOBS-0132`  
**Baseline:** exact packaged v0.1.3.1 `CURAUD-0131`  
**Baseline archive SHA-256:** `9418d2dcbe09e7e8f4e0ac10cd3b0b4ce319c1febbf383745e501fd496390dcd`  
**Save schema:** 9 (loads schemas 1–9)

## Why this release exists

The physical v0.1.3.1 matched curiosity ablation completed both 2M-step descendants. The Control branch (predictor active, intrinsic reward influence zero) and Curiosity branch (existing bounded curiosity reward) both exhibited large intermediate validation swings and ended the final paired checkpoint with no material difference. Curiosity reward magnitude was small and its budget was not being exhausted in normal episodes.

That evidence does not prove curiosity is beneficial, but it makes it a poor explanation for the learner's broader volatility. The next milestone therefore instruments the existing PPO learner instead of retuning it.

## Learning behavior intentionally unchanged

No learning-control parameter was altered:

- policy architecture: unchanged;
- curiosity predictor architecture: unchanged;
- PPO learning rate bounds, clip, KL guards, entropy schedule, epochs, Adam settings, gradient clipping threshold: unchanged;
- external and intrinsic reward coefficients: unchanged;
- curiosity cap/budget: unchanged;
- rehearsal distribution and curriculum logic: unchanged;
- Champion confirmation and validation logic: unchanged;
- evaluation domains: unchanged.

`src/ai/ppo.js` changes only expose measurements of the already-computed update. A deterministic development parity run against v0.1.3.1 confirmed identical policy weights, optimizer numeric state (excluding expanded `lastStats` diagnostics), curiosity predictor state, curriculum state, and step count after the same 3,840-step sequence.

## Stability telemetry

Added per-update diagnostics:

- policy/value losses;
- rollout critic explained variance;
- approximate and max epoch KL;
- PPO clip fraction;
- advantage mean/std;
- gradient norm mean/max;
- gradient-clipped minibatch fraction;
- policy parameter L2 delta;
- relative parameter delta;
- maximum absolute parameter delta;
- rejected hard-KL update count.

The session downsamples telemetry every 50k global steps and keeps up to 240 captures (roughly 12M steps of history at the configured interval).

## Regression-event capture

Ordinary validations compare against the previous validation on the same active lineage. A diagnostic event is emitted for a >=10-point balanced drop or >=15-point individual-skill drop.

Each event preserves:

- balanced delta;
- every per-skill delta;
- largest skill drop;
- current PPO diagnostic snapshot;
- compact preceding telemetry-window extrema;
- rehearsal mix/curriculum;
- curiosity reward-influence mode.

Events are observational only and cannot mutate the learner or archives.

## Save / branch persistence

Schema 9 adds:

- active-lineage stability capture history;
- regression-event history;
- next telemetry capture point.

Frozen learners also carry their own stability history/event stream. Schema-8 saves migrate with empty stability telemetry and otherwise preserve their exact v0.1.3.1 state.

## Compact iPhone UI

To stop the research page growing indefinitely, secondary panels are collapsed by default and show compact live summaries in their headers. Opening a panel reveals the full existing content.

The iPhone viewport regression is also addressed by:

- 16px minimum mobile `select/input/textarea` font sizing to prevent Safari focus zoom;
- strict panel/select width containment;
- horizontal overflow prevention at phone widths;
- no curiosity-canvas redraw while the Curiosity panel is collapsed.

## QA

- `npm test`: **69/69 PASS**.
- PPO diagnostic values finite/bounded: PASS.
- Schema-9 telemetry roundtrip: PASS.
- Schema-8 -> schema-9 migration: PASS.
- Regression recording cannot mutate policy weights: PASS.
- Compact panel/mobile-form safeguards: PASS.
- Existing curiosity A/B, Hall, branching, Champion, continual-learning, evaluation isolation, and final-heldout tests retained.
- Deterministic learning-state parity vs v0.1.3.1: PASS.
- Built `dist/` HTTP resource smoke: PASS.
- Development performance smoke: ~42.5k wall-clock steps/sec; latest profiled simulation ~146.3k steps/sec; PPO ~3.48 ms; curiosity update ~0.19 ms. These are development-machine numbers, not iPhone claims.

## Physical test procedure

1. Before deployment, keep a current Manual Save of the desired post-audit learner.
2. Deploy and verify `v0.1.3.2 • STABOBS-0132`.
3. Load the intended save; schema 8 should migrate to schema 9 and pause for verification.
4. Confirm the same active lineage, Champion/Hall entries, frozen A/B Control branch, step count, and curiosity reward mode.
5. Expand **PPO STABILITY** once and confirm values update while learning.
6. Resume normal Learn. Do not fork or retune anything for this observation run.
7. Let the same learner continue approximately **2–3 million steps**.
8. If validation falls sharply, open PPO Stability and capture the latest validation delta + regression event. That is the evidence used to choose the next targeted stability experiment.
