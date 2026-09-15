# MicroMind v0.1.3.2 — Continual Learning Stability Observatory

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in procedural worlds while its activations, decisions, memory, rewards, rehearsal, Champions, curiosity predictor, and generalization are inspectable.

**Build:** `STABOBS-0132`

v0.1.3.2 follows the completed v0.1.3.1 Curiosity A/B audit. The matched Control and Curiosity descendants both showed substantial checkpoint volatility and finished the 2M-step audit with no material final difference. The next scientific question is therefore **why the continual PPO learner sometimes drops and later recovers**, not whether curiosity merely exists.

## Scientific rule for this release

This is an **observatory**, not a learning retune.

The following stay unchanged from v0.1.3.1:

- 1,040-parameter recurrent policy;
- 441-parameter curiosity predictor;
- PPO hyperparameters and optimizer behavior;
- curiosity reward strength and episode cap;
- external rewards, observations, actions, physics, and world generation;
- curriculum/rehearsal targets;
- Champion/Hall promotion logic;
- validation, comparison, and final-heldout seed domains.

The new code only measures the existing update more deeply and records large validation regressions.

## PPO Stability Observatory

Each PPO update now reports real telemetry that was already implicit in the optimizer but was not visible:

- policy loss and value loss;
- entropy and approximate/max KL;
- PPO objective clip fraction;
- raw advantage mean and standard deviation;
- critic explained variance using rollout-time value predictions;
- gradient L2 norm and fraction of minibatches requiring gradient clipping;
- absolute and relative policy-parameter movement per PPO update;
- maximum single-parameter movement;
- rejected hard-KL update count.

Telemetry capture is downsampled every **50,000 global steps** so it remains useful over multi-million-step physical runs without ballooning the save.

## Regression event recorder

At each ordinary validation, MicroMind compares the new result with the previous validation from the same active lineage.

A compact diagnostic event is preserved when either:

- balanced validation drops by at least **10 percentage points**, or
- any individual skill drops by at least **15 percentage points**.

The event records the validation deltas plus the immediately preceding PPO telemetry window, curriculum, rehearsal mix, and curiosity influence mode. This is **observational only**. It never rolls back, changes learning rate, restores a Champion, or modifies the learner.

## Compact phone UI

The research page had become too vertically long on iPhone. v0.1.3.2 converts secondary research areas into collapsible panels with useful one-line summaries:

- Research Status;
- PPO Stability;
- Curiosity / Prediction and its completed A/B audit;
- Skill Retention;
- Evaluation / Checkpoints.

The World, Live Brain, Training, and Decision panels remain immediately visible. The curiosity canvas is not redrawn while its panel is collapsed, reducing unnecessary UI work.

Mobile form controls use a minimum **16 px** font size to avoid iOS Safari's focus-zoom behavior, and research panels are width-contained to prevent horizontal viewport expansion.

## Persistence

v0.1.3.2 uses **checkpoint schema 9** and loads schemas 1–9. Schema 8 v0.1.3.1 saves migrate directly with empty stability history and immediately begin collecting telemetry. No policy, optimizer, Champion, Hall, branch, curiosity, or A/B result is discarded by the migration.

Stability history and regression events are also kept with frozen learner branches so later branch inspection does not mix telemetry from unrelated lineages.

## Verification

The release test suite includes policy/optimizer parity checks at development time. For a fixed seed and identical training sequence, the v0.1.3.2 instrumented build produced byte-identical policy weights, optimizer state (excluding the deliberately expanded diagnostic `lastStats` object), curiosity predictor state, curriculum state, and step count versus v0.1.3.1.

Run locally with:

```bash
npm test
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

GitHub Pages can serve the repository root directly.
