# Architecture — v0.1.3.2

## Policy learner

`10 observations → 24 recurrent units → 7-action policy + value head`

The policy has **1,040 learned parameters** and remains on the same recurrent PPO implementation.

## Curiosity predictor

`10 observations + 7-way one-hot action → 16 tanh units → 9 predicted dynamic next-observation values`

The auxiliary forward model has **441 learned parameters**. It predicts the next sensory state and learns from real transitions. It does not choose actions directly and is never used to score evaluation performance.

## Reward influence modes and preserved A/B audit

Curiosity still supports `reward` and `observe` modes. The v0.1.3.1 matched CONTROL/CURIOSITY audit remains fully available and its results persist; v0.1.3.2 does not retune it.

## Stability observatory

The PPO trainer exposes diagnostic values from the same update it already performs: losses, KL, clip fraction, advantage distribution, rollout critic explained variance, gradient norms/clipping, and parameter movement. Session-level telemetry is downsampled every 50k global steps.

Ordinary validation can emit a compact regression event when a balanced score falls >=10 points or an individual skill falls >=15 points versus the previous validation on the same lineage. Events include the preceding stability-window summary and are strictly observational.

## Data separation

- training: `train:*`
- Champion validation: `validation:v3` — curiosity reward OFF
- historical comparison: `heldout:compare:v2` — curiosity reward OFF
- curiosity A/B audit: `heldout:curiosity-ablation:v1` — curiosity reward OFF
- final diagnostic: `heldout:final:v2` — curiosity reward OFF

## Persistence

Schema 9 stores the prior schema-8 curiosity/A/B state plus active-lineage stability history, regression events, and the next capture point. Frozen learner branches keep their own telemetry. Schemas 1–9 load; schema 8 migrates with empty stability history and otherwise preserves its exact state.
