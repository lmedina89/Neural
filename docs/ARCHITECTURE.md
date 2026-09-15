# Architecture — v0.1.3.1

## Policy learner

`10 observations → 24 recurrent units → 7-action policy + value head`

The policy has **1,040 learned parameters** and remains on the same recurrent PPO implementation.

## Curiosity predictor

`10 observations + 7-way one-hot action → 16 tanh units → 9 predicted dynamic next-observation values`

The auxiliary forward model has **441 learned parameters**. It predicts the next sensory state and learns from real transitions. It does not choose actions directly and is never used to score evaluation performance.

## Reward influence modes

For sampled transitions, the predictor computes a bounded potential intrinsic bonus from prediction surprise.

- `reward`: PPO receives `external + intrinsic` reward.
- `observe`: the predictor still scores and learns the transition, but PPO receives `external + 0` intrinsic reward.

Both modes retain the same predictor architecture, sampling stride, normalization, and shadow episode-budget accounting.

## Matched curiosity A/B audit

Starting an audit freezes an exact origin and creates two descendants:

`origin → CONTROL (observe-only)`

`origin → CURIOSITY (reward-on)`

Both inherit identical policy weights, PPO optimizer/RNG, curiosity predictor/optimizer, action RNG, curriculum state, environment seed cursor, and rehearsal history. Automatic curriculum movement is frozen during the audit. At equal branch-local progress both branches receive the same PPO schedule age, preventing global experiment order from altering annealing.

Ordinary milestones, normal validation/Champion promotion, and unrelated branch forks are suspended while the audit is active. Audit evaluations use their own read-only seed domain and cannot mutate policy or Champion state.

## Data separation

- training: `train:*`
- Champion validation: `validation:v3` — curiosity reward OFF
- historical comparison: `heldout:compare:v2` — curiosity reward OFF
- curiosity A/B audit: `heldout:curiosity-ablation:v1` — curiosity reward OFF
- final diagnostic: `heldout:final:v2` — curiosity reward OFF

## Persistence

Schema 8 stores curiosity influence mode, predictor state, audit definition/results, branch role/progress, and branch-specific episode windows. Schemas 1–8 load; schema 7 defaults to reward-on with no active audit.
