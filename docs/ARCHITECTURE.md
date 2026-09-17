# Architecture — v0.1.4.1.1

## Policy learner

`10 observations → 24 recurrent units → 7-action policy + value head`

The policy has **1,040 learned parameters** and remains on the same recurrent PPO implementation.

## Curiosity predictor

`10 observations + 7-way one-hot action → 16 tanh units → 9 predicted dynamic next-observation values`

The auxiliary forward model has **441 learned parameters**. It predicts the next sensory state and learns from real transitions. It does not choose actions directly and is never used to score evaluation performance.

## Validation confidence layer

Routine Champion/retention validation still uses `validation:v3` with 12 seeded-stochastic episodes per skill. Each validation checkpoint now stores the exact active policy as the next same-lineage comparison reference.

If balanced performance drops >=10 points or an individual skill drops >=15 points versus the previous checkpoint, `validation:confidence:v1` runs a larger paired replay. Current and reference policies receive identical fixed environment seeds and action-randomness streams. Paired per-episode skill-score differences are summarized with a 95% confidence interval. `CONFIRMED REGRESSION` requires both a material negative paired mean and an upper confidence bound below zero. Otherwise the suspicious raw event is `LIKELY NOISE` (or `MEASURED DROP` when no compatible reference exists yet).

This layer is observational. It never restores weights, changes rewards, alters curriculum, or promotes a Champion.


## Rear-target / rotation audit

`orientation-audit:v1` is a manual, read-only diagnostic domain. It creates temporary empty-arena `World` instances and places one food target at fixed bearings around an agent that starts from rest. It uses the normal 10-value observation, normal stochastic policy sampling, recurrent hidden-state updates, and normal angular dynamics. No training transition is stored and no policy/optimizer/predictor/session state is updated.

The audit reports facing success, target reach, cumulative angular travel, a conservative spin-incidence threshold, BRAKE selection while angular speed is already substantial, and initial policy preference at each bearing. Because food relative X/Y is always present in the current observation, this audit specifically measures turn-control behavior rather than visual object permanence.

## Stability observatory

PPO diagnostics remain read-only: losses, KL, clip fraction, advantage distribution, critic explained variance, gradient norms/clipping, and parameter movement. Regression events now retain the confidence-audit verdict alongside the preceding PPO window.

## Data separation

- training: `train:*`
- ordinary Champion/retention validation: `validation:v3` — curiosity reward OFF
- paired regression confirmation: `validation:confidence:v1` — curiosity reward OFF
- historical comparison: `heldout:compare:v2` — curiosity reward OFF
- curiosity A/B audit: `heldout:curiosity-ablation:v1` — curiosity reward OFF
- final diagnostic: `heldout:final:v2` — curiosity reward OFF

## Persistence

Schema **10** adds the previous-validation policy reference and bounded validation-confidence history to schema 9. Schemas 1–10 load. Schema-9 migration starts with no invented confirmation evidence; the first same-lineage v0.1.4.1+ validation establishes a fresh paired reference. The rear-target audit adds no persistent fields, so schema remains 10.

## Cognitive Observatory

LIVE, PREDICT, MEMORY, HISTORY and RESEARCH remain read-only visualization surfaces. Off-screen heavy canvases sleep independently. Cognitive Flow, Prediction Echo, Attention Fields, Memory Constellation, Experience Ripples, History/Lineage and runtime-throughput diagnostics do not update policy or optimizer state.
