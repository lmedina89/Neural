# Architecture — v0.1.2

The model remains a small recurrent actor-critic: 10 observations → 24 recurrent units → 7-action policy + value head.

Training is now split conceptually into:

`procedural experience → autonomous Learner → validation laboratory → frozen Champions`

The Learner always continues unless the optimizer detects a numerical failure. Validation is read-only with respect to Learner weights. Champions are immutable snapshots used for comparison/inspection.

## Continual rehearsal
When an environment resets, its skill stage is sampled deterministically from a stage-dependent rehearsal distribution. Future stages are never sampled. Curriculum progression is based only on episodes from the current stage; retention metrics do not block progression.

## Champion promotion
A first validated policy seeds empty Champion slots. Thereafter, a Learner must outperform the current Champion by the configured margin on two consecutive validation events. Only then is the current Learner snapshot promoted.

## Data separation
- training: `train:*`
- validation: `validation:v3`
- historical comparison: `heldout:compare:v2`
- final diagnostic: `heldout:final:v2`
