# Architecture — v0.1.3

## Policy learner
Unchanged from v0.1.2.1:

`10 observations → 24 recurrent units → 7-action policy + value head`

This network has **1,040 learned parameters** and is still trained by the same recurrent PPO implementation.

## Curiosity predictor
New in v0.1.3:

`10 observations + 7-way one-hot action → 16 tanh units → 9 predicted dynamic next-observation values`

The forward model has **441 learned parameters**. It is deliberately auxiliary: it does not choose actions directly and it does not alter evaluation scores.

For a sampled transition:

1. the policy chooses an action;
2. the real world advances;
3. before the predictor learns that transition, prediction error is measured;
4. error is normalized against an EMA of recent prediction error;
5. positive novelty yields a tightly bounded intrinsic reward;
6. PPO receives `external reward + intrinsic reward` for that training transition;
7. the predictor then trains on the real `(state, action, next state)` sample.

Curiosity is sampled every fourth environment transition. This preserves a genuine one-step prediction objective while keeping mobile overhead small.

## Reward containment
External world reward is unchanged and remains separately recorded in `World.info().totalReward`.

Curiosity:
- never changes the programmed external reward components;
- is zero on terminal transitions;
- is capped per sampled step;
- is capped to 0.25 total bonus per episode;
- is disabled in validation/comparison/final-holdout evaluation.

This makes curiosity an exploration pressure, not a replacement objective.

## Continual learning / Champions
Unchanged from v0.1.2.1:

`procedural experience → autonomous Learner → validation → frozen Champions`

Behavioral regression remains observational. No automatic Champion rollback is reintroduced.

Curiosity state is saved with schema-7 Learners and new v0.1.3 Champion/Hall snapshots so a deliberate future branch can restore the predictor knowledge associated with that policy. Older Champions without curiosity state remain valid policy snapshots and simply start a fresh predictor if used as a branch parent.

## Data separation
- training: `train:*` — curiosity ON
- validation / Champion selection: `validation:v3` — curiosity OFF
- historical comparison: `heldout:compare:v2` — curiosity OFF
- final diagnostic: `heldout:final:v2` — curiosity OFF
