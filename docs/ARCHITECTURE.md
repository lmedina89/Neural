# Architecture — v0.1.1

## Runtime loop

Eight procedural training worlds feed a shared recurrent actor-critic. Each training rollout gathers on-policy transitions, computes GAE, performs guarded PPO updates, records metrics, captures historical milestone brains, and runs fixed validation when scheduled.

Rendering, training and observation are decoupled. Only one world is rendered; training worlds remain headless.

## Model

The model remains unchanged from the accepted foundation:

- 10 numerical observations
- 24 recurrent tanh state units
- 7-action policy head
- scalar value head
- 1,040 trainable parameters

The recurrent hidden state is real and feeds the next timestep. v0.1.1 still uses the compact stop-gradient-through-time update baseline rather than full sequence BPTT; that limitation is intentional and remains a candidate for a later learning-system milestone.

## PPO stability controller

`src/ai/ppo.js` owns optimizer-level protections:

- normalized advantages;
- clipped PPO objective (`clip=0.12`);
- gradient-norm clipping;
- Adam with state persisted in checkpoints;
- adaptive learning rate constrained to 5e-5…2e-4;
- target KL 0.008;
- hard KL 0.025;
- early epoch termination above target KL;
- exact whole-update rollback above hard KL;
- entropy schedule and adaptive entropy rescue;
- post-recovery learning-rate cooldown.

A hard-KL rejection restores both model parameters and optimizer moment state, so a rejected candidate does not quietly leave optimizer damage behind.

## Fixed retention protocol

`src/evaluation/evaluator.js` evaluates all four curriculum stages using a fixed `validation:v2` seed domain. The protocol no longer changes with the current curriculum stage.

Each stage reports return, food, survival, energy, collision data and a bounded `skillScore`. The average stage skill score is the `balanced` score used by the primary protection guard.

Separate category metrics are also exposed for Overall return, Forager, Survivor and Efficiency archives.

## Catastrophic forgetting

`TrainingSession` stores the strongest validated skill score observed for each stage. If a skill that previously exceeded the competence floor falls by more than the configured forgetting tolerance, the validation record reports catastrophic forgetting.

Curriculum promotion is permitted only when prior/current validated stages remain above the promotion skill floor and no earlier-stage forgetting alarm is active.

## Protected archive

v0.1.1 stores five independent candidate brains. Each archive record contains model parameters, optimizer state, curriculum state, validation results, source and training age.

`bestBrain` remains an alias for Best Balanced for backwards-friendly UI behavior.

## Validation recovery

If validation reports a severe balanced regression or catastrophic forgetting, the training policy is restored from Best Balanced without rewinding `totalSteps`. The current curriculum stage is retained. The restored optimizer receives a reduced learning rate and a recovery cooldown. A 25k-step follow-up validation window is armed.

This event is recorded in `rollbackHistory` with `automatic: true`.

## Save schema

Schema 3 adds:

- `bestArchive`
- `lastSkillValidation`
- `skillBestScores`
- `retentionStatus`
- pending legacy-best migration state
- guarded optimizer learning-rate/cooldown state

Schemas 1 and 2 remain readable.
