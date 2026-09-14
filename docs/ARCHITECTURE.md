# Architecture — v0.1.1.1

## Runtime loop

Eight procedural training worlds feed one shared recurrent actor-critic. Each rollout gathers on-policy transitions, computes GAE, performs guarded PPO updates, records metrics, captures historical milestone brains, and runs validation when scheduled.

Rendering, training and observation remain decoupled. Only one world is visually rendered; training worlds stay headless.

## Model

The model is intentionally unchanged:

- 10 numerical observations
- 24 recurrent tanh state units
- 7-action policy head
- scalar value head
- 1,040 trainable parameters

The recurrent hidden state is real and feeds the next timestep. Recurrence remains stop-gradient through time rather than sequence BPTT/GRU training.

## PPO stability controller

`src/ai/ppo.js` is unchanged from v0.1.1 and retains:

- normalized advantages;
- clipped PPO objective (`clip=0.12`);
- gradient-norm clipping;
- persisted Adam state;
- adaptive LR bounded to 5e-5…2e-4;
- target KL 0.008;
- hard KL 0.025;
- early epoch termination;
- exact whole-update rollback on hard-KL violation;
- entropy schedule/rescue;
- post-recovery LR cooldown.

This milestone calibrates model-selection evidence rather than changing optimizer mechanics.

## Validation:v3

`src/evaluation/evaluator.js` evaluates every curriculum stage on the fixed `validation:v3` domain with 12 seeded stochastic episodes per skill.

Each episode produces a bounded competence sample from:

- resource acquisition;
- survival;
- remaining energy.

Stage score = mean episode competence.

A standard-error-based 95% range is reported as `skillCiLow` / `skillCiHigh`. The balanced score is the equal-episode mean across all stages and also gets a confidence range.

Archive category metrics remain:

- Balanced
- Overall return
- Forager
- Survivor
- Efficiency

## Retention evidence

`TrainingSession` stores one calibrated best record per skill:

- best score;
- confidence bounds;
- step age.

A skill WATCH requires:

1. historical competence above the floor;
2. a configured absolute drop;
3. confidence separation or a catastrophic-size drop.

The evidence must repeat before it becomes CONFIRMED. Confirmation streaks reset when the evidence disappears.

Curriculum promotion is held while an earlier/current required skill is under a retention watch.

## Automatic recovery threshold

Automatic policy restoration is deliberately more conservative than v0.1.1.

A one-off or single-skill specialization does not restore an old brain. Automatic Best-Balanced recovery requires:

- repeat-confirmed balanced regression, or
- at least two confirmed catastrophic skill regressions.

This is designed to avoid destroying a genuinely better generalizing policy just because one narrow validator moved sharply once.

## Archive protocol migration

Schema-3 v0.1.1 archives contain validation:v2 scores. On load they remain inspectable but are marked for recalibration.

Before Latest retention is assessed, every unique preserved archive model is re-evaluated under validation:v3. The active archive is then rebuilt from only v3-compatible candidates.

This ordering also fixes the v0.1.1 first-migration `retention OK` blind spot.

## Generalization domains

Three non-training domains now have separate purposes:

- `validation:v3` — automatic model selection / retention protection;
- `heldout:compare:v2` — historical comparison UI;
- `heldout:final:v2` — final Unseen Test diagnostic.

Final holdout results never call archive-selection, validation-history, curriculum, optimizer, or rollback mutators.

Unseen Test also evaluates the exact current Latest weights on validation:v3 read-only, so a validation/holdout disagreement compares the same model rather than two different training ages.

## Save schema

Schema 4 adds:

- `skillBestRecords`
- `skillRegressionStreaks`
- `balancedRegressionStreak`
- richer `retentionStatus`
- `pendingArchiveMigration`
- `archiveNeedsRebaseline`
- `legacyValidationHistory` for preserved pre-v3 records

Schemas 1–3 remain readable.
