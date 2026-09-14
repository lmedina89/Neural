# Architecture — v0.1.2.1

The learning model is unchanged from v0.1.2:

`10 observations → 24 recurrent units → 7-action policy + value head`

The active learning loop remains:

`procedural experience → autonomous Learner → validation laboratory → frozen Champions`

Behavioral regression is observational. The Learner is not automatically restored from a Champion. Numerical PPO safety can still reject a mathematically destructive update.

## Hall of Fame
A Hall entry is an immutable policy snapshot copied from a validated Champion. It stores model weights, source step/lineage, validation metadata, version/build identity, and (when available) optimizer state for deliberate research branching.

Hall data is persisted twice:
1. a dedicated IndexedDB `hall-of-fame` slot, independent of ordinary Manual/Autosave replacement;
2. schema-6 checkpoint backup.

Exact duplicate policies are fingerprint-deduplicated. There is no automatic Hall replacement.

## Learner branches
Only one learner lineage trains at a time.

- Forking freezes the current Learner state, then starts a new lineage from a selected Champion/Hall policy.
- Switching branches freezes the lineage being left and restores the selected frozen lineage.
- `totalSteps` is global experiment age and never rewinds during a fork/switch.
- `learnerExperienceSteps` is branch-local experience and resets on a new fork.
- Champions and Hall entries are shared research artifacts across branches.

## Continual rehearsal
The v0.1.2 rehearsal distributions are unchanged. At Scarcity the target remains approximately 10% Motor, 15% Foraging, 20% Obstacle Avoidance, 55% Scarcity.

## Performance architecture
Training, visualization, UI, validation, and storage are timed separately. Training uses a capture-free model path so visualization-only snapshots are not allocated inside headless training. The PPO hot loop reuses derivative buffers and computes Adam bias corrections once per update. Canvas/DOM refresh rates are decoupled from training throughput.

Adaptive compute uses the same rollout size as Balanced and adjusts only browser idle delay using observed rendering responsiveness.

## Data separation
- training: `train:*`
- validation / Champion selection: `validation:v3`
- historical comparison: `heldout:compare:v2`
- final diagnostic: `heldout:final:v2`
