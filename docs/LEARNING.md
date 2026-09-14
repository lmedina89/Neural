# What MicroMind Learns — v0.1.1.1

The world defines physics, sensors, rewards and curriculum generation. There is still no hidden rule telling the agent to turn toward food, brake near a wall, or prefer one action in a particular state.

The actor-critic learns policy/value parameters from on-policy experience. The recurrent state comes from real observations and the prior hidden state.

## Four different evidence streams

v0.1.1.1 intentionally separates evidence by purpose:

1. **Training performance** — recent active-curriculum episodes. Noisy and directly optimized.
2. **Validation:v3** — fixed all-skills worlds used to protect/select archived policies.
3. **heldout:compare:v2** — a comparison-only domain for historical checkpoints.
4. **heldout:final:v2** — the final Unseen Test diagnostic.

The final holdout never changes the model-selection state. This matters because once a developer repeatedly chooses models from a holdout, it is no longer meaningfully unseen.

## Why confidence ranges were added

A score such as `33%` from a small evaluation can move because of policy stochasticity and a small sample of worlds. v0.1.1.1 computes skill competence per episode and reports an uncertainty range rather than pretending the point estimate is exact.

A large drop becomes a WATCH first. The same evidence must repeat before it becomes CONFIRMED.

## Specialization versus forgetting

A policy can lose performance on one narrow skill while gaining useful competence elsewhere. v0.1.1.1 therefore separates:

- healthy retention;
- specialization watch;
- skill-regression watch;
- confirmed specialization;
- confirmed skill regression;
- confirmed aggregate regression.

The labels are diagnostics, not claims about cognition.

A single confirmed narrow skill loss can hold curriculum promotion, but does not automatically roll the entire trainable policy back unless aggregate evidence is also severe or multiple skills catastrophically fail.

## Validation / holdout conflict

Unseen Test re-measures the exact current Latest weights on validation:v3 without committing that measurement. It then evaluates Latest and Protected Balanced on `heldout:final:v2`.

If validation prefers one brain and final holdout prefers the other by a material margin, MicroMind prints a **VALIDATION / HELD-OUT CONFLICT** and takes no automatic action.

That conflict is valuable evidence that the selection metric may not predict broader behavior perfectly.

## Known limitation

This release still protects policies externally. It does not yet solve continual learning inside the network itself. Rehearsal, teacher distillation, regularization, sequence/BPTT learning and world-model learning remain future research candidates.
