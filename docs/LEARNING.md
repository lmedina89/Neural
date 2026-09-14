# What MicroMind Learns — v0.1.1

The environment defines physics, sensors, rewards and curriculum worlds. There is no hidden rule saying which direction to turn toward food or how to avoid an obstacle.

The actor-critic learns its policy/value weights from on-policy experience. The recurrent state is computed from real observations and the previous hidden state.

## What v0.1.1 adds to learning

The training objective is still PPO, but a candidate update is no longer accepted blindly. KL limits constrain destructive optimizer moves, while a fixed external validation protocol checks whether useful behavior survives across all learned task stages.

The system distinguishes:

- **training performance** — recent episodes in the active curriculum;
- **validation skill retention** — fixed `validation:v2` worlds never used for training;
- **held-out testing** — a separate `heldout:v1` domain used only when the user runs Unseen Test / Compare Brains.

This separation is critical: validation may select/protect a brain, but the final held-out worlds do not participate in that selection.

## Skill retention score

Each curriculum stage gets a bounded competence score composed primarily of food acquisition plus survival and remaining energy. It is diagnostic, not a claim of intelligence or consciousness.

A catastrophic-forgetting alert requires both:

1. the skill previously achieved meaningful competence; and
2. its current score falls by more than the configured tolerance.

## Known limitation

v0.1.1 improves policy preservation but does not solve continual learning in the research sense. Automatic recovery can return to a known good policy; it does not yet make the network internally consolidate old skills while learning new ones. Rehearsal, distillation, elastic-weight regularization, sequence training and learned world models remain future experiments.
