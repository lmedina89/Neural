# What is learned vs programmed

## Programmed

World physics, collision rules, sensors, energy accounting, reward definitions, PPO itself, procedural generation, curriculum thresholds, validation schedules, and the neural architecture are code.

The protected-best mechanism is also engineering logic: it measures policies on fixed unseen validation worlds and retains the strongest checkpoint. It does not improve a policy by itself.

## Learned

The numerical neural parameters connecting observations, recurrent state, policy outputs, and value estimate are optimized from experience. There is no policy rule such as “if food is left, turn left.”

The introductory curriculum includes a configurable approach-to-food shaping reward. It is intentionally visible in configuration because dense shaping can create unintended strategies.

## Why Latest can become worse

PPO is an optimization process, not a monotonic intelligence meter. A later update can damage behavior that an earlier policy had learned, particularly when the task distribution changes with curriculum progression. v0.1.0 physical testing demonstrated exactly this kind of regression.

v0.1.0.1 therefore distinguishes:

- **Latest** — the policy still being updated
- **Best** — the highest validated policy preserved for the current curriculum validation protocol

This lets MicroMind display forgetting/regression instead of hiding it.

## Generalization layers

MicroMind now has two unseen-data checks:

1. `validation:v1` — automatic bounded checks used for best-brain protection.
2. `heldout:v1` — manual Unseen Test and checkpoint comparison. These worlds are kept separate from validation so the policy-selection process cannot directly optimize against the final held-out display set.

Neither domain is used for PPO training.
