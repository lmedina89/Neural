# What is learned vs programmed

## Programmed

World physics, collision rules, sensors, energy accounting, reward definitions, PPO itself, procedural generation, and the neural architecture are code.

## Learned

The numerical neural parameters connecting observations, recurrent state, policy outputs, and value estimate are optimized from experience. There is no policy rule such as “if food is left, turn left.”

The introductory curriculum includes a configurable approach-to-food shaping reward. It is intentionally visible in configuration because dense shaping can create unintended strategies. Held-out evaluation is used to expose memorization or reward exploitation.
