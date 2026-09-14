# What MicroMind learns

Programmed: physics, sensors, action space, rewards, curriculum generator, validation protocol and rehearsal schedule.

Learned: neural weights, recurrent state dynamics, policy probabilities and value estimates.

v0.1.2 does **not** script food seeking, obstacle avoidance, survival strategy or skill retention. Rehearsal only changes which worlds are experienced; it does not provide correct actions.

Behavioral regression is now evidence, not a restore trigger. Only numerical optimizer failures (for example a hard-KL violation) can reject an update automatically.
