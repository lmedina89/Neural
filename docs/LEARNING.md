# What MicroMind learns — v0.1.4.1.3

MicroMind still learns with the same recurrent actor-critic policy, PPO update, curiosity module, curriculum, rehearsal schedule, reward structure, observation vector, and world physics used by the accepted v0.1.4.1.2 parent.

v0.1.4.1.3 is **observational only**. It does not teach detouring, add line-of-sight to the neural observation, modify danger sensors, add pathfinding, alter rewards, or change angular control.

The new rotation-trap telemetry exists to answer a specific question raised by physical testing: when food remains numerically represented in the policy observation but the direct route is obstructed by a wall, does the policy possess a learned strategy for temporarily moving laterally or away from the food vector, or does it fall into repeated turning/search behavior?

The controlled audit therefore measures actual behavior rather than prescribing it. OPEN, CLEARANCE, centered blockers, mirrored asymmetric blockers, and a long barrier are evaluated with paired stochastic action streams. Metrics such as path clearing, lateral excursion, temporary retreat, cumulative rotation, turn reversals, bearing crossings, action mix, and eventual food reach are diagnostic outputs only.

A “successful detour” means that an initially blocked direct path became clear and the agent later reached food after measurable lateral excursion. It is not a policy label, target action, imitation signal, or reward.

Save schema remains **10**. Diagnostic state is session-only and is never serialized into the learner or Champion.
