# What MicroMind learns — v0.1.4.1.3

MicroMind still learns with the same recurrent actor-critic policy, PPO update, curiosity module, curriculum, rehearsal schedule, reward structure, observation vector, and world physics used by the accepted v0.1.4.1.2 parent.

v0.1.4.1.3 is **observational only**. It does not teach detouring, add line-of-sight to the neural observation, modify danger sensors, add pathfinding, alter rewards, or change angular control.

The new rotation-trap telemetry exists to answer a specific question raised by physical testing: when food remains numerically represented in the policy observation but the direct route is obstructed by a wall, does the policy possess a learned strategy for temporarily moving laterally or away from the food vector, or does it fall into repeated turning/search behavior?

The controlled audit therefore measures actual behavior rather than prescribing it. OPEN, CLEARANCE, centered blockers, mirrored asymmetric blockers, and a long barrier are evaluated with paired stochastic action streams. Metrics such as path clearing, lateral excursion, temporary retreat, cumulative rotation, turn reversals, bearing crossings, action mix, and eventual food reach are diagnostic outputs only.

A “successful detour” means that an initially blocked direct path became clear and the agent later reached food after measurable lateral excursion. It is not a policy label, target action, imitation signal, or reward.

Save schema remains **10**. Diagnostic state is session-only and is never serialized into the learner or Champion.

## v0.1.4.2 Detour Learning A/B

Occlusion characterization showed that the accepted learner can orient and reach rear targets in open space, yet repeatedly enters rotation traps when a wall blocks the direct food path. v0.1.4.2 tests one narrow hypothesis: the signed step-to-step approach reward can punish temporary retreat that is required for a detour.

The experiment preserves the current learner and creates matched CONTROL and DETOUR descendants. CONTROL keeps the legacy signed approach reward. DETOUR awards dense approach reward only for new closest-distance records; retreat is neutral and returning to a prior best distance cannot farm reward. All other reward terms, observations, sensors, PPO, curiosity, recurrence, physics and curriculum mix are held constant. Fixed branch checkpoints compare both retained skills and the seven-geometry occlusion audit. No winner is selected automatically.
