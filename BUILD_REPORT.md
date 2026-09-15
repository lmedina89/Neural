# MicroMind v0.1.4.0.2 — Build Report

**Version:** `0.1.4.0.2`  
**Build marker:** `SPAWNCLR-01402`  
**Parent baseline:** v0.1.4.0.1 `VISPERF-01401`

## Objective

Remove unfair episode starts caused by the original generation order (entities first, walls second) without scripting a wall-escape behavior, changing the three-sensor perception architecture, or retuning learning.

## Root cause confirmed

The parent generator placed the agent, food and hazards before rectangular walls. Wall placement did not test clearance against those existing entities. A baseline audit over seeds 0–4,999 across all four curriculum stages found real clearance violations, including Scarcity seed `2`, where the agent center was inside a wall rectangle (`distance = 0`).

## Implementation

1. Added exact point-to-rectangle clearance geometry using the already-existing `CONFIG.world.wallMargin`.
2. Kept the legacy generation sequence unchanged.
3. After walls are created, only entities that fail physical-radius + wall-margin clearance are corrected.
4. Corrective placement uses the existing seeded PRNG and legacy entity-separation conventions.
5. Added a deterministic center-out fallback if random corrective attempts cannot find a valid point.
6. Applied the same wall validity check to food respawns after collection.
7. Kept all three danger rays and the 10-element observation vector unchanged.

## Verification

- Test suite: **83/83 passed**.
- Bulk geometry audit: **20,000 generated worlds** (5,000 seeds × 4 stages) produced **0** agent/food/hazard wall-or-boundary clearance failures.
- Unit suite independently checks **8,000 generated worlds** (2,000 seeds × 4 stages) on every run.
- Known trapped case: Scarcity seed `2` is deterministically relocated to a valid clear position.
- Legacy-valid seed `77` / Obstacle Avoidance retains exact pre-hotfix agent, food, hazard and wall coordinates.
- Food-respawn validation passed across 160 Scarcity seeds.
- Policy, PPO, rollout, training session, curiosity, curriculum, evaluator, checkpoint, PRNG, all visualization renderers, app controller and CSS are byte-for-byte identical to v0.1.4.0.1. `world.js` is the intentional behavioral change; `config.js` changes only release identity while using the pre-existing `wallMargin` value.
- No-wall deterministic continuation parity: v0.1.4.0.1 and v0.1.4.0.2 trained from the same seed with fixed Motor Nursery curriculum and finished with exactly equal policy weights, PPO optimizer state, curiosity predictor, curriculum state, rehearsal history, episode history, global steps/episodes and Learner experience.
- Save schema remains **9**.
- Build completed successfully.
- Development performance smoke: ~**157.5k simulation steps/sec** in the simulation phase and ~**52.5k training steps/sec** in the Node harness on the final source tree. Browser/Safari throughput remains device-dependent.

## Intentional behavioral difference

Worlds that were already valid remain unchanged. A seed that previously put the agent, food, or a hazard too close to/inside a wall will now generate a deterministic corrected position. This is the purpose of the hotfix and means those specific world trajectories can differ from the parent build.

## Deferred perception change

Rear danger rays are **not** part of this build. A future Perception v2 may deliberately test 5- or 7-ray sensing with controlled weight migration; that architecture change remains separate from this world-validity fix.
