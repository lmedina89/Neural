# MicroMind v0.1.4.0.2 — Spawn Clearance & World Validity Hotfix

**Build:** `SPAWNCLR-01402`  
**Parent:** v0.1.4.0.1 `VISPERF-01401`

This is a narrowly scoped world-generation hotfix for the case where the bot can begin an episode touching or effectively trapped against a newly generated wall. It keeps the current three danger rays and does **not** add scripted escape behavior.

## What changed

- Walls are still generated in the same order and with the same seeded RNG as before.
- After generation, agent, food, and hazard geometry is checked against the existing `world.wallMargin` safety clearance.
- A point that is already valid is left exactly where the legacy generator placed it.
- Only a tight/invalid point is deterministically relocated with the same seeded PRNG and the existing entity-separation rules.
- A deterministic center-out fallback guarantees the correction never silently returns a point inside a wall.
- Food created after collection receives the same wall-clearance validation, so a respawn cannot appear inside/tight against a wall.

## Deliberately unchanged

- 3-ray perception layout (`danger L / F / R`)
- observation size (10), hidden size (24), action size (7)
- policy / PPO / curiosity math and parameters
- reward values and collision response
- curriculum / rehearsal distribution
- Champion, Hall of Fame, lineage and A/B behavior
- validation / held-out protocols
- Cognitive Observatory visuals and the v0.1.4.0.1 off-screen rendering optimization
- save schema (`9`)

## Compatibility / determinism

This patch is intentionally minimal: legacy-valid worlds retain their exact original entity and wall geometry. Seeds that produced unfair geometry change only because the invalid entity is corrected after wall generation. That is the intended behavior.

## iPhone acceptance test

1. Deploy and confirm `v0.1.4.0.2 • SPAWNCLR-01402`.
2. Load the real Manual Save and verify Learner lineage, Champion and step count.
3. Let Scarcity / Obstacle worlds cycle for a while in Observe or Learn.
4. Confirm the bot no longer begins intersecting or pressed tightly against a wall.
5. Confirm the same three front danger rays, Cognitive Flow, Attention/Echo, Memory and History visuals still behave normally.
6. Training throughput should remain in the same range as v0.1.4.0.1 after the visual warm-up/off-screen optimization takes effect.

The planned scientific milestone after this hotfix remains **v0.1.4.1 — Validation Confidence & True Regression Audit**.
