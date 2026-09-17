# MicroMind v0.1.4.1.3 — Occlusion Failure Characterization

**Build:** `OCCCHAR-01413`  
**Parent:** v0.1.4.1.2 `OCCSPIN-01412`

This is a deliberately narrow **read-only diagnostic build**. The previous controlled audit showed a repeatable split: the accepted Learner and Balanced Champion could reach food in OPEN and path-blocked/LOS-clear geometry, but both failed when a wall blocked the food centerline and accumulated much more turning. v0.1.4.1.3 characterizes that failure without changing the policy, sensors, rewards, physics, curriculum, PPO, recurrent network, or Champion behavior.

## What is new

### Broader live rotation-trap recorder

The OBSERVE recorder now catches both sustained one-way spinning and the more visually realistic **turn → hesitate → reverse → turn** loops that the old continuous-spin threshold could miss.

It keeps a bounded session-only history and measures:

- cumulative and net angular travel;
- turn-direction reversals;
- repeated crossings of the food bearing;
- turn / thrust / brake action rates;
- thrust steps that temporarily increase food distance;
- direct-path and centerline LOS blockage;
- danger-ray values, angular velocity, recurrent-state change, target switches, reward components, and recovery context.

A captured event is descriptive telemetry only. The detector never becomes a policy input or reward.

### Seven-geometry occlusion failure audit

**Run Occlusion Failure Audit** pauses training and tests the exact Learner in seven paired geometries using the same start pose, food position, zero recurrent state, and paired stochastic action RNG stream for corresponding trials:

- **OPEN** — no wall;
- **CLEARANCE** — centerline LOS is clear but the agent-radius travel corridor is blocked;
- **NARROW CENTER** — small centered blocker;
- **WIDE CENTER** — taller centered blocker requiring a real bypass;
- **LEFT-HEAVY BLOCK** — asymmetric blocker whose shorter bypass is to the right;
- **RIGHT-HEAVY BLOCK** — mirrored asymmetric blocker whose shorter bypass is to the left;
- **LONG DETOUR** — large barrier requiring sustained lateral commitment.

The audit reports reach rate, direct-path clearing, time to clear, successful detour rate, rotation-trap rate, steps to food, total turns, reversals, food-bearing crossings, turn/thrust/away-thrust mix, maximum temporary retreat, and wall hits. A Balanced Champion is tested too when available.

“Detour” is only a diagnostic label: a run must actually reach food after clearing an initially blocked direct path with measurable lateral excursion. There is no pathfinder or scripted navigation behavior.

## What did not change

No changes were made to:

- policy architecture, weights, recurrent dynamics, or observation size;
- PPO math, hyperparameters, optimizer, or rollout behavior;
- curiosity model or intrinsic reward;
- curriculum, rehearsal, rewards, or spawn logic;
- world physics, collisions, BRAKE behavior, or turn physics;
- the three danger rays or nearest-food X/Y/distance inputs;
- Champion promotion, Hall of Fame, branching, or validation-confidence logic;
- checkpoint/save schema (still **10**);
- final held-out evaluation.

All v0.1.4.1.3 telemetry and controlled-audit results remain session-only.

## iPhone acceptance test

1. Deploy and confirm **`v0.1.4.1.3 • OCCCHAR-01413`**.
2. Load the intended long-running Manual Save and confirm the Learner lineage/step count and Balanced Champion are correct.
3. Open **RESEARCH → OCCLUSION / ROTATION-TRAP TELEMETRY** and run **Occlusion Failure Audit** once.
4. Screenshot the Learner and Champion tables, especially NARROW/WIDE, LEFT/RIGHT-heavy, and LONG DETOUR.
5. Switch to **OBSERVE** and let the normal loaded policy run until the visually observed turning loop occurs.
6. Return to RESEARCH after a trap is captured and screenshot the live event table. Several real events are better than one.
7. Do not change sensors, rewards, PPO, recurrent training, or physics yet. v0.1.4.2 should change only the mechanism supported by these results.
