# Architecture — v0.1.4.1.3

The production learning path is unchanged from v0.1.4.1.2. The new work remains isolated in the application/evaluation layer.

## Read-only geometry measurement

`src/evaluation/occlusionSpinTelemetry.js` computes two external geometry facts for the nearest food:

- raw center-to-center line-of-sight intersection with walls;
- direct travel-corridor intersection using the real agent collision radius to expand walls.

These values are never appended to the 10-value policy observation and never affect rewards, PPO, curriculum, curiosity, or world stepping.

## Live rotation-trap recorder

Protocol: `live-occlusion-rotation-trap:v2`.

The recorder runs only around normal **OBSERVE** stepping. It stores a bounded rolling history and detects two descriptive failure patterns:

1. large cumulative angular travel with poor food-distance progress;
2. a smaller turn loop with poor progress plus repeated turn-direction reversals or repeated food-bearing crossings.

Captured events include action mix, temporary away-from-food thrust, LOS/path blockage, danger rays, recurrent-state summaries, angular state, target identity, and a short recovery tail. At most 12 completed events are retained in session memory.

## Controlled failure characterization

Protocol: `occlusion-failure-characterization:v2`.

The audit creates temporary worlds with identical start pose, food position, zero recurrent state, and paired stochastic action RNG streams. Seven wall geometries are compared: OPEN, CLEARANCE, NARROW CENTER, WIDE CENTER, LEFT-HEAVY, RIGHT-HEAVY, and LONG DETOUR.

Each temporary trajectory uses normal policy inference and recurrent-state evolution but stores no PPO transition and mutates no persistent learner state. Before/after model serialization checks guard both the Learner and Balanced Champion.

The audit measures path clearing, time-to-clear, reach, lateral excursion, temporary retreat, cumulative/net rotation, turn reversals, bearing crossings, action mix, wall hits, and rotation-trap incidence.

## Performance boundary

The rich recorder remains confined to OBSERVE. LEARN hot-loop training is not instrumented by per-step rotation-trap analysis. Research audits run only on explicit user request while training is paused.

Save schema remains **10**.
