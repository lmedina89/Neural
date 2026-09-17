# MicroMind v0.1.4.1.1 — Rear-Target & Rotation Audit

**Build:** `ROTAUD-01411`  
**Parent:** v0.1.4.1 `VALCONF-0141`

This is a narrow **observational diagnostic** build. It preserves the v0.1.4.1 validation-confidence work and adds a controlled way to measure the spinning behavior seen when food is behind the agent.

## Why this build exists

The current policy observation does not have visual occlusion for food: nearest-food relative X/Y and distance remain available even when the target is behind the agent. The three limited rays are danger sensors. That means the observed rear-target spin should be diagnosed first as a **turn-control / action-selection / angular-dynamics interaction**, not treated as proof of failed memory.

Also, the existing `BRAKE` action damps linear velocity only. Angular velocity receives the same global angular drag as every other action. This build does not change that physics; it measures whether the learned policy selects BRAKE while already rotating.

## New read-only audit

Research → **REAR-TARGET / ROTATION AUDIT** adds one manual button. When run:

- training is paused so the tested weights cannot move;
- the Learner is tested at fixed target bearings `-179, -135, -90, -45, 0, +45, +90, +135, +179°`;
- each bearing uses 8 deterministic-seed **stochastic-policy** trials;
- each trial starts from rest, with zero recurrent state, in an empty arena using the real policy observations and real world angular physics;
- metrics include facing success, food reach, spin incidence, steps-to-face, cumulative angular rotations, BRAKE-while-turning rate, and the initial dominant policy action;
- if a Balanced Champion exists, it is tested with the same protocol for comparison;
- an in-runtime model-serialization check verifies the audit did not modify either tested policy.

The audit results are deliberately **not persisted** into save schema. Reloading the page clears them.

## What did not change

No changes were made to:

- policy architecture or weights
- PPO math / hyperparameters
- recurrent training behavior
- curiosity model or reward
- curriculum / rehearsal
- world physics
- observation vector / three-ray perception
- rewards
- spawn clearance
- Champion promotion / Hall of Fame
- save/checkpoint schema (still **10**)
- validation-confidence logic
- final held-out evaluation
- visualization renderers / runtime throughput behavior

## iPhone acceptance test

1. Deploy and confirm `v0.1.4.1.1 • ROTAUD-01411`.
2. Load the intended long-running Manual Save and verify Learner lineage, step count and Champion before doing anything else.
3. Open **RESEARCH → REAR-TARGET / ROTATION AUDIT**.
4. Tap **Run Rear-Target Audit**. Training should pause automatically.
5. Screenshot the Learner table, especially `±135°` and `±179°`, plus the rear summary line.
6. If a Balanced Champion exists, screenshot its table too.
7. Do **not** change sensors, turn physics, rewards, PPO or recurrent training yet. We will use these measurements to choose one targeted change.
