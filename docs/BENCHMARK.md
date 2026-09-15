# v0.1.3.1 benchmark / audit notes

## Baseline

Built directly from exact packaged v0.1.3 `CURIOUS-013` (SHA-256 `df786484b710936674436a61912d5914b0c9e154975d5a58922a35affd9ef19b`). Policy architecture, PPO implementation, world physics/rewards, curriculum definitions, and evaluation math are unchanged.

## What is benchmarked now

This release is an experimental-control release. Its most important performance property is that observe-only curiosity retains predictor learning without adding intrinsic reward, and that matched CONTROL/CURIOSITY descendants receive equal branch-local schedules and fixed audit evaluation seeds.

Automated tests cover those invariants directly.

## Development throughput smoke

Headless Node, 8 environments, Scarcity stage, rollout 36:

- wall-clock: **45,824 steps/sec**;
- last-profile simulation: **142,403 steps/sec**;
- PPO: **3.47 ms/update**;
- curiosity predictor update: **0.20 ms/update**.

These numbers are environment-specific and are not compared directly with earlier five-process release medians. Physical iPhone Safari remains the performance authority.

## Evaluation isolation

Normal validation, historical comparison, final unseen testing, and the new curiosity-ablation evaluation all score external behavior with curiosity reward OFF. The A/B result stream is observational and cannot promote the protected Champion.
