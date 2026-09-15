# v0.1.3.2 benchmark / audit notes

## Baseline

Built directly from exact packaged v0.1.3.1 `CURAUD-0131`. Policy architecture, PPO hyperparameters, reward coefficients, world physics, curriculum definitions, curiosity behavior, and evaluation math are unchanged.

## Numerical parity check

A deterministic development run trained v0.1.3.1 and v0.1.3.2 from the same seed for the same 3,840-step sequence with automatic validation disabled. The resulting policy weights, optimizer numeric state (excluding expanded diagnostic `lastStats` fields), curiosity predictor state, curriculum state, and global step count matched exactly by SHA-256.

## Stability overhead

The added diagnostics reuse values already computed during PPO and perform one parameter-delta scan per PPO update. Long-term stability capture is downsampled every 50k global steps. Physical iPhone Safari remains the performance authority.

## Evaluation isolation

Normal validation, historical comparison, final unseen testing, and curiosity-ablation evaluation all remain curiosity-free. Stability telemetry and regression events cannot promote, restore, or modify any policy.
