# Controlled learning/stability benchmark — v0.1.0.1.1 HISTCONT-01011

A reproducible smoke benchmark was run with:

- brain seed: `424242`
- curriculum: Stage 0 / Motor Nursery (automatic curriculum disabled for the benchmark)
- training environments: 8
- training steps: 60,160
- held-out evaluation episodes: 32
- held-out seed domain: `heldout:v1`
- automatic validation domain: `validation:v1`
- evaluation policy: stochastic categorical policy with deterministic per-seed RNG

## Results

| Policy | Source step | Mean held-out return | Food | Survival |
| --- | ---: | ---: | ---: | ---: |
| Initial | 0 | -9.674 | 0.094 | 0.0% |
| Latest after training | 60,160 | -0.761 | 0.156 | 15.6% |
| Protected Best | 50,176 | **1.211** | **1.344** | **78.1%** |

Automatic validation selected new bests at 10,240 steps (validation score `0.721`) and 50,176 steps (`1.443`). The later 60,160-step policy had regressed substantially on the independent held-out set, while the 50,176-step protected policy retained strong performance.

This is exactly the failure mode v0.1.0.1 is designed to expose and protect against. It does **not** prove the validation score will always predict held-out performance, and it is not evidence of general intelligence. It does demonstrate that the protected-best mechanism can preserve a materially better policy when later PPO updates degrade the live learner.
