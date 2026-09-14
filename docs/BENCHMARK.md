# Controlled learning benchmark — v0.1.0 LEARNLAB-010

A reproducible smoke benchmark was run with:

- brain seed: `424242`
- curriculum: Stage 0 / Motor Nursery (automatic curriculum disabled for the benchmark)
- training environments: 8
- training steps: 40,192
- held-out evaluation episodes: 24
- held-out seed domain: `heldout:v1`
- evaluation policy: stochastic categorical policy with a deterministic per-seed RNG, so repeated benchmark runs are reproducible

## Results

| Metric | Before training | After training |
| --- | ---: | ---: |
| Mean held-out return | -9.775 | 2.189 |
| Mean held-out food collected | 0.083 | 2.292 |
| Mean held-out episode steps | 474.0 | 682.5 |

Training metrics during the same run reached a recent mean return of `2.271`, mean food `2.525`, and policy entropy `1.184` at 40,192 steps.

This benchmark is intentionally small. It demonstrates that the learning loop changes the policy and improves behavior on deterministic held-out worlds not used for training. It is not evidence of general intelligence.
