# v0.1.2.1 benchmark

## Behavior-preservation differential
The exact v0.1.2 `AUTOCONT-012` package and v0.1.2.1 were run side-by-side with the same seed, 8 environments, Scarcity stage, 36-step rollouts, validation disabled, and 50 training rollouts (14,400 transitions each).

Result:
- model parameter elements different: **0**
- maximum parameter difference: **0**
- total steps / episodes: identical
- mean return / food: identical
- action RNG state: identical
- PPO RNG state: identical

This is a focused deterministic equivalence check of the optimized training hot path, not a proof that every browser timing scenario is identical.

## Controlled throughput A/B
Headless Node benchmark, exact same workload: seed 539, 8 environments, Scarcity, 36-step rollouts, 10 warmups + 120 measured rollouts. Five independent process runs per version.

- v0.1.2 median: **27,485 steps/sec**
- v0.1.2.1 median: **48,957 steps/sec**
- median improvement: **~78%**

This is a controlled development-machine benchmark, **not an iPhone performance claim**. Physical Safari throughput depends on thermal state, browser scheduling, visual mode, and device load.

## Learning sanity run
An 80,128-step Scarcity-stage continual-learning run still learned under the unchanged v0.1.2 learning configuration:

- initial all-skills generalization: **0.5%**
- end Learner: **35.3%**
- mean food: **1.344**
- survival: **46.9%**
- observed final rehearsal mix: **5.0% / 18.1% / 21.3% / 55.6%**

Champion promotion remained repeat-confirmed; the run ended before the pending second confirmation promoted the stronger Learner.
