# v0.1.3 benchmark

## Baseline containment
Built from exact packaged v0.1.2.1 `CHAMPEFF-0121`.

The following core files are byte-for-byte unchanged from that baseline:
- `src/ai/model.js`
- `src/ai/ppo.js`
- `src/ai/rollout.js`
- `src/sim/world.js`
- `src/sim/curriculum.js`
- `src/evaluation/evaluator.js`

This keeps the 1,040-parameter policy architecture, PPO math, external reward, world rules, curriculum, and evaluation protocols fixed while curiosity is added around the training reward stream.

## Predictor-learning sanity evidence
In the 80,128-step release run at the hardest curriculum with continual rehearsal:

- mean sampled prediction error near 10k steps: **0.3188**
- mean sampled prediction error near 80k steps: **0.0331**
- predictor batch loss near 10k: **0.15938**
- predictor batch loss near 80k: **0.01656**

The intrinsic bonus remained small (roughly `0.0001–0.00025` averaged across all training transitions in the sampled checkpoints) while the predictor learned familiar dynamics.

External all-skills behavior also improved versus the random initial policy in this short run:
- initial generalization: **0.5%**
- final Learner generalization: **11.4%**
- protected Champion generalization: **15.5%**

This is a short sanity run, not a claim that curiosity beats the pre-curiosity system after only 80k transitions. The physical multi-million-step comparison is the meaningful experiment.

## Throughput containment
Headless Node, identical performance workload, five fresh processes each:

- exact v0.1.2.1 median wall-clock throughput: **31,884 steps/sec**
- v0.1.3 median wall-clock throughput: **29,965 steps/sec**
- development-machine median change: **about -6.0%**

Curiosity uses one-step prediction samples every fourth environment transition and one predictor optimizer batch per rollout to keep overhead bounded. This benchmark is not an iPhone performance claim; physical Safari is the acceptance authority.

## Evaluation isolation
Automated tests and source audits verify that validation, comparison, and final-heldout evaluation do not import/use the curiosity module or intrinsic reward. Champion promotion therefore remains based on the original external-task evaluation protocol.
