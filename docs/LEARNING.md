# What MicroMind learns — v0.1.4.1.1

Programmed are the physics/world rules, observation/action definitions, external rewards, curriculum/rehearsal schedule, validation protocols, curiosity predictor architecture, and intrinsic-reward caps.

Learned by the **1,040-parameter policy** are recurrent hidden-state dynamics, policy probabilities, value estimates, and behavioral strategies. Learned separately by the **441-parameter predictor** is how the nine dynamic sensory values tend to change after each state/action pair.

v0.1.4.1 does **not** add a reward, rule, policy feature, or behavioral intervention. It changes how regression evidence is measured. Ordinary validation stays at 12 fixed seeded-stochastic episodes per skill. Suspicious checkpoint drops trigger a larger paired replay against the exact previous same-lineage policy on the same confirmation seeds. The paired per-episode differences receive a 95% confidence interval and are labeled MEASURED DROP, LIKELY NOISE, or CONFIRMED REGRESSION.

PPO stability telemetry remains observational. Curiosity remains the same experimentally bounded prediction-error reward from v0.1.3; it is not consciousness, desire, or subjective interest.


v0.1.4.1.1 also does **not** teach a new behavior. The rear-target audit merely probes the existing policy under controlled bearings. It does not add rear vision, occlusion rules, angular braking, reward shaping, scripted target memory, or special turn logic. Those remain candidate hypotheses until the long-running Learner is measured.
