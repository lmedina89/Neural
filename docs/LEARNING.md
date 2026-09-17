# What MicroMind learns — v0.1.4.1.2

Programmed are the physics/world rules, observation/action definitions, external rewards, curriculum/rehearsal schedule, validation protocols, curiosity predictor architecture, intrinsic-reward caps, and the read-only research protocols.

Learned by the **1,040-parameter policy** are recurrent hidden-state dynamics, policy probabilities, value estimates, and behavioral strategies. Learned separately by the **441-parameter predictor** is how the nine dynamic sensory values tend to change after each state/action pair.

v0.1.4.1 does **not** add a reward, rule, policy feature, or behavioral intervention. It changes how regression evidence is measured. Suspicious validation drops can trigger a larger paired replay against the exact previous same-lineage policy.

v0.1.4.1.1 also does **not** teach a behavior. Its rear-target audit showed whether an existing policy can orient to clean fixed bearings without changing sensors, memory, rewards or turn physics.

v0.1.4.1.2 remains observational. It computes wall/food geometry and a diagnostic forward cone **outside** the neural observation, records real OBSERVE transitions, and runs temporary controlled worlds. None of those diagnostic values become policy inputs, rewards or PPO targets. The live recorder is session-only and bounded.

PPO stability telemetry remains observational. Curiosity remains the same experimentally bounded prediction-error reward from v0.1.3; it is not consciousness, desire, or subjective interest.
