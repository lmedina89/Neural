# What MicroMind learns — v0.1.3.1

Programmed are the physics/world rules, observation/action definitions, external rewards, curriculum/rehearsal schedule, validation protocols, curiosity predictor architecture, and strict intrinsic-reward caps.

Learned by the **1,040-parameter policy** are recurrent hidden-state dynamics, policy probabilities, value estimates, and behavioral strategies. Learned separately by the **441-parameter predictor** is how the nine dynamic sensory values tend to change after each state/action pair.

v0.1.3.1 separates prediction learning from reward influence. In **Observe only**, the predictor continues learning and reporting real surprise, but that surprise contributes exactly zero reward to PPO. In **Reward ON**, the same bounded potential bonus is applied as in v0.1.3.

The matched A/B audit therefore asks a narrower causal question:

**Starting from the same brain and optimizer state, does adding prediction-surprise reward produce better external-task generalization than merely learning/observing the same predictor?**

Curiosity is not consciousness, desire, or subjective interest. It is an experimentally switchable reinforcement signal derived from learned prediction error.
