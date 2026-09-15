# What MicroMind learns — v0.1.3.2

Programmed are the physics/world rules, observation/action definitions, external rewards, curriculum/rehearsal schedule, validation protocols, curiosity predictor architecture, and intrinsic-reward caps.

Learned by the **1,040-parameter policy** are recurrent hidden-state dynamics, policy probabilities, value estimates, and behavioral strategies. Learned separately by the **441-parameter predictor** is how the nine dynamic sensory values tend to change after each state/action pair.

v0.1.3.2 does **not** add another reward, rule, or behavior. It measures the existing PPO learning process so performance collapses can be diagnosed instead of guessed at. The stability observatory records losses, KL/clipping pressure, critic explained variance, gradients, weight movement, and validation deltas, but never changes training in response.

Curiosity remains an experimentally switchable prediction-error reward from v0.1.3/v0.1.3.1; it is not consciousness, desire, or subjective interest.
