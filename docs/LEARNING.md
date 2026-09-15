# What MicroMind learns — v0.1.3

Programmed:
- physics and procedural world rules;
- observation/action definitions;
- external task rewards;
- curriculum and rehearsal schedule;
- validation/Champion-selection protocols;
- the curiosity model architecture and strict intrinsic-reward caps.

Learned by the 1,040-parameter policy brain:
- recurrent hidden-state dynamics;
- policy probabilities;
- value estimates;
- behavioral strategies.

Learned by the new 441-parameter curiosity predictor:
- how the nine dynamic sensory values tend to change after each state/action pair.

No rule says "unfamiliar places are over there" or "turn toward novelty." The only new training information is the predictor's real error on experienced transitions.

## Familiarity and surprise
At first the forward model predicts poorly. As repeated transitions become familiar, its error usually decreases. A transition that is unusually hard to predict relative to the recent baseline receives a larger novelty score.

The novelty score is normalized, bounded, sampled sparsely, and episode-budgeted. This is intentionally conservative to reduce the risk of classic intrinsic-motivation pathologies such as seeking endlessly unpredictable events at the expense of the real task.

## Scientific interpretation
Curiosity is not consciousness, desire, or subjective interest. It is an intrinsic reinforcement signal derived from learned prediction error.

The experiment asks a measurable question: **does adding a small prediction-surprise signal improve exploration and later external-task generalization compared with the pre-curiosity Champions?**
