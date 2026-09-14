# What MicroMind learns — v0.1.2.1

Programmed: physics, observations, action space, rewards, procedural curriculum, rehearsal schedule, validation protocols, Champion-selection rules, persistence, and research instrumentation.

Learned: recurrent-network weights, hidden-state dynamics, policy probabilities, and value estimates.

v0.1.2.1 does **not** add a new cognitive algorithm. The autonomous continual-learning behavior from v0.1.2 is intentionally frozen for this milestone.

## Knowledge preservation vs behavior control
- The **Learner** remains free to regress, specialize, recover, and discover.
- **Champions** preserve repeat-confirmed competence.
- **Hall-of-Fame** entries preserve selected historic Champions permanently.
- Forking from a Champion/Hall entry is a **manual research action**, never an automatic behavioral rollback.
- Only numerical PPO safety can automatically reject a malformed/destructive optimizer update.

## Rehearsal
Rehearsal changes which previously available task worlds are experienced; it never supplies a correct action or copies Champion behavior into the Learner.
