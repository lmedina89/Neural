# MicroMind Roadmap

## Completed foundation

- genuine trainable recurrent actor-critic
- procedural seeded worlds
- real neural visualization
- checkpoint history beyond 1M steps
- separate validation / held-out seed domains
- recovery-safe manual/autosave slots
- protected best-brain system
- v0.1.1 fixed skill-retention validation
- v0.1.1 guarded PPO and catastrophic-forgetting recovery
- v0.1.1 multi-objective specialist archive

## Next candidate milestone

Do not choose the next milestone until v0.1.1 is physically tested on the long-running iPhone brain.

Likely research directions after acceptance:

1. **Continual-learning training**, not just rollback protection: rehearsal across earlier curricula, policy distillation or regularization against a retained teacher.
2. **Better recurrent credit assignment:** sequence minibatches / truncated BPTT or a true GRU.
3. **Intrinsic curiosity:** novelty reward with careful anti-exploit instrumentation.
4. **Learned world model:** predict latent next state/reward before any imagined-future visualization is added.
5. **Planning / imagined rollouts.**
6. **Memory-specific tasks and delayed cues.**
7. **Open-ended curriculum generation.**

The project should continue favoring measurable capability over network size.
