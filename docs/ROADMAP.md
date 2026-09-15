# Roadmap

Current: **v0.1.3.2 — Continual Learning Stability Observatory**.

The v0.1.3.1 matched curiosity A/B test did not show a durable final advantage for either reward-on curiosity or the reward-zero control, and both branches showed checkpoint volatility. Curiosity therefore stays unchanged while the underlying continual PPO learner is instrumented.

Immediate acceptance goal:
1. preserve the current learner, Champion, Hall, frozen A/B branches, and curiosity state through schema-9 migration;
2. verify compact/collapsible iPhone UI and no Safari focus-zoom regression;
3. let the same learner continue roughly 2–3M steps without retuning;
4. capture PPO telemetry through at least one ordinary validation swing if one occurs;
5. inspect KL/clip pressure, critic loss/explained variance, gradient behavior, parameter movement, and per-skill deltas before choosing any learning change.

Only after evidence identifies a likely mechanism should v0.1.3.3 change one learning variable at a time. Candidate experiments include stricter update guards, critic stabilization, or interference/retention changes—but none are approved until the observatory data points to them.

Longer-term candidates remain world-model/predictive memory, primitive imagination/planning, richer delayed-memory tasks, and controlled policy-capacity experiments.
