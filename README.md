# MicroMind v0.1.2.1 — Champion Preservation & Training Efficiency

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in deterministic procedural worlds while its activations, decisions, memory, rewards, continual rehearsal, validation, and generalization are inspectable.

**Build:** `CHAMPEFF-0121`

## What changed in v0.1.2.1
This is an infrastructure/performance milestone. It intentionally does **not** change the neural architecture, observations, action space, reward coefficients, PPO learning hyperparameters, curriculum definitions, rehearsal targets, validation protocols, physics, or world-generation rules from v0.1.2.

- **Permanent Hall of Fame:** pin a validated Champion into an immutable archive. Hall entries have their own IndexedDB record and are also backed up inside schema-6 checkpoints.
- **Safe learner branching:** `Fork New Learner` freezes the current learner and starts a new lineage from a Champion or Hall entry. Only one lineage trains at a time.
- **Switchable frozen branches:** return to an earlier learner lineage without erasing the branch you are leaving. Global experiment age remains monotonic.
- **Explicit lineage/policy origin:** Research Status separates experiment age, active lineage, parent policy, Champions, Hall entries, and frozen branches.
- **Training profiler:** simulation throughput, PPO time, browser FPS, UI time, validation time, and storage time are visible.
- **Lower-overhead training:** non-visual training paths no longer build visualization snapshots, PPO hot loops reuse buffers, Canvas/DOM work is throttled, and static control DOM is reused.
- **Adaptive compute mode:** changes browser idle duty cycle only; Balanced PPO rollout size/math is preserved.

When a **schema-5 v0.1.2** save is loaded and no Hall exists yet, the current Balanced Champion is automatically pinned as the migration baseline. For the physically tested run this is intended to preserve the ~4.800M Champion as `HOF-001`.

## Controls
- **Learn:** train the active autonomous Learner.
- **Observe / Probe:** inspect Learner, Champions, or Hall-of-Fame brains.
- **Pin Champion:** permanently preserve the selected validated Champion.
- **Fork New Learner:** manually start a new lineage from a Champion/Hall brain while freezing the current lineage.
- **Switch Branch:** freeze the current learner and resume another stored learner branch.
- **Unseen Test:** final all-skills diagnostic only; it does not train or promote.
- **Compare Brains:** historical/Hall comparison on a separate comparison holdout.
- **Compute:** Eco / Balanced / Adaptive / Max. Adaptive changes duty-cycle delay based on browser responsiveness; it does not alter PPO hyperparameters.

## Run
```bash
npm test
npm run benchmark
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

Then open `http://localhost:8080/`.

GitHub Pages can serve the repository root directly.
