# MicroMind v0.1.4.2 — Build Report

**Version:** `0.1.4.2`  
**Build marker:** `DETOURAB-0142`  
**Parent baseline:** v0.1.4.1.3 `OCCCHAR-01413`

## Objective

Run a protected causal A/B test of the strongest mechanism suggested by the v0.1.4.1.3 occlusion characterization: the legacy signed straight-line approach reward may punish the temporary retreat/lateral motion needed to route around an obstacle.

The build deliberately avoids declaring that hypothesis correct. It preserves the current learner and compares two matched descendants before any permanent learning decision is made.

## Learning change under test

`World` now accepts an explicit `approachRewardMode`.

- **legacy**: accepted signed step-to-step distance shaping, including retreat penalties.
- **record-progress**: positive dense reward only when the agent achieves a new closest distance; retreat is neutral and returning to an already-rewarded distance earns no additional approach reward.

All non-approach reward components remain unchanged.

## A/B controls

1. The exact pre-experiment Learner is frozen before either descendant is created.
2. CONTROL and DETOUR start from identical policy, optimizer, curiosity, curriculum, environment-seed cursor, action RNG and rehearsal bytes.
3. CONTROL uses `legacy`; DETOUR uses `record-progress`.
4. PPO schedule age is based on the common audit origin plus branch-local learning experience, preventing the second-run branch from receiving a different learning-rate age merely because global steps are monotonic.
5. Ordinary validation/Champion promotion and adaptive curriculum transitions are suspended during the experiment.
6. Fixed 500k branch checkpoints run the same full retention suite plus `occlusion-failure-characterization:v2`.
7. Branch switching freezes/restores branch-specific training state exactly.
8. No automatic winner selection or Champion promotion occurs.
9. The original learner remains a recoverable frozen branch.

## Persistence

Save schema advances **10 → 11** to store the approach reward mode and Detour A/B experiment/branch state. Schemas 1–10 remain supported and migrate to `legacy` with no active detour experiment.

## Parent parity

With no Detour A/B active, v0.1.4.2 defaults to `legacy`. A deterministic parent/candidate continuation using seed 539, 4 training environments and **3,456 training steps** produced exactly equal:

- policy state;
- PPO optimizer state;
- curiosity predictor state;
- curriculum state;
- environment cursor;
- stochastic action RNG state;
- episodes and episode history;
- rehearsal history;
- learner-experience count.

Thus merely deploying v0.1.4.2 does not alter accepted learning behavior until the controlled experiment is explicitly started.

## Verification

Final test/build/package results are appended after clean packaging.
- Clean-unzip verification: the packaged repository was extracted to a fresh directory and `npm run check` again passed **106/106** with a successful static rebuild.
