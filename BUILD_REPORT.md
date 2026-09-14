# MicroMind v0.1.2.1 — Build Report

**Build marker:** `CHAMPEFF-0121`  
**Baseline:** exact v0.1.2 `AUTOCONT-012` package  
**Baseline SHA-256:** `15cb2a0fc94302e0e1df7f2a1ecc372afef9831afcb34c1b6627df58138a5913`  
**Save schema:** 6 (loads schemas 1–6)

## Scope containment
This is an infrastructure/performance milestone. The following are intentionally unchanged from v0.1.2:
- 1,040-parameter recurrent actor-critic architecture;
- observations/actions;
- rewards;
- PPO hyperparameters and learning objectives;
- curriculum definitions and rehearsal target ratios;
- validation/Champion promotion protocol;
- comparison/final-heldout protocols;
- physics and procedural world rules.

## Implemented
- Permanent, fingerprint-deduplicated **Hall of Fame** with dedicated IndexedDB persistence plus schema-6 checkpoint backup.
- Schema-5 migration bootstrap: when no Hall exists, the preserved Balanced Champion is pinned as the migration baseline.
- Hall brains are inspectable in Observe/Probe and usable as explicit manual fork sources.
- Safe learner branching: forking freezes the current Learner; switching branches freezes the lineage being left. Only one lineage trains at once.
- Global experiment age remains monotonic; branch-local experience is tracked separately.
- Research Status UI exposes experiment age, active lineage, policy origin, active Champion, Hall count/list, and frozen branches.
- Performance instrumentation for simulation throughput, PPO time, browser FPS, UI time, validation time, and storage time.
- Adaptive compute profile changes only idle duty cycle, not PPO rollout size/hyperparameters.
- Training hot-path optimization: capture-free model forward/action, reduced visualization-only allocations, reusable PPO gradient buffers, one-per-update Adam bias correction, direct ray-observation calculation, throttled DOM/Canvas updates, cached control DOM.
- PPO and action RNG state are now checkpointed for stronger reproducible continuation.

## Behavior-preservation evidence
Side-by-side deterministic differential against exact v0.1.2 for 14,400 transitions with identical seed/stage/workload:
- **0 differing model parameter elements**
- **maximum parameter difference 0**
- identical steps, episodes, curriculum, mean return, mean food, action RNG, and PPO RNG state.

## Performance A/B
Controlled headless Node workload, five independent process runs per version:
- v0.1.2 median: **27,485 steps/sec**
- v0.1.2.1 median: **48,957 steps/sec**
- improvement: **~78%**

This benchmark is development-machine evidence only; physical iPhone Safari remains the acceptance authority for mobile throughput/thermal behavior.

## Automated QA
- `npm test`: **50/50 PASS**
- Hall immutability/dedup: PASS
- Hall fork / learner branch switch: PASS
- schema-6 roundtrip: PASS
- schemas 1–5 migration coverage retained: PASS
- capture-free policy equivalence: PASS
- profiler finite/validation-separated: PASS
- optimizer/action RNG persistence: PASS
- recovery-safe Manual/Autosave protections retained: PASS
- final heldout remains diagnostic-only: PASS

## Physical iPhone gate
1. On v0.1.2, press **Save Manual** before deployment.
2. Deploy v0.1.2.1 and verify `v0.1.2.1 • CHAMPEFF-0121`.
3. Confirm the old Manual Save still shows the multi-million-step run; press **Load Manual**.
4. If the persistent Hall was previously empty, verify **HOF-001** appears from the loaded Balanced Champion (expected physical run: ~4,800,240 steps).
5. Verify active Learner lineage/step count and existing Champions remain intact.
6. Press **Save Manual** once after migration so schema 6 backs up Hall/branches.
7. Resume on Balanced or Adaptive and compare sustained physical steps/sec / responsiveness with v0.1.2.
8. Do not fork during the initial acceptance test; first validate preservation and performance.
