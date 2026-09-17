# MicroMind v0.1.4.2 — Detour Learning A/B

**Build:** `DETOURAB-0142`  
**Parent:** v0.1.4.1.3 `OCCCHAR-01413`

v0.1.4.2 is the first learning experiment justified by the occlusion/rotation-trap evidence. The accepted learner and Balanced Champion can turn toward rear targets in open space, but both enter rotation traps when a wall blocks the direct food path. This build does **not** assume the reward change is correct. It creates a protected matched A/B experiment so the current long-running learner can be preserved while two descendants are compared on identical starting state and fixed evaluation protocols.

## Experimental question

Does removing the dense straight-line **retreat penalty**, while preventing approach-reward farming, make autonomous detour learning easier without damaging retained skills?

### CONTROL

Keeps the accepted legacy dense approach term:

`clamp((previousDistance - newDistance) * approachScale, -0.01, +0.01)`

Moving farther from food is penalized even when a wall makes temporary retreat or lateral movement necessary.

### DETOUR candidate

Changes only dense approach shaping:

- reward only **new closest-distance records** to the current food landscape;
- temporary retreat receives `0` dense approach reward instead of a penalty;
- returning to an already-earned distance receives `0`, so moving away and back cannot farm reward;
- after food is eaten/respawned, the closest-distance ledger starts from the new target landscape.

Food reward, wall/hazard penalties, energy, death, curiosity, PPO, observations, recurrent network, action set, physics and danger rays are unchanged.

## Protected matched A/B workflow

**Start 2M + 2M Detour A/B** first freezes the exact current Learner as a recoverable origin. It then creates two descendants with identical:

- policy weights;
- PPO optimizer state;
- curiosity predictor state;
- curriculum state;
- environment seed cursor;
- stochastic action RNG state;
- rehearsal history.

Only `approachRewardMode` differs: `legacy` vs `record-progress`.

The CONTROL descendant runs first. Training pauses at fixed 500k branch-step checkpoints. Each checkpoint runs the same fixed-seed full retention suite and the existing seven-geometry occlusion failure audit. Ordinary validation/Champion promotion and adaptive curriculum transitions remain suspended during the experiment so they cannot contaminate the comparison.

The user explicitly switches between CONTROL and DETOUR. Switching freezes the current descendant exactly and restores the other descendant's model, optimizer, curiosity state, curriculum, RNG continuation and branch-local experience. No winner is selected automatically.

## Save compatibility

Save schema advances **10 → 11** only to persist:

- `approachRewardMode`;
- Detour A/B experiment state;
- active detour branch role/progress.

Schemas 1–10 remain supported. Loading a schema-10 or older checkpoint defaults safely to the accepted legacy reward and no active Detour A/B experiment.

## iPhone acceptance sequence

1. Deploy and confirm **`v0.1.4.2 • DETOURAB-0142`**.
2. Load the intended long-running learner and verify lineage/step count and Balanced Champion.
3. Manual Save before starting the experiment.
4. Open **RESEARCH → DETOUR LEARNING A/B** and press **Start 2M + 2M Detour A/B**.
5. CONTROL starts first. Resume training and let it reach a fixed checkpoint; MicroMind pauses automatically after the checkpoint evaluation.
6. Screenshot the CONTROL checkpoint line, then **Switch to DETOUR** and run the matched candidate to the same checkpoint.
7. Compare balanced retention, blocked-path reach, successful-detour rate and rotation-trap rate. Do not pick a winner from one noisy checkpoint.
8. The preserved origin and inactive descendant remain recoverable throughout the experiment.

## What v0.1.4.2 does not do

It does not add pathfinding, a wall map, rear vision, extra sensors, scripted left/right behavior, angular-brake changes, PPO changes, recurrent/BPTT changes, curiosity changes, or automatic branch selection.
