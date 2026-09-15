# MicroMind v0.1.4.0 — Build Report

**Version:** `0.1.4.0`  
**Build marker:** `COGOBS-0140`  
**Parent:** v0.1.3.4 `PREDATTN-0134`  
**Checkpoint schema:** 9 (unchanged)

## Objective

Turn MicroMind from an increasingly long research page into a compact interactive AI observatory while adding two new, real-data visual systems: **Memory Constellation** and **Learning Timeline / Brain Lineage**. Preserve all learning behavior exactly.

## Implementation

### 1. Observatory navigation
Added a sticky five-view navigator:

- LIVE
- PREDICT
- MEMORY
- HISTORY
- RESEARCH

The existing controls/save slots remain globally available. Secondary research panels now appear only in RESEARCH. Prediction/curiosity content appears only in PREDICT. Heavy canvases are drawn only for the active observatory view.

### 2. Prediction view
Added a dedicated `predictWorldCanvas` using the existing read-only `WorldRenderer`. It shows the same real Prediction Echo / Attention Field telemetry without requiring the full LIVE layout to remain visible. The curiosity predictor panel remains available directly below it.

### 3. Memory Constellation
Added `src/visualization/memoryRenderer.js`.

Data source:
- `session.hidden[0]` while LEARN is active,
- the real observation-world recurrent state while OBSERVE/PROBE is active.

Projection:
- fixed deterministic 2-axis projection of the 24 recurrent hidden units,
- no trainable parameters,
- no RNG calls,
- no learning-state mutation.

Bounded storage:
- 320 runtime points maximum,
- sampled at a low visual cadence,
- resets on lineage replacement/reload,
- intentionally not stored in checkpoint schema 9.

Visual encodings:
- state-space point = real recurrent hidden state,
- trail = temporal path through those states,
- point glow = recent/activity/novelty information,
- current halo = newest sampled state,
- Experience Ripple = real novelty, food/reward, danger/death, or episode boundary event.

### 4. Learning Timeline + Brain Lineage
Added `src/visualization/historyRenderer.js`.

The renderer reconstructs history from existing persisted state only:
- `validationHistory`,
- `bestArchive`,
- `hallOfFame`,
- `lineageHistory`,
- `frozenLearners`,
- current learner metadata.

The upper timeline plots validation score movement plus Champion/Hall milestones. The lower graph maps lineage/branch nodes and parent links when available. Tap inspection is read-only.

### 5. Mobile/UI constraints
- five-view nav is horizontally scrollable on narrow iPhones,
- existing 16px mobile form-control protection remains,
- Memory/History canvases have bounded portrait heights,
- hidden observatory sections use `display:none!important`,
- no hidden heavy canvas animation.

## Learning-state protection

Before editing, hashes were captured for:

- `src/ai/*.js`
- `src/sim/*.js`
- `src/evaluation/*.js`
- `src/storage/*.js`
- `src/utils/*.js`

After the build, every protected file matched the v0.1.3.4 hash exactly.

A separate deterministic parity run instantiated v0.1.3.4 and v0.1.4.0 with the same seed and trained both for 3,840 steps. Exact equality was verified for:

- policy parameters,
- PPO optimizer state,
- curiosity predictor state,
- curriculum state,
- rehearsal history,
- total steps/episodes.

Result: **exact parity**.

## QA results

- tests: **77/77 pass**
- build: pass
- benchmark: pass
- performance smoke: pass (~22k wall-clock steps/sec in the Node smoke run; browser/iPhone rendering still requires physical testing)
- all source JS syntax: pass
- generated `dist/` contains both new renderers
- root and dist `index.html`, `styles.css`, and `src/app/main.js`: identical after build

## Acceptance notes

This build is intentionally observational. It does not resolve the validation-noise / continual-learning-stability question. The next science-focused milestone should use the v0.1.3.2 observatory evidence to improve confidence in whether apparent skill drops are true regressions or noisy estimates.

## Physical test focus

On iPhone Safari, verify:

1. navigation does not zoom/overflow,
2. LIVE and PREDICT preserve the existing visuals,
3. MEMORY fills without heat/FPS becoming unreasonable,
4. Experience Ripples occur but do not become continuous decorative noise,
5. HISTORY reflects the loaded real save/lineage,
6. returning to LIVE restores normal training throughput.
