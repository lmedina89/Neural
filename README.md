# MicroMind v0.1.3.3 — Neural Flow & Cognitive FX

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in procedural worlds while its policy, recurrent state, decisions, rewards, Champion history, stability telemetry, and separate **441-parameter curiosity predictor** can be inspected live.

**Build:** `NEURAFX-0133`  
**Save schema:** 9

## What this release changes

v0.1.3.3 is a **visualization milestone**, not a learning retune.

The new default **Cognitive Flow** brain view turns real runtime signals into a more expressive living-network display:

- moving pulses follow currently strong sensory, recurrent-memory, policy and value pathways;
- the 24 recurrent units form a compact visual cognitive core;
- actual recurrent `wh` connections become visible memory arcs;
- the winning policy output emits a confidence-weighted decision beam;
- a data-driven Cognitive Halo responds to decision confidence, novelty and current reward;
- real novelty/reward can produce brief halo sparks;
- sensor influence is echoed back into the world as food salience, danger-ray emphasis and agent/action light cues.

The Curiosity / Prediction visualization now makes the forward model easier to read: it shows moving predictor activity plus real **predicted → actual** sensory values and ghost markers for mismatch.

## These effects are not fake AI

The new graphics are derived from the actual model and telemetry already used by MicroMind:

- observation values;
- hidden and previous-hidden activations;
- `wx`, `wh`, `wp`, and `wv` weights;
- policy probabilities;
- critic value estimate;
- current external reward components;
- curiosity prediction error and novelty.

The visualization layer never updates the policy, optimizer, predictor, curriculum, Champion, Hall, branch state, or evaluation history.

## Learning behavior remains v0.1.3.2

The learning-critical model/PPO/session/curiosity/world/curriculum/evaluation/storage files are byte-for-byte unchanged from `STABOBS-0132`. The Stability Observatory therefore continues the same experiment that was already running; this release only gives the AI a richer live visual representation.

The existing Active, Weights, and Strongest brain modes remain available if you want a simpler or lighter view on iPhone.

## Run locally

```bash
npm test
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

GitHub Pages can serve the repository root directly.
