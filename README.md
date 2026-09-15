# MicroMind v0.1.3.4 — Prediction Echo & Attention Fields

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in procedural worlds while its policy, recurrent state, decisions, rewards, Champion history, stability telemetry, and separate **441-parameter curiosity predictor** can be inspected live.

**Build:** `PREDATTN-0134`  
**Save schema:** 9

## What this release changes

v0.1.3.4 is another **visualization-only milestone** on top of v0.1.3.3 Neural Flow & Cognitive FX.

The World panel now has a compact cognitive-overlay selector:

- **Attention + Echo** (default)
- **Attention Field**
- **Prediction Echo**
- **Clean World**

### Attention Field

Real sensory influence from the running policy is projected back into the simulated world:

- food influence creates a cyan/green light field and salience path around the nearest real food;
- danger influence brightens the real three danger rays and their sensed contact regions;
- energy influence adds a restrained violet pressure ring around the agent;
- the existing decision/confidence/novelty cues remain tied to actual policy outputs.

The overlay is strictly read-only. It does not alter observations, rewards, actions, physics, or learning.

### Prediction Echo

The curiosity forward model now gets a separate **Echo Lens** inside the World view.

The lens visualizes one genuine sampled env-0 transition in agent-local sensor space:

- **gold / hollow** markers = the forward model's predicted next sensory state;
- **cyan / filled** markers = the actual next sensory state;
- food x/y become paired vector markers;
- the three danger channels become paired proximity markers on fixed local sensor rays;
- speed, turn-rate and energy mismatch appear as short orbit arcs;
- the lens halo strength follows real predictor error and novelty.

The Echo Lens intentionally lives in sensor space rather than pretending an older sampled training transition is the current physical world position at high training speed.

## Learning behavior is unchanged

The following learning-critical files are byte-for-byte identical to v0.1.3.3 `NEURAFX-0133`:

- `src/ai/model.js`
- `src/ai/ppo.js`
- `src/ai/rollout.js`
- `src/ai/session.js`
- `src/ai/curiosity.js`
- `src/sim/world.js`
- `src/sim/curriculum.js`
- `src/evaluation/evaluator.js`
- `src/storage/checkpoints.js`
- `src/utils/prng.js`

So this release does **not** retune PPO, curiosity, rewards, curriculum/rehearsal, Champion logic, evaluation, branch behavior, or saves.

## Run locally

```bash
npm test
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

GitHub Pages can serve the repository root directly.
