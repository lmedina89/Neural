# MicroMind v0.1.0 — Learning Laboratory Foundation

**Build:** `LEARNLAB-010`

MicroMind is a browser-based miniature reinforcement-learning laboratory. A small recurrent actor-critic learns to forage, conserve energy, and avoid hazards in deterministic procedurally generated worlds. The world, neural activations, weights, policy probabilities, recurrent state, value estimate, reward statistics, and historical checkpoints are exposed for inspection.

## Run

This project has no runtime dependencies.

```bash
npm test
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080`. `dist/` is also a static GitHub-Pages-ready build.

## Modes

- **LEARN** — trains PPO using multiple headless environments. Rendering is decoupled from training.
- **OBSERVE** — watches the current stochastic policy in a separate procedural world.
- **PROBE** — freezes movement and lets you reposition food, a hazard, or the agent. Policy outputs update from the real model without taking an action.

## Model

`10 observations → 24 tanh recurrent units → 7-action policy + scalar value`

The recurrent state is real and feeds the next timestep. v0.1.0 deliberately uses a compact stop-gradient recurrent PPO update rather than full BPTT; see `docs/ARCHITECTURE.md`.

## Data integrity

Training seeds and held-out evaluation seeds use different deterministic seed domains. `UNSEEN TEST` never trains the policy. Neural graphics are driven by live model values; there are no random decorative activity pulses.

## iPhone notes

Canvas 2D is used instead of Three.js. Compute presets change training duty cycle. Start with **Balanced**; use **Eco** if Safari warms the phone during long runs.
