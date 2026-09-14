# MicroMind v0.1.0.1.1 — Extended Checkpoint History Hotfix

**Build:** `HISTCONT-01011`

MicroMind is a browser-based miniature reinforcement-learning laboratory. A small recurrent actor-critic learns to forage, conserve energy, and avoid hazards in deterministic procedurally generated worlds. The world, neural activations, learned weights, policy probabilities, recurrent state, value estimate, rewards, validation history, and historical brains are exposed for inspection.

v0.1.0.1.1 contains the full v0.1.0.1 stabilization update plus a narrowly scoped historical-checkpoint continuation hotfix. It is built for direct upgrade from v0.1.0. It does not add curiosity or a world model. It protects good policies from being lost silently during later PPO training and makes the controls easier to understand on mobile.

## Run

This project has no runtime dependencies.

```bash
npm test
npm run build
python3 -m http.server 8080
```

Open `http://localhost:8080`. `dist/` is also a static GitHub-Pages-ready build.

## Modes

- **LEARN** — trains the Latest brain with PPO using multiple headless environments. Rendering is decoupled from training.
- **OBSERVE** — watches either **Latest** or the protected **Best** brain in a separate procedural world.
- **PROBE** — freezes movement and lets you reposition food, a hazard, or the agent. Policy outputs update from the selected real model without taking an action.

## Stability protection

Automatic validation uses a deterministic seed domain named `validation:v1`, separate from both training and the manual held-out test domain. Validation runs at selected training milestones and on curriculum transitions.

The best validation result for each curriculum protocol is preserved with:

- model parameters
- optimizer state
- curriculum state
- validation metrics
- source training step

A worse Latest policy never overwrites that protected brain. **Restore Best** is explicit and never automatic.

Each automatic validation also writes an `autosave` checkpoint to IndexedDB. **Load** chooses the newest of the manual `latest` save and the validation autosave.

## Curriculum stability

Curriculum changes now use hysteresis:

- sustained performance is required to promote
- sustained collapse can demote the curriculum
- a cooldown prevents immediate oscillation after a stage change
- v0.1.0 saves receive a cooldown on migration before demotion is allowed

## Model

`10 observations → 24 tanh recurrent units → 7-action policy + scalar value`

The recurrent state is real and feeds the next timestep. v0.1.0.1.1 deliberately preserves the compact stop-gradient recurrent PPO baseline rather than changing the learning algorithm at the same time as stabilization. See `docs/ARCHITECTURE.md`.

## Data integrity

Training, validation, observation, and held-out evaluation use separate deterministic seed domains. `UNSEEN TEST` never trains the policy. Neural graphics are driven by live model values; there are no random decorative activity pulses.

## iPhone notes

Canvas 2D is used instead of Three.js. Compute presets change training duty cycle. Start with **Balanced**; use **Eco** if Safari warms the phone during long runs. Automatic validation is intentionally infrequent and bounded.

### Upgrading from v0.1.0

Before replacing the old GitHub Pages build, press **Save** in the currently open v0.1.0 session if you want to preserve that in-memory run. v0.1.0.1.1 can load the existing schema-1 IndexedDB checkpoint and migrates it in memory without rewriting the old record until the next save/autosave.


## Historical checkpoint continuation

The original v0.1.0 milestone list ended at 1,000,000 steps even though training continued. v0.1.0.1.1 removes that ceiling. Historical brains are now captured on an adaptive schedule:

- fixed early milestones through 1M
- every 250k from 1M to 5M
- every 500k from 5M to 20M
- every 1M from 20M to 100M
- every 5M beyond 100M

When a v0.1.0 save already beyond 1M is loaded, MicroMind does **not** fabricate missed checkpoints using the current brain. It records one honest migration snapshot at the exact loaded step count, then continues from the next future milestone. Compare Brains preserves early anchors while also showing recent historical brains, Latest, and Protected Best.
