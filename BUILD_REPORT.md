# MicroMind v0.1.0 — Build Report

**Build marker:** `LEARNLAB-010`

## Implemented

- deterministic procedural 2D training worlds
- separate training and held-out seed domains
- continuous inertial movement, food, hazards, walls, energy and ray sensors
- real compact recurrent actor-critic (`10 → 24 recurrent → 7 policy + value`)
- PPO clipped objective, GAE, entropy regularization, value loss, minibatch Adam and gradient clipping
- LEARN / OBSERVE / PROBE modes
- live Canvas 2D neural instrumentation using actual activations, parameters and policy outputs
- neuron touch inspection
- decomposed reward display
- historical in-session milestone brains
- identical held-out checkpoint comparison
- IndexedDB manual save/load including optimizer state
- bounded training chart
- Eco / Balanced / Max training duty-cycle presets
- static GitHub Pages compatible root and `dist/` build

## Verification

`npm run check` passes 10/10 automated tests and builds `dist/`.

Controlled Stage-0 benchmark at 40,192 training steps:

- initial held-out mean return: `-9.775`
- final held-out mean return: `2.189`
- initial held-out food: `0.083`
- final held-out food: `2.292`

See `docs/BENCHMARK.md`.

## Known limitations

- Recurrent state is real and trainable, but v0.1.0 uses stored recurrent state as a stop-gradient PPO input rather than full sequence BPTT. This was chosen deliberately for a smaller, auditable, mobile-friendly first baseline.
- Physical iPhone Safari thermal behavior has not yet been measured. Start on Balanced; switch to Eco if the device warms excessively.
- The curriculum, reward shaping and longer-duration policy stability need physical/extended testing before adding curiosity or a learned world model.
- The neural graph is dense but intentionally capped/thresholded for readability and mobile rendering cost.
