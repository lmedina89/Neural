# MicroMind v0.1.3.1 — Curiosity Audit & Ablation

MicroMind is a browser-based miniature AI research lab. A real **1,040-parameter recurrent actor-critic** learns with PPO in procedural worlds while its activations, decisions, memory, rewards, rehearsal, Champions, and generalization are inspectable.

v0.1.3.1 does **not** add another intelligence feature. It turns the v0.1.3 curiosity layer into a controlled experiment so we can determine whether the intrinsic reward actually helps the policy.

**Build:** `CURAUD-0131`

## What stays fixed

The v0.1.3 learning problem remains the control:

- policy architecture: **1,040 parameters**;
- curiosity forward predictor: **441 parameters**;
- PPO math/hyperparameters and external rewards unchanged;
- observations/actions, physics, world generation, and rehearsal targets unchanged;
- protected Champion/Hall archive unchanged;
- validation, historical comparison, and final unseen evaluation remain curiosity-free.

## New: Observe-only curiosity

Curiosity can now run in two influence modes:

- **Reward ON** — v0.1.3 behavior: the predictor learns and its bounded intrinsic bonus is added to PPO training reward.
- **Observe only (reward 0)** — the predictor still learns, measures prediction error/novelty, and performs shadow budget accounting, but PPO receives **exactly zero** intrinsic reward.

This separates “the predictor works” from “the predictor improves behavior.”

## New: matched A/B audit

**Start Curiosity A/B Audit** freezes the exact current Learner as a recoverable origin and creates two descendants from the same policy weights, optimizer state, curiosity predictor, RNG state, curriculum state, and environment-seed cursor:

- **CONTROL** — curiosity predictor learns, intrinsic reward influence = 0;
- **CURIOSITY** — current v0.1.3 intrinsic reward behavior.

The audit freezes automatic curriculum movement, uses a separate read-only seed domain (`heldout:curiosity-ablation:v1`), evaluates every **500,000 branch-local steps**, and targets **2,000,000 steps per branch** by default.

For experimental fairness, both branches use the same PPO schedule age at equal branch-local progress. Ordinary milestone saves and Champion promotion are suspended during the audit. **No A/B winner is automatically promoted or restored.**

## Curiosity accounting

The panel now exposes:

- potential curiosity bonus;
- applied curiosity bonus;
- curiosity share of reward magnitude;
- predictor error/loss and novelty;
- budget reset count;
- mean budget use per completed episode;
- budget-exhaustion rate and approximate exhaustion point;
- matched Control/Curiosity progress and paired evaluation results.

Important correction from v0.1.3 UI wording: **`0.250 / 0.25` is budget remaining, not budget consumed.** A full `0.250` therefore means the episode has not spent its curiosity allowance yet. v0.1.3.1 labels this explicitly as **BUDGET REMAINING**.

## Persistence

v0.1.3.1 uses **checkpoint schema 8** and loads schemas 1–8. Schema 7 v0.1.3 saves migrate with curiosity reward ON and no active audit. Active A/B state, branch progress, branch-specific episode windows, predictor state, and reward-influence mode survive schema-8 save/reload.

## Run

```bash
npm test
npm run performance
npm run build
python3 -m http.server 8080 --directory dist
```

Then open `http://localhost:8080/`.

GitHub Pages can also serve the repository root directly.
