# MicroMind v0.1.1 — Learning Stability & Skill Retention

**Build:** `STABRET-011`

MicroMind is a browser-based reinforcement-learning laboratory built around a deliberately tiny, inspectable recurrent actor-critic. v0.1.1 keeps the same 1,040-parameter brain and the same physical world from the accepted v0.1.0.1.2 baseline. The milestone targets a problem observed during real iPhone training: a useful policy could learn a strong foraging/survival strategy and then partially destroy it tens of thousands of PPO steps later.

## What changed

### Fixed all-skills validation
Automatic validation no longer changes its definition when the curriculum changes. Every validation evaluates the current policy on all four fixed curriculum stages:

- Motor Nursery
- Foraging
- Obstacle Avoidance
- Scarcity

Each stage produces a bounded skill-retention score. This makes earlier competence measurable after the curriculum advances.

### Catastrophic-forgetting detection
MicroMind tracks the strongest validated score previously seen for each skill. A sufficiently large drop from a previously competent skill is reported as catastrophic forgetting.

Curriculum promotion is gated: a learner cannot advance just because its current-stage episode score is high if earlier validated skills have collapsed.

### Protected specialist archive
Instead of one scalar Best brain, v0.1.1 maintains separate protected candidates for:

- Best Balanced
- Best Overall return
- Best Forager
- Best Survivor
- Best Efficiency

Observe and Probe can inspect any available protected specialist. Manual restoration is still available.

### Guarded PPO
The PPO update path now includes:

- a smaller clipping range;
- a lower conservative learning-rate range;
- target-KL epoch early stopping;
- hard-KL update rejection with exact parameter/optimizer rollback;
- adaptive learning rate;
- learning-rate recovery cooldown after a validation rollback;
- training-step entropy schedule;
- adaptive entropy rescue when the action policy becomes too concentrated.

### Validation recovery guard
If the fixed skill suite detects either a severe balanced-policy regression or catastrophic forgetting, the trainable model is automatically restored from Best Balanced. The experience counter is **not** rewound. The recovery event is recorded and the learning rate is reduced.

For the next recovery period, validation temporarily runs every 25,000 experience steps so another collapse is caught sooner.

This is intentionally different from the old `Restore Best` button: the automatic guard is a safety rail triggered only by the objective retention protocol. The manual restore button remains available for deliberate experimentation.

## Existing saves

v0.1.1 uses checkpoint schema 3 and accepts schema 1, 2, and 3.

When loading a v0.1.0.1.x/schema-2 session:

1. the current Latest model, total experience counter, optimizer, curriculum and milestone history are preserved;
2. the old protected Best is preserved as a migration candidate;
3. training pauses after explicit load, as before;
4. the first v0.1.1 validation re-evaluates Latest and the legacy Best on the new fixed all-skills protocol;
5. no old score is compared numerically against a new incompatible score;
6. subsequent saves use schema 3.

The recovery-safe Manual Save / Validation Autosave split and lower-step overwrite guard remain intact.

## Recommended upgrade from v0.1.0.1.2

Before replacing the deployed files, press **Save Manual** in the currently running build.

After deploying v0.1.1:

1. refresh the page;
2. verify the Manual Save step count;
3. press **Load Manual**;
4. confirm the expected total step count and that training is paused;
5. press **Resume**;
6. wait for the immediate migration/rebaseline validation;
7. inspect the new Skill Retention panel and protected archive entries;
8. press **Save Manual** after the v0.1.1 validation if you want a schema-3 manual checkpoint immediately.

Do not clear Safari website data between versions; checkpoints live in IndexedDB.

## Run locally

```bash
npm test
npm run build
python3 -m http.server 8080 --directory dist
```

Then open `http://localhost:8080/`.

## Benchmark

```bash
npm run benchmark
STEPS=150000 npm run benchmark
```

The benchmark uses a fixed training seed and a separate held-out seed domain. See `docs/BENCHMARK.md` and `BUILD_REPORT.md` for the release run.

## Scope boundary

v0.1.1 deliberately does **not** add a larger brain, world model, curiosity, imagined rollouts, language, or multi-agent behavior. The point of this milestone is to make the existing learner more trustworthy before adding cognitive complexity.
