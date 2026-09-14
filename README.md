# MicroMind v0.1.2 — Autonomous Continual Learning

MicroMind is a browser-based miniature AI research lab. A real 1,040-parameter recurrent actor-critic learns with PPO in procedural worlds while its activations, decisions, memory, rewards and validation are visible.

## What changed in v0.1.2
The active **Learner** now develops continuously. Behavioral regression is measured but never automatically rewound. Frozen **Champions** preserve demonstrated policies. Earlier skills stay in the training stream through continual rehearsal, and a challenger must outperform a Champion on repeated validation checks before promotion.

The final held-out suite is diagnostic only and never trains, promotes, restores or otherwise changes a policy.

## Controls
- **Learn:** train the autonomous Learner.
- **Observe / Probe:** inspect Learner or a frozen Champion.
- **View Brain:** select Learner or Champion specialists.
- **Fork From Champion:** optional manual branch from a Champion; never automatic.
- **Unseen Test:** final all-skills diagnostic only.
- **Compare Brains:** historical comparison on a separate held-out domain.

## Rehearsal
At later curricula, a fraction of new training episodes comes from earlier skills. At Scarcity the target mix is approximately 10% Motor, 15% Foraging, 20% Obstacle Avoidance and 55% Scarcity.

## Run
```bash
npm test
npm run build
python3 -m http.server 8080 --directory dist
```
Then open `http://localhost:8080/`.

GitHub Pages can serve the repository root directly.
