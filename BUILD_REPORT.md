# MicroMind v0.1.3.1 — Build Report

**Milestone:** Curiosity Audit & Ablation  
**Build marker:** `CURAUD-0131`  
**Baseline:** exact packaged v0.1.3 `CURIOUS-013`  
**Baseline archive SHA-256:** `df786484b710936674436a61912d5914b0c9e154975d5a58922a35affd9ef19b`  
**Save schema:** 8 (loads schemas 1–8)

## Why this release exists

Physical multi-million-step testing after v0.1.3 showed that the curiosity predictor was mechanically active, but the continuing Learner did not clearly outperform the pre-curiosity Champion and showed substantial volatility/retention regression. v0.1.3.1 therefore changes the next milestone from “add more intelligence” to **measure whether curiosity reward is helping at all**.

This release deliberately does not retune the curiosity coefficient, episode cap, policy network, PPO hyperparameters, external reward, world rules, or rehearsal targets. Direct hash comparison against the packaged v0.1.3 baseline confirms `model.js`, `ppo.js`, `rollout.js`, `world.js`, `curriculum.js`, `evaluator.js`, and `curiosity.js` are unchanged.

## Controlled ablation

Added two curiosity influence modes:

- `reward`: predictor learns and bounded intrinsic reward is applied to PPO;
- `observe`: predictor learns and all diagnostics/shadow budget accounting continue, while applied PPO intrinsic reward is exactly zero.

Added a matched A/B audit that:

1. freezes the exact current Learner as a recoverable origin;
2. creates CONTROL and CURIOSITY descendants from identical serialized policy, optimizer, predictor, RNG, curriculum, and seed-cursor state;
3. freezes autonomous curriculum changes for the experiment;
4. evaluates both branches on `heldout:curiosity-ablation:v1` with identical protocol/seeds;
5. uses branch-local PPO schedule age so branch order cannot change annealing behavior;
6. isolates each branch's recent episode-return window;
7. suspends ordinary historical milestones and Champion promotion while the audit is active;
8. never auto-promotes, auto-restores, or declares a policy winner.

Default audit budget: **2,000,000 branch-local steps per branch**, read-only evaluation every **500,000 branch-local steps**.

## Curiosity instrumentation

Added episode-level accounting for:

- external reward;
- potential and applied intrinsic reward;
- reward-magnitude share;
- prediction-error mean/max;
- novelty mean/max;
- curiosity samples;
- curiosity budget used;
- budget exhaustion occurrence and approximate episode fraction;
- budget reset count.

The UI now distinguishes **potential bonus** from **applied bonus** and identifies reward influence as Reward ON or Observe only.

### Budget-label correction

The v0.1.3 source initializes `curiosityEpisodeBudget` to `0.25` and decrements it as intrinsic bonus is spent. Therefore a display of `0.250 / 0.25` means **full budget remaining**, not “budget exhausted.” v0.1.3.1 corrects the label to `BUDGET REMAINING` and adds direct episode accounting so no visual guess is required.

## Evaluation isolation

Curiosity reward remains absent from:

- `validation:v3` Champion selection;
- `heldout:compare:v2` historical comparison;
- `heldout:final:v2` final Unseen Test;
- new `heldout:curiosity-ablation:v1` A/B evaluation.

A/B audit evaluation is observational and cannot mutate the Champion archive.

## Persistence / recovery

Schema 8 adds persistence for:

- curiosity influence mode;
- curiosity episode history/reset counters;
- active A/B experiment state/results;
- branch role and branch-local audit progress;
- branch-specific recent episode history.

Schema-7 v0.1.3 checkpoints migrate safely to reward-on/no-audit behavior. The current Learner is preserved before starting an audit, and the UI also saves the pre-audit state before creating descendants.

## Automated QA

- `npm test`: **65/65 PASS**.
- Observe-only predictor learning with exactly zero applied intrinsic reward: PASS.
- Budget remaining/reset semantics: PASS.
- Matched A/B descendant creation: PASS.
- Champion non-promotion during audit: PASS.
- Branch-specific reward mode/progress restore: PASS.
- Equal PPO schedule age at equal branch-local progress: PASS.
- Ordinary milestone and recent-return-window isolation: PASS.
- Schema-8 active-audit roundtrip + schema-7 migration: PASS.
- Existing PPO, Hall, Champion, continual-learning, evaluation-isolation, and final-heldout tests retained.
- Built `dist/` HTTP resource smoke: PASS.
- Full 100k+100k branch A/B execution smoke reached both targets and produced paired checkpoints: PASS (scores are not treated as scientific evidence).

## Development performance smoke

Headless Node performance workload after the audit changes:

- wall-clock throughput: **45,824 steps/sec**;
- last-profile simulation throughput: **142,403 steps/sec**;
- last-profile PPO time: **3.47 ms**;
- last-profile curiosity update time: **0.20 ms**.

This is a development-machine smoke test, not an iPhone claim. Physical Safari remains the performance authority.

## Physical test procedure

1. Deploy and verify `v0.1.3.1 • CURAUD-0131`.
2. Load the intended current ~15M Learner and confirm the protected 6,050,304-step Champion/Hall entry still exists.
3. Stay paused and confirm the Curiosity panel says `BUDGET REMAINING`.
4. Press **Start Curiosity A/B Audit**. The exact current Learner is preserved and CONTROL becomes active first.
5. Run CONTROL to its 2M branch-local target. It auto-pauses at completion; audit checkpoints appear at ~0.5M intervals.
6. Switch to CURIOSITY and run the same 2M branch-local budget.
7. Compare paired A/B results. Do not promote a winner from one noisy checkpoint; inspect the trajectory across paired checkpoints and confidence ranges.
8. End the audit only after preserving the result you want to inspect. Ending does not automatically replace the Champion.
