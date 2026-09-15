# Roadmap

Current: **v0.1.3.1 — Curiosity Audit & Ablation**.

Immediate acceptance goal: answer whether v0.1.3 intrinsic reward improves external-task learning versus an otherwise matched observe-only control.

Required evidence:
1. current Learner/Champion/Hall survive migration;
2. CONTROL and CURIOSITY start from identical serialized state;
3. CONTROL predictor continues learning while applied intrinsic reward stays exactly zero;
4. both branches receive equal branch-local training budgets and PPO schedule age;
5. audit evaluations use the same read-only seeds and never promote Champions;
6. paired 0.5M-step checkpoints are inspected through the 2M/branch target;
7. confidence ranges and multi-checkpoint trend, not a single spike, determine the interpretation;
8. iPhone Safari remains stable and acceptably cool/fast.

Only after the ablation result:
- if curiosity helps, retain it and proceed toward a richer learned world model;
- if curiosity is neutral, decide whether its exploration value justifies complexity;
- if curiosity hurts, disable its reward influence while keeping the predictor as an observational/world-model foundation;
- do not retune curiosity and PPO simultaneously, because that would destroy causal clarity.

Future candidates remain world-model/predictive memory, primitive imagination/planning, richer delayed-memory tasks, and controlled policy-capacity experiments.
