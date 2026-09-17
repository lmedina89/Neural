# Roadmap

Current diagnostic milestone: **v0.1.4.1.1 — Rear-Target & Rotation Audit**.

Immediate acceptance goal:
1. deploy on top of v0.1.4.1 without changing the saved Learner;
2. load the long-running Learner and run the fixed rear-target audit;
3. compare Learner and Balanced Champion at the same target bearings;
4. determine whether the failure is dominated by rear-bearing action selection, angular overshoot/spin, BRAKE use while rotating, or a broader inability to orient;
5. preserve the current brain while collecting this evidence;
6. only then select one targeted mechanism for v0.1.4.2.

The v0.1.4.1 **Validation Confidence & True Regression Audit** remains intact underneath this diagnostic build. Its longer-run goal remains to separate noisy validation swings from defensible continual-learning regression evidence.

Next: **v0.1.4.2 — Targeted Continual-Learning / Turn-Control Fix**. Change exactly one mechanism supported by the collected evidence, not several systems at once.

After stability is understood: **v0.1.5.0 — World Model / Imagination Foundation**, followed by controlled imagination/planning and richer memory work.
