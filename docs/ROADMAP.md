# MicroMind Roadmap

## Completed foundation

- genuine recurrent actor-critic training
- seeded procedural worlds
- real neural activation/weight visualization
- historical brains beyond 1M steps
- recovery-safe Manual/Autosave slots
- guarded PPO / KL rollback
- fixed all-skills validation
- protected multi-objective specialist archive
- v0.1.1 skill-retention instrumentation
- v0.1.1.1 confidence-calibrated validation
- repeat-confirmed regression handling
- distinct comparison and final-holdout domains
- validation/holdout conflict diagnostics

## Physical acceptance gate for v0.1.1.1

Before another cognitive feature is added, verify on the long-running iPhone session that:

1. schema-3 v0.1.1 Manual Save loads at the correct step age;
2. old specialists remain visible while calibration is pending;
3. Restore Selected is disabled until recalibration finishes;
4. first Resume recalibrates all unique protected candidates before training advances;
5. Skill Retention shows ranges and WATCH/CONFIRMED states correctly;
6. a single watch does not roll back the model;
7. repeated destructive aggregate evidence still triggers guarded recovery;
8. Unseen Test covers all four skills and reports final-holdout conflict without changing archives;
9. Compare Brains explicitly uses the non-final comparison domain;
10. Safari remains responsive and thermally reasonable in Balanced mode.

## Next candidate milestone after acceptance

The next research step should attack continual learning **inside** the learner rather than adding more rollback logic.

Candidates, in preferred order:

1. rehearsal/interleaved old-skill experience during later curricula;
2. teacher-policy distillation or retention regularization;
3. sequence minibatches / truncated BPTT or a true GRU;
4. dedicated delayed-memory tasks;
5. intrinsic curiosity;
6. learned world model and only then imagined rollouts/planning.

Network size should remain secondary to measurable retention and generalization.
