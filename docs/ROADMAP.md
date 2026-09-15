# Roadmap

Current: **v0.1.3 — Intrinsic Curiosity Foundation**.

Physical acceptance goals:
1. migrate the current v0.1.2.1 run without losing its Learner/Champions/Hall;
2. verify the curiosity predictor begins fresh on schema-6 migration and persists after schema-7 save/reload;
3. verify prediction error/loss generally decreases for familiar experience;
4. verify novelty visibly rises on less predictable transitions;
5. verify intrinsic reward stays small and never appears in validation or final holdout scoring;
6. verify no obvious curiosity reward hacking (spinning, wall-seeking, intentional death);
7. verify iPhone Safari throughput remains acceptable relative to v0.1.2.1;
8. compare curiosity-trained Learner/Champions against pinned pre-curiosity policy references using the unchanged external evaluation protocols.

After acceptance:
1. **v0.1.4 — learned world model / predictive memory**: extend one-step prediction into a richer latent dynamics model;
2. **v0.1.5 — imagination / primitive planning**: score short model-generated futures before acting;
3. richer hidden-rule and delayed-memory tasks;
4. controlled policy-capacity experiment (1,040 vs ~4k vs ~12k parameters) if evidence suggests a representational ceiling;
5. recurrent-PPO modernization as a separate controlled milestone rather than mixing it with curiosity/world-model changes.
