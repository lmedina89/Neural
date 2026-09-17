# MicroMind roadmap

Current diagnostic milestone: **v0.1.4.1.3 — Occlusion Failure Characterization**.

The goal of this milestone is to establish whether the loaded long-running policy has any genuine learned detour/navigation strategy when the food vector points through an obstacle, and to distinguish sustained one-way spinning from turn/reverse oscillation.

## Decision gate before v0.1.4.2

Do not change learning yet. Accept v0.1.4.1.3 only after physical testing provides:

- paired results across OPEN, CLEARANCE, NARROW/WIDE centered blockers, LEFT/RIGHT-heavy blockers, and LONG DETOUR;
- preferably several live OBSERVE rotation-trap captures from the real procedural world;
- confirmation that the accepted Learner/Champion save loads normally and ordinary learning behavior remains intact.

Then v0.1.4.2 should make **one targeted continual-learning change** supported by the evidence. Candidate directions remain deliberately unresolved until the audit distinguishes among policy detour learning, recurrent/temporal learning, reward/curriculum exposure, or angular-control conflict.

After that targeted stability/navigation work is validated, the broader roadmap can proceed toward the planned **v0.1.5.0 World Model / Imagination Foundation**.
