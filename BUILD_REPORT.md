# MicroMind v0.1.2 — Build Report

**Build marker:** `AUTOCONT-012`  
**Baseline:** exact v0.1.1.1 `VALCAL-0111` package  
**Save schema:** 5 (loads schemas 1–5)

## Scope
v0.1.2 changes the training philosophy from behavioral rollback to autonomous continual learning. The 1,040-parameter recurrent actor-critic is unchanged.

### Implemented
- **Learner / Champion separation:** the active Learner is never automatically replaced for behavioral regression. Frozen Champions preserve repeatedly validated policies.
- **Continual rehearsal:** training episodes are interleaved across previously learned skills. Target mixes by current curriculum stage are 100/0/0/0, 25/75/0/0, 15/20/65/0, and 10/15/20/55 percent for Motor/Foraging/Obstacle/Scarcity.
- **Repeat-confirmed Champion promotion:** after an initial Champion exists, a challenger must beat it on two consecutive validation checks before promotion.
- **Observational retention:** forgetting/regression remains visible and can schedule an earlier re-check, but does not restore weights or alter the Learner.
- **Numerical safety retained:** PPO hard-KL rejection still restores a mathematically destructive update; this is optimizer safety, not behavioral selection.
- **Lineage tracking:** Champion snapshots record Learner lineage. Manual `Fork From Champion` creates a new explicit lineage; it is never automatic.
- **UI diagnostics:** Learner lineage, observed rehearsal mix and pending Champion challenges are visible.
- Final holdout remains diagnostic-only and separate from training/validation.

## Automated QA
- `npm test`: **40/40 PASS**
- JS/MJS syntax checks: PASS
- Static production build: PASS
- Clean-unzip test/build: PASS
- Static HTTP resource smoke: PASS
- ZIP integrity: PASS

## Controlled benchmark
120,064 experience steps, hardest-stage training with continual rehearsal enabled.
- Initial all-skills score: **0.5%**
- Learner at end: **33.4%**
- Repeat-confirmed Balanced Champion @ 100,096: **34.0%** on the benchmark suite
- Observed rehearsal mix over final 160 episodes: **9.4% / 15.0% / 18.8% / 56.9%**, close to the 10/15/20/55 target.
- Behavioral automatic rollbacks: **0**

## Physical iPhone gate
Load the existing v0.1.1.1 Manual Save, confirm the multi-million-step Learner and Champions survive schema-5 migration, then Resume. Verify:
1. no automatic behavioral rollback during a regression;
2. `rehearsal mix` includes earlier skills as curriculum advances;
3. a superior Learner shows `Champion challenge 1/2` before promotion;
4. only a second qualifying validation promotes it;
5. final Unseen Test does not change Learner/Champion state.
