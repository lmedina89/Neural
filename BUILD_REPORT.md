# MicroMind v0.1.0.1 — Learning Stability & Best-Brain Protection

**Build marker:** `LEARNSTAB-0101`

**Baseline:** v0.1.0 `LEARNLAB-010`

## Why this update exists

Physical iPhone testing showed genuine learning but also a non-monotonic training failure: a ~500k-step historical brain outperformed the ~600k-step Latest policy on held-out worlds. v0.1.0.1 treats that as a learning-system observation, not something to hide.

## Implemented

- automatic fixed-seed validation domain separate from training and manual held-out evaluation
- protected Best brain per curriculum validation protocol
- best snapshots include model + optimizer + curriculum state
- Latest can regress without destroying Best
- explicit **Restore Best**; never automatic rollback
- **Latest / Best** selection in Observe and Probe
- periodic validation plus validation on curriculum transitions
- automatic IndexedDB autosave after every automatic validation
- Load chooses the newest manual or validation autosave
- curriculum demotion as well as promotion
- promotion/demotion hysteresis and transition cooldown
- stale episodes from old curriculum stages cannot vote on a new stage
- schema-1 v0.1.0 checkpoint migration
- migrated legacy run validates its existing policy before resumed training can change curriculum
- built-in control/metric help for mobile
- chart markers for curriculum changes and validation events
- increased manual Unseen Test to 32 held-out episodes
- historical comparison uses 24 identical held-out episodes per brain
- core neural architecture and PPO hyperparameters intentionally unchanged

## Automated verification

`npm test`: **21/21 PASS**

Coverage includes:

- deterministic PRNG/worlds
- seed-domain separation
- observation/action/reward finiteness
- GAE
- model roundtrip
- real parameter updates
- curriculum promotion + demotion
- validation repeatability
- best-brain preservation
- regression detection
- explicit restore-best rollback
- schema-2 restore
- schema-1 migration
- legacy long-run pre-resume validation
- UI control presence

All JS/MJS files pass `node --check`.

Static production build succeeds.

## Controlled benchmark

At 60,160 training steps with seed `424242`:

- initial held-out return: `-9.674`
- Latest held-out return: `-0.761`
- Protected Best held-out return: `1.211`
- Protected Best source step: `50,176`
- Protected Best food: `1.344`
- Protected Best survival: `78.1%`

The benchmark intentionally encountered late-policy regression and verified that Best retained the better earlier policy. See `docs/BENCHMARK.md`.

## Compatibility / migration

Checkpoint schema advances from 1 to 2 internally. v0.1.0 schema-1 saves remain loadable. The IndexedDB database/store names are unchanged.

**Important physical-upgrade step:** if the current v0.1.0 training run is still only in memory, press **Save** before replacing the deployed files. Deploying/reloading cannot recover an unsaved in-memory model from the old page.

## Known limitations

- Recurrent state still uses stop-gradient sample updates rather than truncated BPTT.
- Validation is a small fixed suite; Best means “best under this protocol,” not universally best.
- Automatic validation is synchronous and can cause a brief training/UI pause at infrequent validation points; physical iPhone timing should be checked.
- PPO can still regress. v0.1.0.1 preserves and exposes the regression rather than claiming to eliminate catastrophic forgetting.
- Physical iPhone testing is required before this becomes the accepted baseline.
