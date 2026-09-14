# MicroMind v0.1.0.1.2 — Recovery-Safe Save Slots Hotfix

**Build marker:** `SAVEREC-01012`

**Baseline:** exact v0.1.0.1.1 `HISTCONT-01011`, itself containing v0.1.0.1 stabilization and v0.1.0 foundation

## Why this update exists

Physical iPhone testing showed genuine learning but also a non-monotonic training failure: a ~500k-step historical brain outperformed the ~600k-step Latest policy on held-out worlds. the stabilization line treats that as a learning-system observation, not something to hide.

## Implemented

- automatic fixed-seed validation domain separate from training and manual held-out evaluation
- protected Best brain per curriculum validation protocol
- best snapshots include model + optimizer + curriculum state
- Latest can regress without destroying Best
- explicit **Restore Best**; never automatic rollback
- **Latest / Best** selection in Observe and Probe
- periodic validation plus validation on curriculum transitions
- automatic IndexedDB autosave after every automatic validation
- manual and validation-autosave slots are displayed and loaded explicitly; no newest-wins ambiguity
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

`npm test`: **26/26 PASS**

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
- checkpoint scheduling beyond 1M
- long-run migration without fabricated backfill

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
- PPO can still regress. v0.1.0.1.2 preserves and exposes the regression rather than claiming to eliminate catastrophic forgetting.
- Physical iPhone testing is required before this becomes the accepted baseline.


## v0.1.0.1.1 checkpoint-history correction

Physical testing also exposed that the historical milestone schedule stopped at 1,000,000 steps. Training itself did not stop; only historical policy capture did. This hotfix:

- continues milestone capture indefinitely using an adaptive sparse schedule
- prevents fake backfilling when loading an old run already beyond 1M
- captures the exact loaded legacy brain as a migration snapshot
- keeps early anchor brains visible while Compare Brains also shows recent milestones
- leaves model architecture, PPO hyperparameters, rewards, curriculum logic, validation logic, physics, and checkpoint schema unchanged


## v0.1.0.1.2 recovery correction

Physical testing showed a refresh could start a new in-memory brain while the older high-step Manual Save still existed in IndexedDB. The prior single **Load** action selected the newest timestamp, which could make a newer low-step validation autosave obscure the older mature Manual Save. This hotfix:

- exposes Manual Save and Validation Autosave separately
- shows each slot's exact training steps, episodes, schema, and save time before loading
- pauses training after any restore so the user can verify the recovered checkpoint
- prevents a lower-step current brain from overwriting a higher-step Manual Save
- leaves the IndexedDB database name, object store, keys (`latest`, `autosave`), checkpoint schema, model, PPO, reward system, curriculum, and physics unchanged
- includes the extended post-1M historical checkpoint schedule from v0.1.0.1.1

The hotfix cannot guarantee an old manual checkpoint exists until the deployed browser reads its IndexedDB. It is designed to inspect and recover it safely if it is still present.
