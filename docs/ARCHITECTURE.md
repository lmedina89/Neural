# Architecture — MicroMind v0.1.0.1.1

MicroMind is split into simulation, learning, evaluation, persistence, and visualization layers.

## Simulation

`World` owns continuous 2D motion, finite energy, food, walls, hazards, ray sensors, rewards, and procedural generation. All random generation uses a seeded PRNG. Episode info now carries the curriculum stage identity so stale environments finishing after a stage transition cannot incorrectly vote on the new stage.

## Observation vector

1. egocentric food X
2. egocentric food Y
3. food distance
4. left danger ray
5. forward danger ray
6. right danger ray
7. speed
8. angular velocity
9. energy
10. constant bias/context channel

## Policy/value model

A 24-unit tanh recurrent state receives the observation and previous recurrent state. It feeds a 7-action categorical policy and scalar value head. This is intentionally small enough to visualize.

## PPO

Rollouts retain observation, previous recurrent state, action, old log probability, reward, done flag, and value. GAE computes advantages/returns. PPO uses clipped policy loss, entropy regularization, value loss, Adam, minibatches, and gradient clipping.

### Recurrent limitation retained intentionally

The stored recurrent state is treated as a stop-gradient input during each PPO sample update. Recurrent weights are trainable and state carries across timesteps, but gradients are not propagated backward through long sequences. v0.1.0.1.1 deliberately does not alter this while solving policy-stability and checkpoint-protection issues. Sequence minibatches/truncated BPTT remain a later research milestone.

## Seed domains

The project now uses distinct deterministic domains for different purposes:

- training: `train:<brain-seed>`
- automatic validation: `validation:v1`
- manual held-out evaluation: `heldout:v1`
- observation/demo worlds: `observe:v2`

Validation and held-out worlds are never inserted into PPO rollouts.

## Automatic validation and best-brain protection

`TrainingSession.runValidation()` evaluates a fixed curriculum suite from stage 0 through the current stage. Each curriculum depth has a distinct protocol identity. The best policy for each protocol is retained in `bestBrains`, so a temporary curriculum demotion does not erase an older best from another protocol.

A validation record stores:

- validation score (`meanReturn` across the suite)
- food, survival, energy, hazard/wall metrics
- per-stage results
- source training step
- whether it improved the protected best
- whether it crossed the regression threshold

The current protected best additionally stores model parameters, optimizer state, and curriculum state. Later policies may become worse; they do not overwrite the protected snapshot.

Validation runs around 10k, 50k, 100k, 250k, 500k, 750k, 1M steps and every 250k thereafter, plus curriculum transitions. Exact execution can occur just after a threshold because PPO rollouts advance in batches.

When migrating a v0.1.0 save, the first resumed training call validates the existing policy before further experience can change the curriculum. This is specifically intended to protect long-running v0.1.0 experiments.

## Curriculum hysteresis

The curriculum uses a bounded rolling score based on food acquisition and survival. Promotion and demotion thresholds are intentionally separated. After any transition, a cooldown prevents immediate stage oscillation.

Only episodes actually generated under the current curriculum stage are allowed to influence the stage manager.

## Persistence

Checkpoint schema 2 adds:

- protected best brains by validation protocol
- validation history
- next/last validation step
- explicit best-restore history
- curriculum transition/cooldown state

Schema 1 from v0.1.0 is still accepted.

Manual saves use IndexedDB id `latest`. Automatic validation autosaves use id `autosave`. The Load control chooses whichever record has the newest timestamp.

## Visualization data flow

The brain renderer reads the exact observation vector, current recurrent activations, policy probabilities, value estimate, and parameter arrays from the selected model. Connection brightness is based on actual activation × weight contribution; weight mode is based on actual learned weight magnitude/sign.

LEARN always displays and trains **Latest**. OBSERVE and PROBE can switch between **Latest** and the current protocol's protected **Best**.

The training chart adds curriculum transition lines and validation markers. No visualization value is substituted for model/runtime state.
