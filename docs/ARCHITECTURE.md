# Architecture

MicroMind is split into simulation, learning, evaluation, persistence, and visualization layers.

## Simulation

`World` owns continuous 2D motion, finite energy, food, walls, hazards, ray sensors, rewards, and procedural generation. All random generation uses a seeded PRNG.

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

### v0.1.0 recurrent limitation

For mobile simplicity the stored recurrent state is treated as a stop-gradient input during each PPO sample update. Recurrent weights are trainable, and state carries across timesteps, but gradients are not propagated backward through long sequences. A future milestone can add sequence minibatches and truncated BPTT after the baseline is physically validated.

## Seed separation

Training worlds use `train:<brain-seed>` seed domains. Evaluation uses `heldout:v1`. The domains hash to distinct deterministic sequences.

## Visualization data flow

The brain renderer reads the exact observation vector, current recurrent activations, policy probabilities, value estimate, and current parameter arrays from the active model. Connection brightness is based on actual activation × weight contribution; weight mode is based on actual learned weight magnitude/sign.
