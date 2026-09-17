# MicroMind v0.1.4.1.2 — Live Occlusion & Spin-Cause Telemetry

**Build:** `OCCSPIN-01412`  
**Parent:** v0.1.4.1.1 `ROTAUD-01411`

This is another deliberately narrow **observational** build. The accepted rear-target audit showed that the long-running Learner and Balanced Champion can both turn to and reach targets behind them in an empty arena with 0% detected spin. That rules out “food is behind me” as the simple cause of the live-world behavior.

The remaining observation to test is the one seen on the phone: MicroMind can start spinning when the target does not have a clean/direct route or visual line through the world geometry.

## What is new

Research → **LIVE OCCLUSION / SPIN TELEMETRY** adds two read-only tools.

### 1. Live OBSERVE recorder

The recorder arms automatically whenever **OBSERVE** mode is entered. It watches the real selected policy in the normal procedural Observe world and keeps only a small session-memory buffer.

For each step it records diagnostic-only context such as:

- nearest-food bearing and distance;
- whether a wall geometrically crosses the center-to-center food line (`LOS BLOCKED`);
- whether the agent-radius travel corridor to food is blocked even when the centerline stays clear (`PATH BLOCKED`);
- a diagnostic ±70° forward cone (not a new sensor);
- left/front/right danger-ray values;
- angular velocity and linear speed;
- sampled action, policy probabilities and value;
- recurrent hidden-state magnitude/change;
- reward components and food-distance progress;
- target switches.

A potential spin event is captured only after substantial rolling angular travel **and** poor progress toward food. The event stores the lead-up plus a short recovery window, then assigns a descriptive context such as `wall/food-conflict`, `clearance/danger-conflict`, `outside-forward-cone`, `angular-momentum`, or `target-switch`.

Those labels are diagnostic classifications only. They do not tell the policy what happened and they do not change behavior.

### 2. Controlled occlusion audit

**Run Controlled Occlusion Audit** pauses training and tests the exact Learner in three paired geometries:

- **OPEN** — no wall;
- **CLEARANCE** — food centerline remains visible, but a wall intersects the agent-radius direct travel corridor;
- **OCCLUDED** — the wall crosses the food centerline itself.

Each case uses the same start pose, food position, zero recurrent state and paired stochastic-action RNG seed for the corresponding trial. The audit reports reach rate, spin incidence, time to food, rolling/total angular travel, food-distance progress and wall hits. A Balanced Champion is tested too when available.

## Important interpretation

`diag LOS BLOCKED`, `diag PATH BLOCKED`, and the forward-cone flag are **external measurements only**. The neural network still receives the exact same 10-value observation as before. Nearest-food relative X/Y and distance remain present even through walls. No occlusion bit or “vision” bit has been added to the brain.

## What did not change

No changes were made to:

- policy architecture, weights or recurrent dynamics;
- PPO math or hyperparameters;
- curiosity model or intrinsic reward;
- curriculum or rehearsal;
- world physics or collisions;
- observation vector or three danger rays;
- rewards;
- spawn clearance;
- Champion promotion / Hall of Fame;
- validation-confidence logic;
- checkpoint/save schema (still **10**);
- final held-out evaluation.

Live telemetry and controlled-audit results are session-only and disappear on reload.

## iPhone acceptance test

1. Deploy and confirm **`v0.1.4.1.2 • OCCSPIN-01412`**.
2. Load the intended long-running Manual Save and confirm the Learner lineage/step count and Champion are correct.
3. In **RESEARCH → LIVE OCCLUSION / SPIN TELEMETRY**, tap **Run Controlled Occlusion Audit** once and screenshot the Learner and Champion tables.
4. Switch to **OBSERVE**. The recorder automatically shows `ARMED`; no extra switch is required.
5. Watch normal behavior. The LIVE world header now says `diag DIRECT`, `diag PATH BLOCKED`, or `diag LOS BLOCKED` for the current nearest food. This is only a diagnostic label.
6. When the spinning behavior occurs, let it continue long enough for the recorder to capture the event and recovery window.
7. Return to **RESEARCH → LIVE OCCLUSION / SPIN TELEMETRY** and screenshot the event table. Several events are even better than one.
8. Do **not** change sensors, rewards, PPO, recurrent training, BRAKE physics or turn physics yet. v0.1.4.2 should change exactly one mechanism supported by this evidence.
