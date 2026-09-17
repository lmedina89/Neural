import { ACTIONS, CONFIG } from '../config.js';
import { World } from '../sim/world.js';
import { PRNG, domainSeed } from '../utils/prng.js';

const TAU = Math.PI * 2;
const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;

// Read-only diagnostic protocol. These constants intentionally live outside the
// training configuration so changing or running this audit cannot alter PPO,
// rewards, curriculum, world generation, or checkpoint behavior.
export const ORIENTATION_AUDIT = Object.freeze({
  protocol: 'orientation-audit:v1',
  bearingsDeg: Object.freeze([-179, -135, -90, -45, 0, 45, 90, 135, 179]),
  trialsPerBearing: 8,
  maxSteps: 180,
  targetDistance: 0.24,
  facingThresholdDeg: 15,
  highOmegaFraction: 0.35,
  spinRotationThreshold: 1.25,
});

const AUDIT_STAGE = Object.freeze({
  id: -41,
  name: 'Rear-Target Rotation Audit',
  foods: 1,
  hazards: 0,
  walls: 0,
});

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function wrapAngle(a) {
  while (a > Math.PI) a -= TAU;
  while (a < -Math.PI) a += TAU;
  return a;
}
function mean(xs) { return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0; }
function rate(xs, predicate = Boolean) { return xs.length ? xs.filter(predicate).length / xs.length : 0; }
function finiteMean(xs) { const f = xs.filter(Number.isFinite); return f.length ? mean(f) : null; }
function dominantAction(probs) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { index: best, label: ACTIONS[best], probability: probs[best] };
}
function bearingToFood(env) {
  const f = env.nearestFood();
  if (!f) return 0;
  return wrapAngle(Math.atan2(f.y - env.agent.y, f.x - env.agent.x) - env.agent.angle);
}

function prepareAuditWorld(seed, bearingDeg) {
  const env = new World(seed, AUDIT_STAGE);
  const angle = bearingDeg * RAD;
  env.walls = [];
  env.hazards = [];
  env.food = [{
    x: 0.5 + Math.cos(angle) * ORIENTATION_AUDIT.targetDistance,
    y: 0.5 + Math.sin(angle) * ORIENTATION_AUDIT.targetDistance,
    r: CONFIG.world.foodRadius,
  }];
  env.agent = { x: 0.5, y: 0.5, vx: 0, vy: 0, angle: 0, omega: 0, energy: CONFIG.energy.initial };
  env.stepCount = 0;
  env.totalReward = 0;
  env.foodCollected = 0;
  env.hazardHits = 0;
  env.wallHits = 0;
  env.done = false;
  env.lastRewardParts = null;
  env.prevFoodDist = env.nearestFoodDistance();
  return env;
}

function runTrial(model, bearingDeg, trialIndex) {
  const seed = domainSeed(`${ORIENTATION_AUDIT.protocol}:bearing:${bearingDeg}`, trialIndex);
  const env = prepareAuditWorld(seed, bearingDeg);
  const rng = new PRNG(seed ^ 0x74a2c91d);
  let obs = env.observe();
  let hidden = model.zeroHidden();
  const initial = model.forward(obs, hidden, false);
  const initialProbs = Array.from(initial.probs);
  const initialDominant = dominantAction(initialProbs);
  const facingThreshold = ORIENTATION_AUDIT.facingThresholdDeg * RAD;
  const highOmegaThreshold = CONFIG.physics.maxAngularSpeed * ORIENTATION_AUDIT.highOmegaFraction;
  let firstFacingStep = Math.abs(bearingToFood(env)) <= facingThreshold ? 0 : null;
  let cumulativeTurn = 0;
  let cumulativeTurnBeforeFacing = 0;
  let brakeSteps = 0;
  let brakeWhileTurningSteps = 0;
  let turnActionSteps = 0;
  let reached = false;
  let lastAngle = env.agent.angle;
  let steps = 0;

  for (; steps < ORIENTATION_AUDIT.maxSteps && !env.done; steps++) {
    const act = model.act(obs, hidden, rng, false, false);
    if (act.action === 2) {
      brakeSteps++;
      if (Math.abs(env.agent.omega) >= highOmegaThreshold) brakeWhileTurningSteps++;
    }
    if (act.action === 3 || act.action === 4 || act.action === 5 || act.action === 6) turnActionSteps++;

    const beforeFood = env.foodCollected;
    const result = env.step(act.action);
    const deltaAngle = Math.abs(wrapAngle(env.agent.angle - lastAngle));
    cumulativeTurn += deltaAngle;
    if (firstFacingStep == null) cumulativeTurnBeforeFacing += deltaAngle;
    lastAngle = env.agent.angle;
    obs = result.obs;
    hidden = act.hidden;

    if (firstFacingStep == null && Math.abs(bearingToFood(env)) <= facingThreshold) firstFacingStep = steps + 1;
    if (env.foodCollected > beforeFood) { reached = true; steps++; break; }
  }

  const rotations = cumulativeTurn / TAU;
  const rotationsBeforeFacing = cumulativeTurnBeforeFacing / TAU;
  const spin = rotations >= ORIENTATION_AUDIT.spinRotationThreshold || (firstFacingStep == null && rotations >= 1);
  return {
    bearingDeg,
    trialIndex,
    seed,
    steps,
    faced: firstFacingStep != null,
    firstFacingStep,
    reached,
    spin,
    rotations,
    rotationsBeforeFacing,
    brakeFraction: steps ? brakeSteps / steps : 0,
    brakeWhileTurningFraction: brakeSteps ? brakeWhileTurningSteps / brakeSteps : 0,
    brakeWhileTurningSteps,
    turnActionFraction: steps ? turnActionSteps / steps : 0,
    finalBearingDeg: bearingToFood(env) * DEG,
    finalOmegaFraction: clamp(env.agent.omega / CONFIG.physics.maxAngularSpeed, -1, 1),
    initialProbs,
    initialDominant,
  };
}

function summarizeBearing(bearingDeg, trials) {
  const p = trials[0]?.initialProbs || [];
  const dominant = dominantAction(p);
  return {
    bearingDeg,
    trials: trials.length,
    facingRate: rate(trials, x => x.faced),
    reachRate: rate(trials, x => x.reached),
    spinRate: rate(trials, x => x.spin),
    meanFacingSteps: finiteMean(trials.map(x => x.firstFacingStep)),
    meanRotations: mean(trials.map(x => x.rotations)),
    meanRotationsBeforeFacing: mean(trials.map(x => x.rotationsBeforeFacing)),
    meanBrakeFraction: mean(trials.map(x => x.brakeFraction)),
    brakeWhileTurningRate: mean(trials.map(x => x.brakeWhileTurningFraction)),
    meanTurnActionFraction: mean(trials.map(x => x.turnActionFraction)),
    initialDominantAction: dominant.label,
    initialDominantProbability: dominant.probability,
    initialBrakeProbability: p[2] ?? 0,
    initialLeftProbability: (p[3] ?? 0) + (p[5] ?? 0),
    initialRightProbability: (p[4] ?? 0) + (p[6] ?? 0),
  };
}

export function runOrientationAudit(model, options = {}) {
  const bearings = options.bearingsDeg || ORIENTATION_AUDIT.bearingsDeg;
  const trialsPerBearing = Math.max(1, Number(options.trialsPerBearing) || ORIENTATION_AUDIT.trialsPerBearing);
  const trialRows = [];
  const byBearing = [];
  for (const bearingDeg of bearings) {
    const trials = [];
    for (let i = 0; i < trialsPerBearing; i++) {
      const row = runTrial(model, Number(bearingDeg), i);
      trials.push(row);
      trialRows.push(row);
    }
    byBearing.push(summarizeBearing(Number(bearingDeg), trials));
  }
  const rear = trialRows.filter(x => Math.abs(x.bearingDeg) >= 135);
  const side = trialRows.filter(x => Math.abs(x.bearingDeg) >= 45 && Math.abs(x.bearingDeg) < 135);
  const front = trialRows.filter(x => Math.abs(x.bearingDeg) < 45);
  const summarizeBand = rows => ({
    trials: rows.length,
    facingRate: rate(rows, x => x.faced),
    reachRate: rate(rows, x => x.reached),
    spinRate: rate(rows, x => x.spin),
    meanFacingSteps: finiteMean(rows.map(x => x.firstFacingStep)),
    meanRotations: mean(rows.map(x => x.rotations)),
    brakeWhileTurningRate: mean(rows.map(x => x.brakeWhileTurningFraction)),
  });
  return {
    protocol: `${ORIENTATION_AUDIT.protocol}|seeded-stochastic|${trialsPerBearing}x${bearings.length}|${ORIENTATION_AUDIT.maxSteps}-step`,
    settings: { ...ORIENTATION_AUDIT, bearingsDeg: [...bearings], trialsPerBearing },
    summary: {
      trials: trialRows.length,
      facingRate: rate(trialRows, x => x.faced),
      reachRate: rate(trialRows, x => x.reached),
      spinRate: rate(trialRows, x => x.spin),
      meanFacingSteps: finiteMean(trialRows.map(x => x.firstFacingStep)),
      meanRotations: mean(trialRows.map(x => x.rotations)),
      brakeWhileTurningRate: mean(trialRows.map(x => x.brakeWhileTurningFraction)),
    },
    bands: { front: summarizeBand(front), side: summarizeBand(side), rear: summarizeBand(rear) },
    bearings: byBearing,
  };
}
