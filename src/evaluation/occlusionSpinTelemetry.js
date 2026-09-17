import { ACTIONS, CONFIG } from '../config.js';
import { World } from '../sim/world.js';
import { PRNG, domainSeed } from '../utils/prng.js';

const TAU = Math.PI * 2;
const DEG = 180 / Math.PI;

// Entirely observational. None of these thresholds feed the policy, rewards,
// curriculum, world dynamics, PPO, checkpoint format, or Champion selection.
// The v2 detector intentionally broadens the old "continuous spin" definition
// into a rotation-trap definition that also catches turn/hesitate/reverse loops.
export const SPIN_TELEMETRY = Object.freeze({
  protocol: 'live-occlusion-rotation-trap:v2',
  historySteps: 160,              // ~8.8 s at the 55 ms OBSERVE stepping cadence
  trapWindowSteps: 96,            // ~5.3 s rolling diagnostic window
  postCaptureSteps: 24,           // capture a short recovery tail
  rotationTrapThreshold: 0.55,    // absolute angular travel inside the window
  oscillationRotationThreshold: 0.34,
  poorProgressDistance: 0.050,
  oscillationProgressDistance: 0.040,
  minDirectionReversals: 2,
  minBearingCrossings: 2,
  eventCooldownSteps: 110,
  maxEvents: 12,
  forwardConeDeg: 70,
  angleMotionEpsilon: 0.006,
  bearingDeadbandDeg: 8,
  awayProgressEpsilon: 0.00025,
});

export const OCCLUSION_AUDIT = Object.freeze({
  protocol: 'occlusion-failure-characterization:v2',
  trialsPerCase: 8,
  maxSteps: 360,
  cases: Object.freeze([
    'open',
    'clearance',
    'narrow-center',
    'wide-center',
    'left-heavy',
    'right-heavy',
    'long-detour',
  ]),
});

const CASE_LABELS = Object.freeze({
  open: 'OPEN',
  clearance: 'CLEARANCE',
  'narrow-center': 'NARROW CENTER',
  'wide-center': 'WIDE CENTER',
  'left-heavy': 'LEFT-HEAVY BLOCK',
  'right-heavy': 'RIGHT-HEAVY BLOCK',
  'long-detour': 'LONG DETOUR',
});

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function wrapAngle(a) {
  while (a > Math.PI) a -= TAU;
  while (a < -Math.PI) a += TAU;
  return a;
}
function mean(xs) { return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0; }
function rate(xs, pred = Boolean) { return xs.length ? xs.filter(pred).length / xs.length : 0; }
function rms(xs) { return xs?.length ? Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / xs.length) : 0; }
function isTurnAction(action) { return action === 3 || action === 4 || action === 5 || action === 6; }
function isThrustAction(action) { return action === 1 || action === 5 || action === 6; }
function isBrakeAction(action) { return action === 2; }
function nearestFoodIndex(world) {
  if (!world?.food?.length) return -1;
  let best = 0;
  let bd = Math.hypot(world.agent.x - world.food[0].x, world.agent.y - world.food[0].y);
  for (let i = 1; i < world.food.length; i++) {
    const d = Math.hypot(world.agent.x - world.food[i].x, world.agent.y - world.food[i].y);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
function targetKey(food) {
  return food ? `${food.x.toFixed(6)},${food.y.toFixed(6)}` : 'none';
}

// Liang-Barsky segment/rectangle intersection. Raw rectangle = visual centerline
// obstruction. Expanded rectangle = direct travel corridor obstruction.
function segmentIntersectsRect(x0, y0, x1, y1, rect, expand = 0) {
  const xmin = rect.x - expand, xmax = rect.x + rect.w + expand;
  const ymin = rect.y - expand, ymax = rect.y + rect.h + expand;
  const dx = x1 - x0, dy = y1 - y0;
  const p = [-dx, dx, -dy, dy];
  const q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0];
  let u1 = 0, u2 = 1;
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < 1e-12) {
      if (q[i] < 0) return false;
      continue;
    }
    const t = q[i] / p[i];
    if (p[i] < 0) u1 = Math.max(u1, t);
    else u2 = Math.min(u2, t);
    if (u1 > u2) return false;
  }
  return true;
}

export function foodVisibilityDiagnostic(world) {
  const index = nearestFoodIndex(world);
  const food = index >= 0 ? world.food[index] : null;
  if (!food) return {
    foodIndex: -1, targetKey: 'none', distance: 1.4, bearingRad: 0, bearingDeg: 0,
    forwardCone: false, rearHemisphere: false, losBlocked: false, pathBlocked: false,
    losWallIndex: -1, pathWallIndex: -1,
  };
  const dx = food.x - world.agent.x, dy = food.y - world.agent.y;
  const bearingRad = wrapAngle(Math.atan2(dy, dx) - world.agent.angle);
  let losWallIndex = -1, pathWallIndex = -1;
  // Match actual collision geometry: a direct centerline is non-traversable when
  // the agent-radius-expanded wall intersects it. wallMargin is spawn hygiene,
  // not runtime collision physics, so it is intentionally excluded here.
  const travelExpand = CONFIG.world.agentRadius;
  for (let i = 0; i < world.walls.length; i++) {
    const wall = world.walls[i];
    if (losWallIndex < 0 && segmentIntersectsRect(world.agent.x, world.agent.y, food.x, food.y, wall, 0)) losWallIndex = i;
    if (pathWallIndex < 0 && segmentIntersectsRect(world.agent.x, world.agent.y, food.x, food.y, wall, travelExpand)) pathWallIndex = i;
  }
  const bearingDeg = bearingRad * DEG;
  return {
    foodIndex: index,
    targetKey: targetKey(food),
    distance: Math.hypot(dx, dy),
    bearingRad,
    bearingDeg,
    forwardCone: Math.abs(bearingDeg) <= SPIN_TELEMETRY.forwardConeDeg,
    rearHemisphere: Math.abs(bearingDeg) > 90,
    losBlocked: losWallIndex >= 0,
    pathBlocked: pathWallIndex >= 0,
    losWallIndex,
    pathWallIndex,
  };
}

function hiddenDeltaRms(hidden, previousHidden) {
  if (!hidden?.length || !previousHidden?.length || hidden.length !== previousHidden.length) return 0;
  let s = 0;
  for (let i = 0; i < hidden.length; i++) {
    const d = hidden[i] - previousHidden[i];
    s += d * d;
  }
  return Math.sqrt(s / hidden.length);
}

export function captureSpinSample(world, snapshot, action, previous = null) {
  const vis = foodVisibilityDiagnostic(world);
  const obs = snapshot?.obs || world.observe();
  const probs = snapshot?.probs ? Array.from(snapshot.probs) : Array(CONFIG.model.actionSize).fill(0);
  const hidden = snapshot?.h ? Array.from(snapshot.h) : [];
  return {
    worldStep: world.stepCount,
    stageId: world.stage?.id ?? -1,
    stageName: world.stage?.name ?? 'Unknown',
    x: world.agent.x, y: world.agent.y,
    angle: world.agent.angle,
    speed: Math.hypot(world.agent.vx, world.agent.vy),
    omega: world.agent.omega,
    omegaFraction: clamp(world.agent.omega / CONFIG.physics.maxAngularSpeed, -1, 1),
    energy: world.agent.energy,
    foodIndex: vis.foodIndex,
    targetKey: vis.targetKey,
    foodDistance: vis.distance,
    foodBearingDeg: vis.bearingDeg,
    forwardCone: vis.forwardCone,
    rearHemisphere: vis.rearHemisphere,
    losBlocked: vis.losBlocked,
    pathBlocked: vis.pathBlocked,
    dangerL: Number(obs[3]) || 0,
    dangerF: Number(obs[4]) || 0,
    dangerR: Number(obs[5]) || 0,
    action: Number(action),
    actionLabel: ACTIONS[action] || `A${action}`,
    probs,
    value: Number(snapshot?.value) || 0,
    hiddenRms: rms(hidden),
    hiddenDeltaRms: hiddenDeltaRms(hidden, previous?.hidden),
    hidden,
  };
}

function rollingTurn(samples) {
  let turn = 0;
  for (let i = 1; i < samples.length; i++) turn += Math.abs(wrapAngle(samples[i].angle - samples[i - 1].angle));
  return turn;
}

function signedTurn(samples) {
  let turn = 0;
  for (let i = 1; i < samples.length; i++) turn += wrapAngle(samples[i].angle - samples[i - 1].angle);
  return turn;
}

function countTurnReversals(samples, epsilon = SPIN_TELEMETRY.angleMotionEpsilon) {
  let reversals = 0;
  let previousSign = 0;
  for (let i = 1; i < samples.length; i++) {
    const d = wrapAngle(samples[i].angle - samples[i - 1].angle);
    if (Math.abs(d) < epsilon) continue;
    const sign = d < 0 ? -1 : 1;
    if (previousSign && sign !== previousSign) reversals++;
    previousSign = sign;
  }
  return reversals;
}

function countBearingCrossings(samples, deadband = SPIN_TELEMETRY.bearingDeadbandDeg) {
  let crossings = 0;
  let previousSign = 0;
  for (const sample of samples) {
    const b = Number(sample.foodBearingDeg) || 0;
    if (Math.abs(b) <= deadband) continue;
    const sign = b < 0 ? -1 : 1;
    if (previousSign && sign !== previousSign) crossings++;
    previousSign = sign;
  }
  return crossings;
}

function windowMetrics(samples, cfg = SPIN_TELEMETRY) {
  if (!samples.length) return {
    rotations: 0, netRotations: 0, progress: 0, reversals: 0, bearingCrossings: 0,
    turnActionRate: 0, thrustActionRate: 0, brakeActionRate: 0, awayThrustRate: 0,
    losBlockedRate: 0, pathBlockedRate: 0,
  };
  const rotations = rollingTurn(samples) / TAU;
  const netRotations = signedTurn(samples) / TAU;
  const progress = (samples[0].foodDistance || 0) - (samples.at(-1).foodDistance || 0);
  const thrust = samples.filter(x => isThrustAction(x.action));
  const awayThrust = thrust.filter(x => (Number(x.foodProgress) || 0) < -cfg.awayProgressEpsilon);
  return {
    rotations,
    netRotations,
    progress,
    reversals: countTurnReversals(samples, cfg.angleMotionEpsilon),
    bearingCrossings: countBearingCrossings(samples, cfg.bearingDeadbandDeg),
    turnActionRate: rate(samples, x => isTurnAction(x.action)),
    thrustActionRate: rate(samples, x => isThrustAction(x.action)),
    brakeActionRate: rate(samples, x => isBrakeAction(x.action)),
    awayThrustRate: thrust.length ? awayThrust.length / thrust.length : 0,
    losBlockedRate: rate(samples, x => x.losBlocked),
    pathBlockedRate: rate(samples, x => x.pathBlocked),
  };
}

function isRotationTrap(metrics, cfg = SPIN_TELEMETRY) {
  const poorProgressTrap = metrics.rotations >= cfg.rotationTrapThreshold
    && metrics.progress <= cfg.poorProgressDistance;
  const oscillationTrap = metrics.rotations >= cfg.oscillationRotationThreshold
    && metrics.progress <= cfg.oscillationProgressDistance
    && (metrics.reversals >= cfg.minDirectionReversals
      || metrics.bearingCrossings >= cfg.minBearingCrossings);
  return poorProgressTrap || oscillationTrap;
}

function classifyEvent(samples, triggerWindow, post, triggerMetrics) {
  const onset = triggerWindow.at(-1) || samples.at(-1);
  const before = triggerWindow[0] || onset;
  const blockedRate = triggerMetrics.losBlockedRate;
  const pathBlockedRate = triggerMetrics.pathBlockedRate;
  const forwardConeRate = rate(triggerWindow, x => x.forwardCone);
  const outsideForwardRate = 1 - forwardConeRate;
  const targetSwitch = triggerWindow.some((x, i) => i && x.targetKey !== triggerWindow[i - 1].targetKey);
  const dangerPeak = Math.max(...triggerWindow.map(x => Math.max(x.dangerL, x.dangerF, x.dangerR)), 0);
  const highOmegaAtOnset = Math.abs(onset?.omegaFraction || 0) >= 0.55;
  const sweepRatio = triggerMetrics.rotations > 1e-9 ? Math.abs(triggerMetrics.netRotations) / triggerMetrics.rotations : 0;
  const rotationMode = triggerMetrics.reversals >= SPIN_TELEMETRY.minDirectionReversals || triggerMetrics.bearingCrossings >= SPIN_TELEMETRY.minBearingCrossings
    ? 'oscillatory-turn-loop'
    : sweepRatio >= 0.72 ? 'one-way-sweep' : 'mixed-turn-loop';

  let cause = 'unclassified';
  if (targetSwitch) cause = 'target-switch';
  else if (blockedRate >= 0.5 && dangerPeak >= 0.25) cause = 'wall/food-conflict';
  else if (blockedRate >= 0.5) cause = 'line-of-sight-blocked';
  else if (pathBlockedRate >= 0.5 && dangerPeak >= 0.25) cause = 'clearance/danger-conflict';
  else if (outsideForwardRate >= 0.7 && dangerPeak >= 0.25) cause = 'off-axis/danger-conflict';
  else if (outsideForwardRate >= 0.7) cause = 'outside-forward-cone';
  else if (highOmegaAtOnset) cause = 'angular-momentum';
  else if (onset?.rearHemisphere) cause = 'rear-target-context';

  const end = post.at(-1) || onset;
  const last12 = post.slice(-12);
  const endTurn = rollingTurn(last12) / TAU;
  let recovery = 'capture-ended-still-active';
  if (end?.foodReached) recovery = 'food-reached';
  else if (end?.done) recovery = 'episode-ended';
  else if (onset?.losBlocked && end && !end.losBlocked) recovery = 'line-of-sight-cleared';
  else if (onset?.pathBlocked && end && !end.pathBlocked) recovery = 'direct-path-cleared';
  else if (endTurn < 0.15) recovery = 'rotation-settled';
  else if (end?.targetKey !== onset?.targetKey) recovery = 'target-switched';

  return {
    cause,
    rotationMode,
    recovery,
    blockedRate,
    pathBlockedRate,
    forwardConeRate,
    outsideForwardRate,
    targetSwitch,
    dangerPeak,
    highOmegaAtOnset,
    onsetBearingDeg: onset?.foodBearingDeg ?? 0,
    onsetOmegaFraction: onset?.omegaFraction ?? 0,
    startDistance: before?.foodDistance ?? 0,
    onsetDistance: onset?.foodDistance ?? 0,
    endDistance: end?.foodDistance ?? 0,
  };
}

export class LiveSpinRecorder {
  constructor(options = {}) {
    this.cfg = { ...SPIN_TELEMETRY, ...options };
    this.reset();
  }
  reset() {
    this.events = [];
    this.eventCounter = 0;
    this.totalSamples = 0;
    this.losBlockedSamples = 0;
    this.pathBlockedSamples = 0;
    this.lastTriggerStep = -Infinity;
    this.beginEpisode();
  }
  beginEpisode() {
    this.history = [];
    this.active = null;
    this.previousHidden = null;
  }
  beginSample(world, snapshot, action) {
    const s = captureSpinSample(world, snapshot, action, { hidden: this.previousHidden });
    this.previousHidden = s.hidden;
    return s;
  }
  endSample(sample, world, result) {
    const nextVis = foodVisibilityDiagnostic(world);
    sample.nextFoodDistance = nextVis.distance;
    sample.foodProgress = sample.foodDistance - nextVis.distance;
    sample.reward = Number(result?.reward) || 0;
    sample.rewardParts = { ...(world.lastRewardParts || {}) };
    sample.foodReached = (Number(sample.rewardParts.food) || 0) > 0;
    sample.done = Boolean(result?.done);
    sample.nextLosBlocked = nextVis.losBlocked;
    sample.nextPathBlocked = nextVis.pathBlocked;
    this.totalSamples++;
    if (sample.losBlocked) this.losBlockedSamples++;
    if (sample.pathBlocked) this.pathBlockedSamples++;
    this.history.push(sample);
    if (this.history.length > this.cfg.historySteps) this.history.shift();

    if (this.active) {
      this.active.post.push(sample);
      if (this.active.post.length >= this.cfg.postCaptureSteps || sample.done || sample.foodReached) return this.finalizeActive();
      return null;
    }

    const window = this.history.slice(-this.cfg.trapWindowSteps);
    if (window.length < Math.min(32, this.cfg.trapWindowSteps)) return null;
    const metrics = windowMetrics(window, this.cfg);
    const cooldownOk = sample.worldStep - this.lastTriggerStep >= this.cfg.eventCooldownSteps || sample.worldStep < this.lastTriggerStep;
    const recentFoodReached = window.some(x => x.foodReached);
    if (cooldownOk && !recentFoodReached && isRotationTrap(metrics, this.cfg)) {
      this.lastTriggerStep = sample.worldStep;
      this.active = {
        id: ++this.eventCounter,
        triggerStep: sample.worldStep,
        seed: world.seed,
        stageName: world.stage?.name || 'Unknown',
        pre: this.history.slice(),
        triggerWindow: window.slice(),
        triggerMetrics: metrics,
        post: [],
      };
      if (sample.done || sample.foodReached) return this.finalizeActive();
    }
    return null;
  }
  finalizeActive() {
    if (!this.active) return null;
    const e = this.active;
    const cls = classifyEvent(e.pre, e.triggerWindow, e.post, e.triggerMetrics);
    const completed = {
      id: e.id,
      seed: e.seed,
      stageName: e.stageName,
      triggerStep: e.triggerStep,
      triggerRotations: e.triggerMetrics.rotations,
      triggerNetRotations: e.triggerMetrics.netRotations,
      triggerProgress: e.triggerMetrics.progress,
      triggerReversals: e.triggerMetrics.reversals,
      triggerBearingCrossings: e.triggerMetrics.bearingCrossings,
      triggerTurnActionRate: e.triggerMetrics.turnActionRate,
      triggerThrustActionRate: e.triggerMetrics.thrustActionRate,
      triggerBrakeActionRate: e.triggerMetrics.brakeActionRate,
      triggerAwayThrustRate: e.triggerMetrics.awayThrustRate,
      ...cls,
      samples: [...e.pre, ...e.post],
    };
    this.events.push(completed);
    if (this.events.length > this.cfg.maxEvents) this.events.shift();
    this.active = null;
    return completed;
  }
  summary() {
    const events = this.events;
    return {
      protocol: this.cfg.protocol,
      samples: this.totalSamples,
      events: events.length,
      active: Boolean(this.active),
      losBlockedSampleRate: this.totalSamples ? this.losBlockedSamples / this.totalSamples : 0,
      pathBlockedSampleRate: this.totalSamples ? this.pathBlockedSamples / this.totalSamples : 0,
      blockedAtOnsetRate: rate(events, e => e.blockedRate >= 0.5),
      wallConflictRate: rate(events, e => e.cause === 'wall/food-conflict'),
      oscillatoryRate: rate(events, e => e.rotationMode === 'oscillatory-turn-loop'),
      latestCause: events.at(-1)?.cause || '—',
      latestMode: events.at(-1)?.rotationMode || '—',
      latestRecovery: events.at(-1)?.recovery || '—',
    };
  }
}

const AUDIT_STAGE = Object.freeze({ id: -42, name: 'Occlusion Failure Characterization', foods: 1, hazards: 0, walls: 0 });

function wallsForCase(caseName) {
  switch (caseName) {
    case 'open': return [];
    // Raw centerline remains clear; collision-radius corridor is blocked.
    case 'clearance': return [{ x: 0.47, y: 0.52, w: 0.08, h: 0.18 }];
    // Small centered obstacle: minimum detour demand.
    case 'narrow-center': return [{ x: 0.485, y: 0.46, w: 0.05, h: 0.08 }];
    // Taller centered obstacle: either side is possible but requires a real bypass.
    case 'wide-center': return [{ x: 0.46, y: 0.34, w: 0.08, h: 0.32 }];
    // Heading starts to +X. LEFT means negative-Y/up in this world convention.
    // This wall occupies more of the left-turn side, so the shorter bypass is RIGHT.
    case 'left-heavy': return [{ x: 0.46, y: 0.25, w: 0.08, h: 0.29 }];
    // Mirror image: shorter bypass is LEFT.
    case 'right-heavy': return [{ x: 0.46, y: 0.46, w: 0.08, h: 0.29 }];
    // Large barrier that requires committing to a substantial lateral detour.
    case 'long-detour': return [{ x: 0.46, y: 0.18, w: 0.08, h: 0.64 }];
    default: throw new Error(`Unknown occlusion-audit case: ${caseName}`);
  }
}

function prepareAuditWorld(seed, caseName) {
  const env = new World(seed, AUDIT_STAGE);
  env.agent = { x: 0.24, y: 0.50, vx: 0, vy: 0, angle: 0, omega: 0, energy: CONFIG.energy.initial };
  env.food = [{ x: 0.76, y: 0.50, r: CONFIG.world.foodRadius }];
  env.hazards = [];
  env.walls = wallsForCase(caseName);
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

function analyzeTrial(samples, initialVisibility, reached, firstReachStep, env) {
  let maxWindowRotations = 0;
  let rotationTrap = false;
  let trapWindows = 0;
  for (let i = 0; i < samples.length; i++) {
    const w = samples.slice(Math.max(0, i - SPIN_TELEMETRY.trapWindowSteps + 1), i + 1);
    if (w.length < 32) continue;
    const metrics = windowMetrics(w);
    maxWindowRotations = Math.max(maxWindowRotations, metrics.rotations);
    if (isRotationTrap(metrics)) {
      rotationTrap = true;
      trapWindows++;
    }
  }

  const totalRotations = rollingTurn(samples) / TAU;
  const netRotations = signedTurn(samples) / TAU;
  const turnReversals = countTurnReversals(samples);
  const bearingCrossings = countBearingCrossings(samples);
  const turnActionRate = rate(samples, x => isTurnAction(x.action));
  const thrustActionRate = rate(samples, x => isThrustAction(x.action));
  const brakeActionRate = rate(samples, x => isBrakeAction(x.action));
  const thrustSamples = samples.filter(x => isThrustAction(x.action));
  const awayThrustRate = thrustSamples.length
    ? rate(thrustSamples, x => (Number(x.foodProgress) || 0) < -SPIN_TELEMETRY.awayProgressEpsilon)
    : 0;

  const firstPathClear = initialVisibility.pathBlocked
    ? samples.findIndex(x => x.nextPathBlocked === false)
    : -1;
  const firstLosClear = initialVisibility.losBlocked
    ? samples.findIndex(x => x.nextLosBlocked === false)
    : -1;
  const pathCleared = initialVisibility.pathBlocked ? firstPathClear >= 0 : true;
  const losCleared = initialVisibility.losBlocked ? firstLosClear >= 0 : true;
  const pathClearStep = firstPathClear >= 0 ? firstPathClear + 1 : null;
  const losClearStep = firstLosClear >= 0 ? firstLosClear + 1 : null;

  const startDistance = initialVisibility.distance;
  const observedDistances = [startDistance, ...samples.map(x => x.nextFoodDistance ?? x.foodDistance)];
  const maxDistanceIncrease = Math.max(...observedDistances) - startDistance;
  const maxLateralExcursion = samples.length ? Math.max(...samples.map(x => Math.abs(x.y - 0.5))) : 0;
  const endDistance = samples.at(-1)?.nextFoodDistance ?? startDistance;
  const distanceProgress = startDistance - endDistance;
  const successfulDetour = Boolean(reached && initialVisibility.pathBlocked && pathCleared && maxLateralExcursion >= CONFIG.world.agentRadius * 1.5);
  const temporaryRetreat = maxDistanceIncrease >= 0.012 && awayThrustRate > 0;

  return {
    reached,
    firstReachStep,
    rotationTrap,
    trapWindows,
    totalRotations,
    netRotations,
    maxWindowRotations,
    turnReversals,
    bearingCrossings,
    turnActionRate,
    thrustActionRate,
    brakeActionRate,
    awayThrustRate,
    pathCleared,
    pathClearStep,
    losCleared,
    losClearStep,
    maxDistanceIncrease,
    maxLateralExcursion,
    successfulDetour,
    temporaryRetreat,
    distanceProgress,
    wallHits: env.wallHits,
    steps: samples.length,
  };
}

function runOcclusionTrial(model, caseName, trialIndex) {
  const seed = domainSeed(`${OCCLUSION_AUDIT.protocol}:trial`, trialIndex);
  const env = prepareAuditWorld(seed, caseName);
  // Same stochastic-action stream for the same trial across every geometry case.
  const rng = new PRNG(domainSeed(`${OCCLUSION_AUDIT.protocol}:action`, trialIndex));
  let obs = env.observe();
  let hidden = model.zeroHidden();
  const samples = [];
  let reached = false;
  let firstReachStep = null;
  const initialVisibility = foodVisibilityDiagnostic(env);

  for (let step = 0; step < OCCLUSION_AUDIT.maxSteps && !env.done; step++) {
    const act = model.act(obs, hidden, rng, false, true);
    const sample = captureSpinSample(env, act.snapshot, act.action, samples.at(-1) ? { hidden: samples.at(-1).hidden } : null);
    const result = env.step(act.action);
    const nextVis = foodVisibilityDiagnostic(env);
    sample.nextFoodDistance = nextVis.distance;
    sample.foodProgress = sample.foodDistance - nextVis.distance;
    sample.reward = result.reward;
    sample.rewardParts = { ...(env.lastRewardParts || {}) };
    sample.foodReached = (Number(sample.rewardParts.food) || 0) > 0;
    sample.done = result.done;
    sample.nextLosBlocked = nextVis.losBlocked;
    sample.nextPathBlocked = nextVis.pathBlocked;
    samples.push(sample);
    obs = result.obs;
    hidden = act.hidden;
    if (sample.foodReached) { reached = true; firstReachStep = step + 1; break; }
  }

  return {
    caseName,
    caseLabel: CASE_LABELS[caseName] || caseName,
    trialIndex,
    seed,
    initialLosBlocked: initialVisibility.losBlocked,
    initialPathBlocked: initialVisibility.pathBlocked,
    ...analyzeTrial(samples, initialVisibility, reached, firstReachStep, env),
  };
}

function summarizeCase(caseName, trials) {
  const reachSteps = trials.filter(x => x.firstReachStep != null).map(x => x.firstReachStep);
  const pathSteps = trials.filter(x => x.pathClearStep != null).map(x => x.pathClearStep);
  const losSteps = trials.filter(x => x.losClearStep != null).map(x => x.losClearStep);
  return {
    caseName,
    caseLabel: CASE_LABELS[caseName] || caseName,
    trials: trials.length,
    initialLosBlocked: Boolean(trials[0]?.initialLosBlocked),
    initialPathBlocked: Boolean(trials[0]?.initialPathBlocked),
    reachRate: rate(trials, x => x.reached),
    rotationTrapRate: rate(trials, x => x.rotationTrap),
    pathClearRate: rate(trials, x => x.pathCleared),
    losClearRate: rate(trials, x => x.losCleared),
    successfulDetourRate: rate(trials, x => x.successfulDetour),
    temporaryRetreatRate: rate(trials, x => x.temporaryRetreat),
    meanReachSteps: mean(reachSteps),
    meanPathClearStep: mean(pathSteps),
    meanLosClearStep: mean(losSteps),
    meanTotalRotations: mean(trials.map(x => x.totalRotations)),
    meanNetRotations: mean(trials.map(x => x.netRotations)),
    meanMaxWindowRotations: mean(trials.map(x => x.maxWindowRotations)),
    meanTurnReversals: mean(trials.map(x => x.turnReversals)),
    meanBearingCrossings: mean(trials.map(x => x.bearingCrossings)),
    meanTurnActionRate: mean(trials.map(x => x.turnActionRate)),
    meanThrustActionRate: mean(trials.map(x => x.thrustActionRate)),
    meanBrakeActionRate: mean(trials.map(x => x.brakeActionRate)),
    meanAwayThrustRate: mean(trials.map(x => x.awayThrustRate)),
    meanMaxDistanceIncrease: mean(trials.map(x => x.maxDistanceIncrease)),
    meanMaxLateralExcursion: mean(trials.map(x => x.maxLateralExcursion)),
    meanDistanceProgress: mean(trials.map(x => x.distanceProgress)),
    meanWallHits: mean(trials.map(x => x.wallHits)),
  };
}

export function runOcclusionAudit(model, options = {}) {
  const cases = options.cases || OCCLUSION_AUDIT.cases;
  const trialsPerCase = Math.max(1, Number(options.trialsPerCase) || OCCLUSION_AUDIT.trialsPerCase);
  const rows = [];
  const trials = [];
  for (const caseName of cases) {
    const xs = [];
    for (let i = 0; i < trialsPerCase; i++) {
      const r = runOcclusionTrial(model, caseName, i);
      xs.push(r); trials.push(r);
    }
    rows.push(summarizeCase(caseName, xs));
  }
  return {
    protocol: `${OCCLUSION_AUDIT.protocol}|paired-action-stream|${trialsPerCase}x${cases.length}|${OCCLUSION_AUDIT.maxSteps}-step`,
    settings: { ...OCCLUSION_AUDIT, cases: [...cases], trialsPerCase },
    cases: rows,
    trials,
  };
}
