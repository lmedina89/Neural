import { ACTIONS, CONFIG } from '../config.js';
import { World } from '../sim/world.js';
import { PRNG, domainSeed } from '../utils/prng.js';

const TAU = Math.PI * 2;
const DEG = 180 / Math.PI;
const RAD = Math.PI / 180;

// Entirely observational. None of these thresholds feed the policy, rewards,
// curriculum, world dynamics, PPO, checkpoint format, or Champion selection.
export const SPIN_TELEMETRY = Object.freeze({
  protocol: 'live-occlusion-spin:v1',
  historySteps: 72,          // ~4 s at the 55 ms OBSERVE stepping cadence
  spinWindowSteps: 48,       // ~2.6 s rolling diagnostic window
  postCaptureSteps: 24,      // capture recovery after a detected event
  spinRotationThreshold: 0.90,
  poorProgressDistance: 0.035,
  eventCooldownSteps: 54,
  maxEvents: 12,
  forwardConeDeg: 70,
});

export const OCCLUSION_AUDIT = Object.freeze({
  protocol: 'occlusion-conflict-audit:v1',
  trialsPerCase: 8,
  maxSteps: 240,
  cases: Object.freeze(['open', 'clearance', 'occluded']),
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

function classifyEvent(samples, triggerWindow, post) {
  const onset = triggerWindow.at(-1) || samples.at(-1);
  const before = triggerWindow[0] || onset;
  const blockedRate = rate(triggerWindow, x => x.losBlocked);
  const pathBlockedRate = rate(triggerWindow, x => x.pathBlocked);
  const forwardConeRate = rate(triggerWindow, x => x.forwardCone);
  const outsideForwardRate = 1 - forwardConeRate;
  const targetSwitch = triggerWindow.some((x, i) => i && x.targetKey !== triggerWindow[i - 1].targetKey);
  const dangerPeak = Math.max(...triggerWindow.map(x => Math.max(x.dangerL, x.dangerF, x.dangerR)), 0);
  const highOmegaAtOnset = Math.abs(onset?.omegaFraction || 0) >= 0.55;
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
  const endTurn = last12.reduce((s, x, i) => i ? s + Math.abs(wrapAngle(x.angle - last12[i - 1].angle)) : s, 0) / TAU;
  let recovery = 'capture-ended-still-active';
  if (end?.foodReached) recovery = 'food-reached';
  else if (end?.done) recovery = 'episode-ended';
  else if (onset?.losBlocked && end && !end.losBlocked) recovery = 'line-of-sight-cleared';
  else if (endTurn < 0.15) recovery = 'rotation-settled';
  else if (end?.targetKey !== onset?.targetKey) recovery = 'target-switched';

  return {
    cause,
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

function rollingTurn(samples) {
  let turn = 0;
  for (let i = 1; i < samples.length; i++) turn += Math.abs(wrapAngle(samples[i].angle - samples[i - 1].angle));
  return turn;
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

    const window = this.history.slice(-this.cfg.spinWindowSteps);
    if (window.length < Math.min(20, this.cfg.spinWindowSteps)) return null;
    const turn = rollingTurn(window);
    const progress = (window[0].foodDistance || 0) - (window.at(-1).foodDistance || 0);
    const cooldownOk = sample.worldStep - this.lastTriggerStep >= this.cfg.eventCooldownSteps || sample.worldStep < this.lastTriggerStep;
    const recentFoodReached = window.some(x => x.foodReached);
    if (cooldownOk && !recentFoodReached && turn >= this.cfg.spinRotationThreshold * TAU && progress <= this.cfg.poorProgressDistance) {
      this.lastTriggerStep = sample.worldStep;
      this.active = {
        id: ++this.eventCounter,
        triggerStep: sample.worldStep,
        seed: world.seed,
        stageName: world.stage?.name || 'Unknown',
        pre: this.history.slice(),
        triggerWindow: window.slice(),
        triggerRotations: turn / TAU,
        triggerProgress: progress,
        post: [],
      };
      if (sample.done || sample.foodReached) return this.finalizeActive();
    }
    return null;
  }
  finalizeActive() {
    if (!this.active) return null;
    const e = this.active;
    const cls = classifyEvent(e.pre, e.triggerWindow, e.post);
    const completed = {
      id: e.id,
      seed: e.seed,
      stageName: e.stageName,
      triggerStep: e.triggerStep,
      triggerRotations: e.triggerRotations,
      triggerProgress: e.triggerProgress,
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
      latestCause: events.at(-1)?.cause || '—',
      latestRecovery: events.at(-1)?.recovery || '—',
    };
  }
}

const AUDIT_STAGE = Object.freeze({ id: -42, name: 'Occlusion Conflict Audit', foods: 1, hazards: 0, walls: 0 });

function prepareAuditWorld(seed, caseName) {
  const env = new World(seed, AUDIT_STAGE);
  env.agent = { x: 0.24, y: 0.50, vx: 0, vy: 0, angle: 0, omega: 0, energy: CONFIG.energy.initial };
  env.food = [{ x: 0.76, y: 0.50, r: CONFIG.world.foodRadius }];
  env.hazards = [];
  if (caseName === 'open') env.walls = [];
  else if (caseName === 'clearance') env.walls = [{ x: 0.47, y: 0.52, w: 0.08, h: 0.18 }];
  else env.walls = [{ x: 0.47, y: 0.44, w: 0.08, h: 0.12 }];
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

function runOcclusionTrial(model, caseName, trialIndex) {
  const seed = domainSeed(`${OCCLUSION_AUDIT.protocol}:trial`, trialIndex);
  const env = prepareAuditWorld(seed, caseName);
  // Same stochastic-action stream for the same trial across all geometry cases.
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
    samples.push(sample);
    obs = result.obs;
    hidden = act.hidden;
    if (sample.foodReached) { reached = true; firstReachStep = step + 1; break; }
  }

  let maxWindowRotations = 0;
  let spin = false;
  for (let i = 0; i < samples.length; i++) {
    const w = samples.slice(Math.max(0, i - SPIN_TELEMETRY.spinWindowSteps + 1), i + 1);
    if (w.length < 20) continue;
    const rotations = rollingTurn(w) / TAU;
    const progress = w[0].foodDistance - w.at(-1).foodDistance;
    maxWindowRotations = Math.max(maxWindowRotations, rotations);
    if (rotations >= SPIN_TELEMETRY.spinRotationThreshold && progress <= SPIN_TELEMETRY.poorProgressDistance) spin = true;
  }
  const totalRotations = rollingTurn(samples) / TAU;
  const endDistance = samples.at(-1)?.nextFoodDistance ?? initialVisibility.distance;
  return {
    caseName,
    trialIndex,
    seed,
    initialLosBlocked: initialVisibility.losBlocked,
    initialPathBlocked: initialVisibility.pathBlocked,
    reached,
    firstReachStep,
    spin,
    totalRotations,
    maxWindowRotations,
    distanceProgress: initialVisibility.distance - endDistance,
    wallHits: env.wallHits,
    steps: samples.length,
  };
}

function summarizeCase(caseName, trials) {
  return {
    caseName,
    trials: trials.length,
    initialLosBlocked: Boolean(trials[0]?.initialLosBlocked),
    initialPathBlocked: Boolean(trials[0]?.initialPathBlocked),
    reachRate: rate(trials, x => x.reached),
    spinRate: rate(trials, x => x.spin),
    meanReachSteps: mean(trials.filter(x => x.firstReachStep != null).map(x => x.firstReachStep)),
    meanTotalRotations: mean(trials.map(x => x.totalRotations)),
    meanMaxWindowRotations: mean(trials.map(x => x.maxWindowRotations)),
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
