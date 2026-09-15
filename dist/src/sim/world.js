import { CONFIG } from '../config.js';
import { PRNG } from '../utils/prng.js';

const TAU = Math.PI * 2;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const wrapAngle = (a) => {
  while (a > Math.PI) a -= TAU;
  while (a < -Math.PI) a += TAU;
  return a;
};
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function pointRectDistance(p, w) {
  const cx = clamp(p.x, w.x, w.x + w.w);
  const cy = clamp(p.y, w.y, w.y + w.h);
  return Math.hypot(p.x - cx, p.y - cy);
}

function hasWorldClearance(p, entityRadius, walls, margin = CONFIG.world.wallMargin) {
  const clearance = entityRadius + margin;
  if (p.x < clearance || p.x > 1 - clearance || p.y < clearance || p.y > 1 - clearance) return false;
  return walls.every(w => pointRectDistance(p, w) >= clearance);
}

function separatedFromOccupied(p, occupied, radius) {
  return occupied.every(o => Math.hypot(p.x - o.x, p.y - o.y) > radius + (o.r ?? 0.04));
}

function correctiveClearPoint(rng, occupied, separationRadius, entityRadius, walls, tries = 160) {
  const clearance = entityRadius + CONFIG.world.wallMargin;
  const lo = Math.max(0.08, clearance);
  const hi = Math.min(0.92, 1 - clearance);
  for (let i = 0; i < tries; i++) {
    const p = { x: rng.range(lo, hi), y: rng.range(lo, hi) };
    if (separatedFromOccupied(p, occupied, separationRadius) && hasWorldClearance(p, entityRadius, walls)) return p;
  }

  // Deterministic center-out fallback. This should be rare, but unlike the old
  // clearPoint fallback it never silently returns a point inside a wall.
  const grid = 21;
  const points = [];
  for (let iy = 0; iy < grid; iy++) {
    for (let ix = 0; ix < grid; ix++) {
      const x = lo + (hi - lo) * (ix + 0.5) / grid;
      const y = lo + (hi - lo) * (iy + 0.5) / grid;
      points.push({ x, y, d2: (x - 0.5) ** 2 + (y - 0.5) ** 2 });
    }
  }
  points.sort((a, b) => a.d2 - b.d2 || a.y - b.y || a.x - b.x);
  for (const p of points) {
    if (separatedFromOccupied(p, occupied, separationRadius) && hasWorldClearance(p, entityRadius, walls)) return { x: p.x, y: p.y };
  }
  throw new Error('Unable to find valid world spawn point');
}

function clearPoint(rng, occupied, radius = 0.08, tries = 100) {
  for (let i = 0; i < tries; i++) {
    const p = { x: rng.range(0.08, 0.92), y: rng.range(0.08, 0.92) };
    if (occupied.every(o => Math.hypot(p.x - o.x, p.y - o.y) > radius + (o.r ?? 0.04))) return p;
  }
  return { x: 0.5, y: 0.5 };
}

export class World {
  constructor(seed, stage) {
    this.seed = seed >>> 0;
    this.stage = stage;
    this.rng = new PRNG(this.seed);
    this.reset();
  }

  reset() {
    const cfg = CONFIG;
    this.stepCount = 0;
    this.totalReward = 0;
    this.foodCollected = 0;
    this.hazardHits = 0;
    this.wallHits = 0;
    this.done = false;
    const occupied = [];
    const p = clearPoint(this.rng, occupied, 0.12);
    this.agent = { x: p.x, y: p.y, vx: 0, vy: 0, angle: this.rng.range(-Math.PI, Math.PI), omega: 0, energy: cfg.energy.initial };
    occupied.push({ ...p, r: 0.09 });
    this.food = [];
    for (let i = 0; i < this.stage.foods; i++) {
      const q = clearPoint(this.rng, occupied, 0.11);
      this.food.push({ ...q, r: cfg.world.foodRadius });
      occupied.push({ ...q, r: 0.07 });
    }
    this.hazards = [];
    for (let i = 0; i < this.stage.hazards; i++) {
      const q = clearPoint(this.rng, occupied, 0.14);
      const r = cfg.world.hazardRadius * this.rng.range(0.85, 1.15);
      this.hazards.push({ ...q, r });
      occupied.push({ ...q, r: r + 0.05 });
    }
    this.walls = [];
    for (let i = 0; i < this.stage.walls; i++) {
      const horizontal = this.rng.next() > 0.5;
      const w = horizontal ? this.rng.range(0.14, 0.25) : this.rng.range(0.035, 0.06);
      const h = horizontal ? this.rng.range(0.035, 0.06) : this.rng.range(0.14, 0.25);
      this.walls.push({ x: this.rng.range(0.15, 0.85 - w), y: this.rng.range(0.15, 0.85 - h), w, h });
    }

    // Walls are generated after the entities so legacy-valid seeds retain their
    // exact geometry. Only a genuinely tight/invalid spawn is corrected here.
    this.correctSpawnClearance();
    this.prevFoodDist = this.nearestFoodDistance();
    return this.observe();
  }

  correctSpawnClearance() {
    const cfg = CONFIG.world;
    const occupancy = (skipType = '', skipIndex = -1) => {
      const out = [];
      if (skipType !== 'agent') out.push({ x: this.agent.x, y: this.agent.y, r: 0.09 });
      for (let i = 0; i < this.food.length; i++) if (!(skipType === 'food' && i === skipIndex)) {
        out.push({ x: this.food[i].x, y: this.food[i].y, r: 0.07 });
      }
      for (let i = 0; i < this.hazards.length; i++) if (!(skipType === 'hazard' && i === skipIndex)) {
        const h = this.hazards[i];
        out.push({ x: h.x, y: h.y, r: h.r + 0.05 });
      }
      return out;
    };

    if (!hasWorldClearance(this.agent, cfg.agentRadius, this.walls)) {
      const p = correctiveClearPoint(this.rng, occupancy('agent'), 0.12, cfg.agentRadius, this.walls);
      this.agent.x = p.x;
      this.agent.y = p.y;
    }

    for (let i = 0; i < this.food.length; i++) {
      const f = this.food[i];
      if (!hasWorldClearance(f, f.r, this.walls)) {
        const p = correctiveClearPoint(this.rng, occupancy('food', i), 0.11, f.r, this.walls);
        f.x = p.x;
        f.y = p.y;
      }
    }

    for (let i = 0; i < this.hazards.length; i++) {
      const h = this.hazards[i];
      if (!hasWorldClearance(h, h.r, this.walls)) {
        const p = correctiveClearPoint(this.rng, occupancy('hazard', i), 0.14, h.r, this.walls);
        h.x = p.x;
        h.y = p.y;
      }
    }
  }

  nearestFood() {
    if (!this.food.length) return null;
    let best = this.food[0], bd = dist(this.agent, best);
    for (let i = 1; i < this.food.length; i++) {
      const d = dist(this.agent, this.food[i]);
      if (d < bd) { bd = d; best = this.food[i]; }
    }
    return best;
  }
  nearestFoodDistance() { const f = this.nearestFood(); return f ? dist(this.agent, f) : 1.4; }

  rayDistance(angle) {
    const a = this.agent;
    const dx = Math.cos(a.angle + angle), dy = Math.sin(a.angle + angle);
    const maxD = CONFIG.world.sensorRange;
    let best = maxD;
    const step = 0.018;
    for (let d = step; d <= maxD; d += step) {
      const x = a.x + dx * d, y = a.y + dy * d;
      if (x < 0.02 || x > 0.98 || y < 0.02 || y > 0.98) return d;
      for (const hz of this.hazards) if (Math.hypot(x - hz.x, y - hz.y) < hz.r) return d;
      for (const w of this.walls) if (x > w.x && x < w.x + w.w && y > w.y && y < w.y + w.h) return d;
    }
    return best;
  }

  observe() {
    const a = this.agent;
    const f = this.nearestFood();
    let relX = 0, relY = 0, fd = 1;
    if (f) {
      const dx = f.x - a.x, dy = f.y - a.y;
      const c = Math.cos(-a.angle), s = Math.sin(-a.angle);
      relX = clamp((dx * c - dy * s) / 0.75, -1, 1);
      relY = clamp((dx * s + dy * c) / 0.75, -1, 1);
      fd = clamp(Math.hypot(dx, dy) / 1.2, 0, 1);
    }
    const speed = Math.hypot(a.vx, a.vy) / CONFIG.physics.maxSpeed;
    const ray0 = this.rayDistance(CONFIG.world.rayAngles[0]) / CONFIG.world.sensorRange;
    const ray1 = this.rayDistance(CONFIG.world.rayAngles[1]) / CONFIG.world.sensorRange;
    const ray2 = this.rayDistance(CONFIG.world.rayAngles[2]) / CONFIG.world.sensorRange;
    const obs = new Float64Array([
      relX, relY, fd,
      1 - ray0, 1 - ray1, 1 - ray2,
      clamp(speed, 0, 1),
      clamp(a.omega / CONFIG.physics.maxAngularSpeed, -1, 1),
      clamp(a.energy, 0, 1),
      1,
    ]);
    for (const v of obs) if (!Number.isFinite(v)) throw new Error('Non-finite observation');
    return obs;
  }

  collidesWall(x, y) {
    const r = CONFIG.world.agentRadius;
    if (x < r || x > 1-r || y < r || y > 1-r) return true;
    for (const w of this.walls) {
      const cx = clamp(x, w.x, w.x + w.w), cy = clamp(y, w.y, w.y + w.h);
      if (Math.hypot(x - cx, y - cy) < r) return true;
    }
    return false;
  }

  step(action) {
    if (this.done) return { obs: this.observe(), reward: 0, done: true, info: this.info() };
    const a = this.agent, p = CONFIG.physics, e = CONFIG.energy, rw = CONFIG.rewards;
    this.stepCount++;
    let energyCost = e.baseDrain;
    const thrusting = action === 1 || action === 5 || action === 6;
    const turnLeft = action === 3 || action === 5;
    const turnRight = action === 4 || action === 6;
    if (thrusting) {
      a.vx += Math.cos(a.angle) * p.accel;
      a.vy += Math.sin(a.angle) * p.accel;
      energyCost += e.thrustDrain;
    }
    if (action === 2) {
      a.vx *= 0.72; a.vy *= 0.72;
    }
    if (turnLeft) { a.omega -= p.turnAccel; energyCost += e.turnDrain; }
    if (turnRight) { a.omega += p.turnAccel; energyCost += e.turnDrain; }
    a.vx *= p.drag; a.vy *= p.drag; a.omega *= p.angularDrag;
    const sp = Math.hypot(a.vx, a.vy);
    if (sp > p.maxSpeed) { a.vx *= p.maxSpeed / sp; a.vy *= p.maxSpeed / sp; }
    a.omega = clamp(a.omega, -p.maxAngularSpeed, p.maxAngularSpeed);
    a.angle = wrapAngle(a.angle + a.omega);
    const ox = a.x, oy = a.y;
    const nx = a.x + a.vx, ny = a.y + a.vy;
    let wallPenalty = 0;
    if (this.collidesWall(nx, ny)) {
      a.vx *= -0.18; a.vy *= -0.18;
      a.x = ox; a.y = oy; wallPenalty = rw.wall; this.wallHits++;
    } else { a.x = nx; a.y = ny; }

    a.energy = clamp(a.energy - energyCost, 0, 1);
    const rewardParts = { survival: rw.survival, energy: rw.energyScale * energyCost, wall: wallPenalty, food: 0, hazard: 0, approach: 0, death: 0 };
    let reward = rewardParts.survival + rewardParts.energy + rewardParts.wall;
    let ate = false;
    for (let i = this.food.length - 1; i >= 0; i--) {
      if (dist(a, this.food[i]) < CONFIG.world.agentRadius + CONFIG.world.foodRadius) {
        this.food.splice(i, 1);
        a.energy = clamp(a.energy + e.foodGain, 0, 1);
        rewardParts.food += rw.food; reward += rw.food;
        this.foodCollected++;
        ate = true;
        const occupiedForFood = [{x:a.x,y:a.y,r:0.12}, ...this.hazards];
        let q = clearPoint(this.rng, occupiedForFood, 0.12);
        if (!hasWorldClearance(q, CONFIG.world.foodRadius, this.walls)) {
          q = correctiveClearPoint(this.rng, [...occupiedForFood, ...this.food], 0.12, CONFIG.world.foodRadius, this.walls);
        }
        this.food.push({ ...q, r: CONFIG.world.foodRadius });
      }
    }
    for (const hz of this.hazards) {
      if (dist(a, hz) < CONFIG.world.agentRadius + hz.r) {
        rewardParts.hazard += rw.hazard; reward += rw.hazard;
        this.hazardHits++;
        a.energy = 0;
        break;
      }
    }
    const newD = this.nearestFoodDistance();
    if (!ate) { rewardParts.approach = clamp((this.prevFoodDist - newD) * rw.approachScale, -0.01, 0.01); reward += rewardParts.approach; }
    this.prevFoodDist = newD;
    if (a.energy <= 0) { rewardParts.death = rw.death; reward += rw.death; this.done = true; }
    if (this.stepCount >= CONFIG.world.maxSteps) this.done = true;
    this.lastRewardParts = rewardParts;
    this.totalReward += reward;
    if (!Number.isFinite(reward)) throw new Error('Non-finite reward');
    return { obs: this.observe(), reward, done: this.done, info: this.info() };
  }

  info() {
    return {
      seed: this.seed,
      stageId: this.stage.id ?? 0,
      stageName: this.stage.name ?? 'Unknown',
      steps: this.stepCount,
      food: this.foodCollected,
      hazardHits: this.hazardHits,
      wallHits: this.wallHits,
      energy: this.agent.energy,
      survived: this.stepCount >= CONFIG.world.maxSteps,
      totalReward: this.totalReward,
      rewardParts: this.lastRewardParts || { survival:0, energy:0, wall:0, food:0, hazard:0, approach:0, death:0 },
    };
  }
}
