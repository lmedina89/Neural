import { ACTIONS, CONFIG } from '../config.js';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const TAU = Math.PI * 2;

export class WorldRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.world = null;
    this.showSensors = true;
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: rect.width, h: rect.height };
  }
  draw(world, mode = 'OBSERVE', noveltyTrail = [], cognitive = null, now = performanceNow()) {
    this.world = world;
    const { w, h } = this.resize(), c = this.ctx;
    c.clearRect(0, 0, w, h);
    c.fillStyle = '#071018'; c.fillRect(0, 0, w, h);
    c.strokeStyle = 'rgba(124,180,200,.08)'; c.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      c.beginPath(); c.moveTo((w * i) / 10, 0); c.lineTo((w * i) / 10, h); c.stroke();
      c.beginPath(); c.moveTo(0, (h * i) / 10); c.lineTo(w, (h * i) / 10); c.stroke();
    }
    const sx = x => x * w, sy = y => y * h;

    // Real prediction-surprise trail from the training environment. No random particles.
    if (Array.isArray(noveltyTrail) && noveltyTrail.length) {
      for (let i = 0; i < noveltyTrail.length; i++) {
        const q = noveltyTrail[i];
        const age = noveltyTrail.length > 1 ? i / (noveltyTrail.length - 1) : 1;
        const n = clamp(Number(q.novelty) || 0, 0, 1);
        if (n <= 0) continue;
        c.save(); c.globalCompositeOperation = 'lighter';
        c.beginPath(); c.arc(sx(q.x), sy(q.y), (4 + 16 * n) * (.55 + .45 * age), 0, TAU);
        c.fillStyle = `rgba(246,187,91,${(.015 + .11 * n) * age})`; c.fill();
        c.strokeStyle = `rgba(255,106,183,${(.025 + .22 * n) * age})`; c.lineWidth = .7; c.stroke();
        c.restore();
      }
    }

    for (const wall of world.walls) {
      c.fillStyle = 'rgba(135,155,170,.36)'; c.fillRect(sx(wall.x), sy(wall.y), wall.w * w, wall.h * h);
      c.strokeStyle = 'rgba(190,220,230,.32)'; c.strokeRect(sx(wall.x), sy(wall.y), wall.w * w, wall.h * h);
    }
    for (const hz of world.hazards) {
      c.beginPath(); c.arc(sx(hz.x), sy(hz.y), hz.r * Math.min(w, h), 0, TAU);
      c.fillStyle = 'rgba(234,76,120,.24)'; c.fill(); c.strokeStyle = 'rgba(255,105,145,.85)'; c.stroke();
    }
    for (const f of world.food) {
      c.beginPath(); c.arc(sx(f.x), sy(f.y), CONFIG.world.foodRadius * Math.min(w, h), 0, TAU);
      c.fillStyle = 'rgba(91,231,196,.82)'; c.fill();
      c.beginPath(); c.arc(sx(f.x), sy(f.y), CONFIG.world.foodRadius * Math.min(w, h) * 1.8, 0, TAU);
      c.strokeStyle = 'rgba(91,231,196,.20)'; c.stroke();
    }

    const a = world.agent, px = sx(a.x), py = sy(a.y), scale = Math.min(w, h);
    const influence = Array.isArray(cognitive?.inputInfluence) ? cognitive.inputInfluence : [];
    if (this.showSensors) {
      for (let i = 0; i < CONFIG.world.rayAngles.length; i++) {
        const ang = CONFIG.world.rayAngles[i], d = world.rayDistance(ang);
        const salience = clamp(influence[3 + i] || 0, 0, 1);
        c.save();
        c.strokeStyle = `rgba(${salience > .45 ? '255,103,164' : '114,205,255'},${.14 + .42 * salience})`;
        c.lineWidth = 1 + 1.1 * salience;
        if (salience > .28) { c.shadowBlur = 7 * salience; c.shadowColor = 'rgba(255,103,164,.75)'; }
        c.beginPath(); c.moveTo(px, py); c.lineTo(px + Math.cos(a.angle + ang) * d * w, py + Math.sin(a.angle + ang) * d * h); c.stroke();
        c.restore();
      }
    }

    drawFoodSalience(c, world, cognitive, sx, sy, px, py, scale, now);
    drawAgentCognitiveFx(c, a, cognitive, px, py, scale, now);

    c.save(); c.translate(px, py); c.rotate(a.angle);
    c.beginPath(); c.moveTo(14, 0); c.lineTo(-10, -8); c.lineTo(-6, 0); c.lineTo(-10, 8); c.closePath();
    c.fillStyle = '#d9f7ff'; c.fill(); c.strokeStyle = '#6dd9ff'; c.stroke(); c.restore();
    c.font = '12px system-ui'; c.fillStyle = 'rgba(220,240,247,.75)';
    c.fillText(`${mode} • seed ${world.seed}`, 10, 18);
  }
}

function drawFoodSalience(c, world, fx, sx, sy, px, py, scale, now) {
  const f = world.nearestFood?.();
  if (!f) return;
  const influence = Array.isArray(fx?.inputInfluence) ? fx.inputInfluence : [];
  const salience = clamp(Math.max(influence[0] || 0, influence[1] || 0, influence[2] || 0), 0, 1);
  if (salience < .08) return;
  const x = sx(f.x), y = sy(f.y), pulse = .78 + .22 * Math.sin(now * .006);
  c.save(); c.globalCompositeOperation = 'lighter';
  c.setLineDash([3, 5]); c.strokeStyle = `rgba(91,231,196,${.08 + .28 * salience})`; c.lineWidth = .7 + salience;
  c.beginPath(); c.moveTo(px, py); c.lineTo(x, y); c.stroke(); c.setLineDash([]);
  c.strokeStyle = `rgba(108,255,220,${.18 + .55 * salience})`; c.lineWidth = .8 + 1.2 * salience; c.shadowBlur = 9 * salience; c.shadowColor = 'rgba(91,231,196,.8)';
  c.beginPath(); c.arc(x, y, CONFIG.world.foodRadius * scale * (2.2 + .75 * pulse * salience), 0, TAU); c.stroke();
  c.restore();
}

function drawAgentCognitiveFx(c, agent, fx, px, py, scale, now) {
  if (!fx) return;
  const novelty = clamp(Number(fx.curiosityNovelty) || 0, 0, 1);
  const confidence = clamp(Number(fx.decisionConfidence) || 0, 0, 1);
  const p = clamp(Number(fx.dominantProb) || 0, 0, 1);
  const action = Number.isInteger(fx.dominantAction) ? fx.dominantAction : 0;
  const pulse = .75 + .25 * Math.sin(now * .008);
  c.save(); c.globalCompositeOperation = 'lighter';
  c.strokeStyle = `rgba(94,218,255,${.14 + .34 * confidence})`; c.lineWidth = 1 + confidence; c.shadowBlur = 8 + 8 * confidence; c.shadowColor = 'rgba(76,215,255,.78)';
  c.beginPath(); c.arc(px, py, scale * (.040 + .010 * pulse), now * .001, now * .001 + Math.PI * (1.15 + confidence * .55)); c.stroke();
  if (novelty > .015) {
    c.strokeStyle = `rgba(255,94,178,${.12 + .58 * novelty})`; c.shadowColor = 'rgba(255,85,178,.9)';
    c.beginPath(); c.arc(px, py, scale * (.050 + .012 * pulse), -now * .0014, -now * .0014 + Math.PI * (1 + novelty)); c.stroke();
  }

  let offset = 0, backwards = false, beam = true;
  if (action === 3) offset = -.72;
  else if (action === 4) offset = .72;
  else if (action === 5) offset = -.38;
  else if (action === 6) offset = .38;
  else if (action === 2) { backwards = true; beam = false; }
  else if (action === 0) beam = false;
  if (beam) {
    const ang = agent.angle + offset + (backwards ? Math.PI : 0);
    const len = scale * (.07 + .14 * p);
    const ex = px + Math.cos(ang) * len, ey = py + Math.sin(ang) * len;
    const g = c.createLinearGradient(px, py, ex, ey);
    g.addColorStop(0, `rgba(117,235,255,${.18 + .54 * p})`); g.addColorStop(1, 'rgba(229,92,255,0)');
    c.strokeStyle = g; c.lineWidth = 1 + 2.2 * p; c.shadowBlur = 10 + 9 * p; c.shadowColor = 'rgba(104,224,255,.9)';
    c.beginPath(); c.moveTo(px, py); c.lineTo(ex, ey); c.stroke();
  } else if (action === 2) {
    c.strokeStyle = `rgba(255,187,91,${.16 + .56 * p})`; c.shadowColor = 'rgba(255,187,91,.85)'; c.lineWidth = 1.2 + 1.8 * p;
    c.beginPath(); c.arc(px, py, scale * (.031 + .012 * p * pulse), 0, TAU); c.stroke();
  }
  c.restore();
}

function performanceNow() { return globalThis.performance?.now?.() ?? Date.now(); }
