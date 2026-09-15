import { ACTIONS, CONFIG } from '../config.js';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const TAU = Math.PI * 2;

export class WorldRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.world = null;
    this.showSensors = true;
    this.overlayMode = 'BOTH';
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
    const showAttention = this.overlayMode === 'BOTH' || this.overlayMode === 'ATTENTION';
    const showEcho = this.overlayMode === 'BOTH' || this.overlayMode === 'ECHO';

    if (showAttention) drawAttentionField(c, world, cognitive, sx, sy, w, h, now);

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

    if (showAttention) {
      drawFoodSalience(c, world, cognitive, sx, sy, px, py, scale, now);
      drawDangerFocus(c, world, cognitive, px, py, w, h, now);
      drawEnergyPressure(c, cognitive, px, py, scale, now);
    }
    if (showEcho) drawPredictionEcho(c, world, cognitive?.predictionEcho, px, py, w, h, scale, now);

    drawAgentCognitiveFx(c, a, cognitive, px, py, scale, now);

    c.save(); c.translate(px, py); c.rotate(a.angle);
    c.beginPath(); c.moveTo(14, 0); c.lineTo(-10, -8); c.lineTo(-6, 0); c.lineTo(-10, 8); c.closePath();
    c.fillStyle = '#d9f7ff'; c.fill(); c.strokeStyle = '#6dd9ff'; c.stroke(); c.restore();
    c.font = '12px system-ui'; c.fillStyle = 'rgba(220,240,247,.75)';
    c.fillText(`${mode} • seed ${world.seed}`, 10, 18);
    drawOverlayLegend(c, this.overlayMode, cognitive?.predictionEcho, w, h);
  }
}

function drawAttentionField(c, world, fx, sx, sy, w, h, now) {
  if (!fx) return;
  const influence = Array.isArray(fx.inputInfluence) ? fx.inputInfluence : [];
  const food = clamp(Math.max(influence[0] || 0, influence[1] || 0, influence[2] || 0), 0, 1);
  const danger = clamp(Math.max(influence[3] || 0, influence[4] || 0, influence[5] || 0), 0, 1);
  const energy = clamp(influence[8] || 0, 0, 1);
  const pulse = .76 + .24 * Math.sin(now * .0048);

  // A soft, data-driven light field sits under the physical world. Stronger real
  // sensory influence produces brighter field energy; it never affects physics.
  const nearest = world.nearestFood?.();
  if (nearest && food > .04) {
    const x = sx(nearest.x), y = sy(nearest.y), r = Math.max(34, Math.min(w, h) * (.10 + .08 * food * pulse));
    const g = c.createRadialGradient(x, y, 2, x, y, r);
    g.addColorStop(0, `rgba(72,245,208,${.09 + .14 * food})`);
    g.addColorStop(.28, `rgba(42,198,255,${.035 + .08 * food})`);
    g.addColorStop(1, 'rgba(42,198,255,0)');
    c.fillStyle = g; c.fillRect(x - r, y - r, r * 2, r * 2);
  }
  if (danger > .04) {
    const a = world.agent, px = sx(a.x), py = sy(a.y);
    for (let i = 0; i < CONFIG.world.rayAngles.length; i++) {
      const salience = clamp(influence[3 + i] || 0, 0, 1); if (salience < .04) continue;
      const angle = a.angle + CONFIG.world.rayAngles[i], d = world.rayDistance(CONFIG.world.rayAngles[i]);
      const x = px + Math.cos(angle) * d * w, y = py + Math.sin(angle) * d * h;
      const r = 18 + 34 * salience;
      const g = c.createRadialGradient(x, y, 1, x, y, r);
      g.addColorStop(0, `rgba(255,77,153,${.08 + .18 * salience})`);
      g.addColorStop(.35, `rgba(255,123,88,${.025 + .08 * salience})`);
      g.addColorStop(1, 'rgba(255,77,153,0)');
      c.fillStyle = g; c.fillRect(x - r, y - r, r * 2, r * 2);
    }
  }
  if (energy > .08) {
    const a = world.agent, x = sx(a.x), y = sy(a.y), r = 24 + 28 * energy;
    const g = c.createRadialGradient(x, y, 4, x, y, r);
    g.addColorStop(0, `rgba(124,104,255,${.02 + .07 * energy})`);
    g.addColorStop(1, 'rgba(124,104,255,0)');
    c.fillStyle = g; c.fillRect(x - r, y - r, r * 2, r * 2);
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

function drawDangerFocus(c, world, fx, px, py, w, h, now) {
  const influence = Array.isArray(fx?.inputInfluence) ? fx.inputInfluence : [];
  const a = world.agent;
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < CONFIG.world.rayAngles.length; i++) {
    const salience = clamp(influence[3 + i] || 0, 0, 1); if (salience < .16) continue;
    const ang = a.angle + CONFIG.world.rayAngles[i], d = world.rayDistance(CONFIG.world.rayAngles[i]);
    const x = px + Math.cos(ang) * d * w, y = py + Math.sin(ang) * d * h;
    const r = 5 + 6 * salience + 2 * Math.sin(now * .007 + i);
    c.strokeStyle = `rgba(255,91,160,${.16 + .48 * salience})`; c.lineWidth = .7 + 1.5 * salience;
    c.shadowBlur = 7 + 9 * salience; c.shadowColor = 'rgba(255,78,160,.82)';
    c.beginPath(); c.arc(x, y, r, 0, TAU); c.stroke();
  }
  c.restore();
}

function drawEnergyPressure(c, fx, px, py, scale, now) {
  const influence = Array.isArray(fx?.inputInfluence) ? fx.inputInfluence : [];
  const salience = clamp(influence[8] || 0, 0, 1); if (salience < .18) return;
  c.save(); c.globalCompositeOperation = 'lighter';
  c.setLineDash([2, 5]);
  c.strokeStyle = `rgba(158,124,255,${.10 + .30 * salience})`; c.lineWidth = .8 + salience;
  c.beginPath(); c.arc(px, py, scale * (.060 + .006 * Math.sin(now * .004)), 0, TAU); c.stroke();
  c.restore();
}

function drawPredictionEcho(c, world, echo, px, py, w, h, scale, now) {
  if (!echo?.prediction || !echo?.actual) return;
  const pred = echo.prediction, actual = echo.actual;
  if (pred.length < 9 || actual.length < 9) return;
  const strength = clamp((Number(echo.error) || 0) * 5 + (Number(echo.novelty) || 0) * .6, .08, 1);

  // Echo Lens: an agent-local sensor-space projection of one genuine sampled
  // env-0 transition. Keeping it in its own lens avoids pretending a sampled
  // predictor trace is the current physical world state at high train speed.
  const radius = Math.min(50, Math.max(36, Math.min(w, h) * .125));
  const cx = w - radius - 12, cy = radius + 30;
  c.save(); c.globalCompositeOperation = 'lighter';
  const bg = c.createRadialGradient(cx, cy, 2, cx, cy, radius * 1.08);
  bg.addColorStop(0, `rgba(17,45,65,${.16 + .10 * strength})`);
  bg.addColorStop(.72, 'rgba(7,20,31,.18)'); bg.addColorStop(1, 'rgba(7,20,31,0)');
  c.fillStyle = bg; c.beginPath(); c.arc(cx, cy, radius * 1.08, 0, TAU); c.fill();
  c.strokeStyle = `rgba(100,204,232,${.15 + .24 * strength})`; c.lineWidth = .8;
  c.beginPath(); c.arc(cx, cy, radius, 0, TAU); c.stroke();
  c.strokeStyle = 'rgba(106,174,198,.10)';
  c.beginPath(); c.arc(cx, cy, radius * .58, 0, TAU); c.stroke();

  // Food vector: predicted (gold hollow) versus actual next observation (cyan).
  const foodPoint = (arr) => ({
    x: cx + clamp(Number(arr[0]) || 0, -1, 1) * radius * .68,
    y: cy + clamp(Number(arr[1]) || 0, -1, 1) * radius * .68,
  });
  const pf = foodPoint(pred), af = foodPoint(actual);
  c.setLineDash([3, 4]); c.strokeStyle = `rgba(255,193,88,${.18 + .45 * strength})`; c.lineWidth = .7 + .7 * strength;
  c.beginPath(); c.moveTo(cx, cy); c.lineTo(pf.x, pf.y); c.stroke(); c.setLineDash([]);
  c.strokeStyle = `rgba(88,229,255,${.18 + .45 * strength})`;
  c.beginPath(); c.moveTo(cx, cy); c.lineTo(af.x, af.y); c.stroke();
  drawEchoDot(c, pf.x, pf.y, '255,192,88', strength, true, now);
  drawEchoDot(c, af.x, af.y, '88,229,255', strength, false, now);
  c.strokeStyle = `rgba(247,131,205,${.08 + .34 * strength})`; c.lineWidth = .6 + .5 * strength;
  c.beginPath(); c.moveTo(pf.x, pf.y); c.lineTo(af.x, af.y); c.stroke();

  // The three danger channels are shown on fixed agent-local rays. Each gold
  // marker is the model's predicted next proximity; cyan is the actual sample.
  for (let i = 0; i < 3; i++) {
    const angle = -Math.PI / 2 + CONFIG.world.rayAngles[i];
    const pd = (1 - clamp(Number(pred[3 + i]) || 0, 0, 1)) * radius * .82;
    const ad = (1 - clamp(Number(actual[3 + i]) || 0, 0, 1)) * radius * .82;
    const edge = { x: cx + Math.cos(angle) * radius * .82, y: cy + Math.sin(angle) * radius * .82 };
    const p = { x: cx + Math.cos(angle) * pd, y: cy + Math.sin(angle) * pd };
    const q = { x: cx + Math.cos(angle) * ad, y: cy + Math.sin(angle) * ad };
    const err = clamp(Math.abs((Number(pred[3 + i]) || 0) - (Number(actual[3 + i]) || 0)) * 5, 0, 1);
    c.strokeStyle = 'rgba(104,176,201,.12)'; c.lineWidth = .55;
    c.beginPath(); c.moveTo(cx, cy); c.lineTo(edge.x, edge.y); c.stroke();
    c.setLineDash([2, 3]); c.strokeStyle = `rgba(255,190,82,${.10 + .28 * strength})`;
    c.beginPath(); c.moveTo(p.x, p.y); c.lineTo(q.x, q.y); c.stroke(); c.setLineDash([]);
    drawEchoDot(c, p.x, p.y, '255,190,82', Math.max(strength * .55, err), true, now + i * 113);
    drawEchoDot(c, q.x, q.y, '89,226,255', Math.max(strength * .55, err), false, now + i * 113);
  }

  // Speed, turn-rate and energy differences become three short orbit arcs so
  // the lens visualizes all nine predicted dynamic features without more text.
  const orbitValues = [6, 7, 8];
  const orbitColors = ['255,124,198', '170,129,255', '255,205,106'];
  for (let j = 0; j < orbitValues.length; j++) {
    const k = orbitValues[j], rr = radius * (.70 + j * .08);
    const delta = clamp(Math.abs((Number(pred[k]) || 0) - (Number(actual[k]) || 0)) * 6, 0, 1);
    if (delta < .02) continue;
    c.strokeStyle = `rgba(${orbitColors[j]},${.06 + .30 * delta})`; c.lineWidth = .7 + 1.1 * delta;
    c.beginPath(); c.arc(cx, cy, rr, now * .0008 + j, now * .0008 + j + Math.PI * (.30 + .65 * delta)); c.stroke();
  }

  c.shadowBlur = 8 + 12 * strength; c.shadowColor = 'rgba(255,88,184,.72)';
  c.strokeStyle = `rgba(255,115,199,${.05 + .30 * strength})`; c.lineWidth = .7 + strength;
  c.beginPath(); c.arc(cx, cy, radius * (1.0 + .035 * Math.sin(now * .006)), now * .0014, now * .0014 + Math.PI * (1 + strength)); c.stroke();
  c.restore();

  c.save(); c.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = 'center';
  c.fillStyle = 'rgba(139,190,207,.68)'; c.fillText('PREDICTION ECHO', cx, cy + radius + 13); c.restore();
}
function drawEchoDot(c, x, y, rgb, strength, hollow, now) {
  const r = 2.5 + 2.8 * strength + .8 * Math.sin(now * .008);
  c.save(); c.shadowBlur = 6 + 8 * strength; c.shadowColor = `rgba(${rgb},.8)`;
  c.beginPath(); c.arc(x, y, Math.max(1.8, r), 0, TAU);
  if (hollow) { c.strokeStyle = `rgba(${rgb},${.28 + .55 * strength})`; c.lineWidth = .8 + strength; c.stroke(); }
  else { c.fillStyle = `rgba(${rgb},${.20 + .60 * strength})`; c.fill(); }
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

function drawOverlayLegend(c, mode, echo, w, h) {
  if (mode === 'OFF') return;
  c.save(); c.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = 'right';
  const bits = [];
  if (mode === 'BOTH' || mode === 'ATTENTION') bits.push('ATTENTION FIELD');
  if ((mode === 'BOTH' || mode === 'ECHO') && echo) bits.push('ECHO gold→cyan');
  else if (mode === 'BOTH' || mode === 'ECHO') bits.push('ECHO waiting for LEARN');
  c.fillStyle = 'rgba(123,176,195,.66)'; c.fillText(bits.join('  •  '), w - 9, h - 9); c.restore();
}

function performanceNow() { return globalThis.performance?.now?.() ?? Date.now(); }
