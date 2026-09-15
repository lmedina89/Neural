import { ACTIONS } from '../config.js';

const INPUT_NAMES = ['food x','food y','food dist','danger L','danger F','danger R','speed','turn rate','energy','bias'];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const TAU = Math.PI * 2;

export class NeuralRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.mode = 'FLOW';
    this.nodes = [];
    this.model = null;
    this.snapshot = null;
    this.geometryKey = '';
    this.geometry = null;
    this.activeEdges = [];
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const d = Math.min(2, globalThis.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * d));
    const h = Math.max(1, Math.round(r.height * d));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    return { w: r.width, h: r.height };
  }

  draw(model, snapshot, fx = {}, now = performanceNow()) {
    this.model = model;
    this.snapshot = snapshot;
    this.nodes = [];
    const { w, h } = this.resize(), c = this.ctx;
    c.clearRect(0, 0, w, h);
    drawBackground(c, w, h, this.mode);
    if (!snapshot) {
      c.fillStyle = '#78909c'; c.font = '13px system-ui';
      c.fillText('Run OBSERVE or PROBE to inspect live activity.', 16, 28);
      return;
    }

    const p = model.params, obs = snapshot.obs, hidden = snapshot.h, probs = snapshot.probs;
    const flow = this.mode === 'FLOW';
    const geometryKey = `${this.mode}:${w.toFixed(2)}:${h.toFixed(2)}:${obs.length}:${hidden.length}:${probs.length}`;
    if (this.geometryKey !== geometryKey || !this.geometry) {
      this.geometryKey = geometryKey;
      this.geometry = buildNeuralGeometry(obs.length, hidden.length, probs.length, w, h, flow);
    }
    const { inPts, hidPts, outPts, sensoryEdges, policyEdges, valueEdges, recurrentEdges } = this.geometry;

    if (flow) drawCognitiveHalo(c, w, h, probs, fx, now);

    // Geometry is stable for a given canvas/layout. Reuse edge objects and only
    // refresh real weights/activity each frame to avoid hundreds of short-lived
    // allocations while preserving the exact same sorting and drawing paths.
    const edges = this.activeEdges; edges.length = 0;
    for (const edge of sensoryEdges) {
      const weight = p.wx[edge.paramIndex]; edge.w = weight; edge.act = Math.abs(obs[edge.obsIndex] * weight); edges.push(edge);
    }
    for (const edge of policyEdges) {
      const weight = p.wp[edge.paramIndex]; edge.w = weight; edge.act = Math.abs(hidden[edge.hiddenIndex] * weight); edges.push(edge);
    }
    for (const edge of valueEdges) {
      const weight = p.wv[edge.hiddenIndex]; edge.w = weight; edge.act = Math.abs(hidden[edge.hiddenIndex] * weight); edges.push(edge);
    }
    if (flow && snapshot.hPrev && p.wh) {
      for (const edge of recurrentEdges) {
        const weight = p.wh[edge.paramIndex], activity = Math.abs(snapshot.hPrev[edge.sourceHiddenIndex] * weight);
        if (activity < 0.012) continue;
        edge.w = weight; edge.act = activity; edges.push(edge);
      }
    }

    const valuePoint = outPts.at(-1);
    edges.sort((a, b) => (this.mode === 'WEIGHTS' ? Math.abs(b.w) - Math.abs(a.w) : b.act - a.act));
    const maxEdges = this.mode === 'STRONGEST' ? 120 : flow ? Math.min(285, edges.length) : Math.min(330, edges.length);
    c.lineCap = 'round';
    for (let k = maxEdges - 1; k >= 0; k--) drawEdge(c, edges[k], this.mode, flow);

    if (flow) drawSignalPulses(c, edges.slice(0, 24), now);

    const inputInfluence = Array.isArray(fx.inputInfluence) ? fx.inputInfluence : [];
    for (let i = 0; i < inPts.length; i++) {
      drawNode(c, inPts[i], obs[i], flow, inputInfluence[i] || 0);
      this.nodes.push({ x: inPts[i].x, y: inPts[i].y, kind: 'input', index: i, value: obs[i] });
    }
    for (let i = 0; i < hidPts.length; i++) {
      drawNode(c, hidPts[i], hidden[i], flow, Math.abs(hidden[i]));
      this.nodes.push({ x: hidPts[i].x, y: hidPts[i].y, kind: 'hidden', index: i, value: hidden[i] });
    }
    for (let i = 0; i < probs.length; i++) {
      const isDominant = i === dominantIndex(probs);
      drawNode(c, outPts[i], probs[i], flow, probs[i], isDominant);
      this.nodes.push({ x: outPts[i].x, y: outPts[i].y, kind: 'output', index: i, value: probs[i] });
    }
    drawNode(c, valuePoint, Math.tanh(snapshot.value), flow, clamp(Math.abs(snapshot.value), 0, 1));
    this.nodes.push({ x: valuePoint.x, y: valuePoint.y, kind: 'value', index: 0, value: snapshot.value });

    if (flow) drawDecisionBeam(c, outPts, probs, w, now);

    c.font = flow ? '10px system-ui' : '11px system-ui';
    c.fillStyle = 'rgba(220,240,247,.72)';
    for (let i = 0; i < inPts.length; i++) c.fillText(INPUT_NAMES[i], inPts[i].x + 9, inPts[i].y + 4);
    for (let i = 0; i < probs.length; i++) {
      c.textAlign = 'right';
      c.fillText(`${ACTIONS[i]} ${(probs[i] * 100).toFixed(0)}%`, outPts[i].x - 9, outPts[i].y + 4);
    }
    c.fillText(`VALUE ${snapshot.value.toFixed(2)}`, valuePoint.x - 9, valuePoint.y + 4);
    c.textAlign = 'left';
    c.fillStyle = 'rgba(175,210,224,.55)';
    c.fillText('SENSORY', w * .035, 14);
    c.fillText(flow ? 'RECURRENT MEMORY CORE' : 'RECURRENT STATE', flow ? w * .39 : w * .44, 14);
    c.fillText('POLICY / VALUE', w * .76, 14);
    if (flow) {
      const novelty = clamp(Number(fx.curiosityNovelty) || 0, 0, 1);
      const reward = Number(fx.reward) || 0;
      c.textAlign = 'center';
      c.fillStyle = 'rgba(143,203,221,.62)';
      c.fillText(`LIVE FLOW • memory loops • novelty ${(novelty * 100).toFixed(0)}% • reward ${reward >= 0 ? '+' : ''}${reward.toFixed(3)}`, w * .52, h - 6);
      c.textAlign = 'left';
    }
  }

  inspectAt(clientX, clientY) {
    if (!this.snapshot || !this.model) return null;
    const r = this.canvas.getBoundingClientRect(), x = clientX - r.left, y = clientY - r.top;
    let best = null, bd = 18;
    for (const n of this.nodes) { const d = Math.hypot(x - n.x, y - n.y); if (d < bd) { bd = d; best = n; } }
    if (!best) return null;
    const p = this.model.params;
    if (best.kind === 'input') return `INPUT ${best.index} (${INPUT_NAMES[best.index]}): activation ${best.value.toFixed(4)}`;
    if (best.kind === 'output') return `POLICY ${ACTIONS[best.index]}: probability ${(best.value * 100).toFixed(2)}%, bias ${p.bp[best.index].toFixed(4)}`;
    if (best.kind === 'value') return `VALUE HEAD: estimate ${best.value.toFixed(4)}, bias ${p.bv[0].toFixed(4)}`;
    const i = best.index, obsN = this.snapshot.obs.length, hN = this.snapshot.h.length;
    const incoming = [];
    for (let j = 0; j < obsN; j++) incoming.push([INPUT_NAMES[j], p.wx[i * obsN + j]]);
    incoming.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const recurrent = [];
    for (let j = 0; j < hN; j++) recurrent.push([j, p.wh[i * hN + j]]);
    recurrent.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const outgoing = [];
    for (let a = 0; a < ACTIONS.length; a++) outgoing.push([ACTIONS[a], p.wp[a * hN + i]]);
    outgoing.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    return `RECURRENT UNIT ${i}: activation ${best.value.toFixed(4)}, bias ${p.bh[i].toFixed(4)} • strongest sensor ${incoming.slice(0, 2).map(x => `${x[0]} ${x[1].toFixed(3)}`).join(', ')} • memory from H${recurrent[0][0]} ${recurrent[0][1].toFixed(3)} • strongest out ${outgoing.slice(0, 2).map(x => `${x[0]} ${x[1].toFixed(3)}`).join(', ')}`;
  }
}

function drawBackground(c, w, h, mode) {
  c.fillStyle = '#040910'; c.fillRect(0, 0, w, h);
  if (mode !== 'FLOW') return;
  const g = c.createRadialGradient(w * .52, h * .5, 10, w * .52, h * .5, Math.max(w, h) * .55);
  g.addColorStop(0, 'rgba(25,77,104,.20)'); g.addColorStop(.48, 'rgba(23,34,78,.10)'); g.addColorStop(1, 'rgba(3,7,12,0)');
  c.fillStyle = g; c.fillRect(0, 0, w, h);
}

function layoutColumn(n, x, h) {
  const top = 30, bottom = h - 18, span = Math.max(1, bottom - top);
  return Array.from({ length: n }, (_, i) => ({ x, y: top + (n === 1 ? span / 2 : (i / (n - 1)) * span) }));
}

function layoutCognitiveCore(n, w, h) {
  const pts = [];
  const cx = w * .52, cy = h * .51;
  for (let i = 0; i < n; i++) {
    const r = .30 + .70 * Math.sqrt((i + .5) / n);
    const a = i * 2.3999632297 - .7;
    const lobe = (i & 1) ? -1 : 1;
    pts.push({
      x: cx + Math.cos(a) * w * .145 * r + lobe * w * .025 * (1 - r * .35),
      y: cy + Math.sin(a) * h * .34 * r,
    });
  }
  return pts;
}

function buildNeuralGeometry(obsN, hiddenN, probsN, w, h, flow) {
  const inPts = layoutColumn(obsN, w * .11, h);
  const hidPts = flow ? layoutCognitiveCore(hiddenN, w, h) : layoutColumn(hiddenN, w * .52, h);
  const outPts = layoutColumn(probsN + 1, w * .89, h);
  const sensoryEdges = [], policyEdges = [], valueEdges = [], recurrentEdges = [];
  for (let i = 0; i < hiddenN; i++) for (let j = 0; j < obsN; j++) {
    const edge = makeEdge(inPts[j], hidPts[i], 0, 0, 'sensory', i * 31 + j);
    edge.paramIndex = i * obsN + j; edge.obsIndex = j; sensoryEdges.push(edge);
  }
  for (let a = 0; a < probsN; a++) for (let i = 0; i < hiddenN; i++) {
    const edge = makeEdge(hidPts[i], outPts[a], 0, 0, 'policy', a * 37 + i);
    edge.paramIndex = a * hiddenN + i; edge.hiddenIndex = i; policyEdges.push(edge);
  }
  const valuePoint = outPts.at(-1);
  for (let i = 0; i < hiddenN; i++) {
    const edge = makeEdge(hidPts[i], valuePoint, 0, 0, 'value', 700 + i);
    edge.hiddenIndex = i; valueEdges.push(edge);
  }
  if (flow) for (let i = 0; i < hiddenN; i++) for (let j = 0; j < hiddenN; j++) {
    if (i === j) continue;
    const edge = makeRecurrentEdge(hidPts[j], hidPts[i], 0, 0, i * 97 + j);
    edge.paramIndex = i * hiddenN + j; edge.sourceHiddenIndex = j; recurrentEdges.push(edge);
  }
  return { inPts, hidPts, outPts, sensoryEdges, policyEdges, valueEdges, recurrentEdges };
}

function makeEdge(a, b, w, act, kind, phase) {
  const mx = (a.x + b.x) / 2;
  return { a, b, c1: { x: mx, y: a.y }, c2: { x: mx, y: b.y }, w, act, kind, phase };
}

function makeRecurrentEdge(a, b, w, act, phase) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / len, ny = dx / len;
  const bend = 10 + Math.min(34, len * .42) * ((phase & 1) ? 1 : -1);
  return {
    a, b,
    c1: { x: a.x + dx * .35 + nx * bend, y: a.y + dy * .35 + ny * bend },
    c2: { x: a.x + dx * .65 + nx * bend, y: a.y + dy * .65 + ny * bend },
    w, act, kind: 'memory', phase,
  };
}

function drawEdge(c, e, mode, flow) {
  const mag = clamp(Math.abs(e.w), 0, 1.5), act = clamp(e.act, 0, 1);
  const base = mode === 'WEIGHTS' ? mag : act;
  let alpha = mode === 'WEIGHTS' ? .08 + .32 * mag : .025 + .50 * act;
  if (e.kind === 'memory') alpha *= .78;
  if (e.kind === 'value') alpha *= .82;
  c.strokeStyle = e.w >= 0 ? `rgba(75,211,245,${alpha})` : `rgba(246,75,151,${alpha})`;
  c.lineWidth = .30 + (flow ? 1.55 : 1.8) * base;
  c.beginPath(); c.moveTo(e.a.x, e.a.y); c.bezierCurveTo(e.c1.x, e.c1.y, e.c2.x, e.c2.y, e.b.x, e.b.y); c.stroke();
}

function drawSignalPulses(c, edges, now) {
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const strength = clamp(e.act * 1.5, 0, 1);
    if (strength < .05) continue;
    const t = ((now * (.00016 + .000035 * (i % 5))) + ((e.phase % 101) / 101)) % 1;
    const q = cubicPoint(e, t);
    c.shadowBlur = 8 + 9 * strength;
    c.shadowColor = e.w >= 0 ? 'rgba(69,226,255,.95)' : 'rgba(255,75,163,.95)';
    c.fillStyle = e.w >= 0 ? `rgba(130,241,255,${.42 + .55 * strength})` : `rgba(255,131,192,${.42 + .55 * strength})`;
    c.beginPath(); c.arc(q.x, q.y, 1.2 + 2.3 * strength, 0, TAU); c.fill();
  }
  c.restore();
}

function drawCognitiveHalo(c, w, h, probs, fx, now) {
  const cx = w * .52, cy = h * .51;
  const best = sortedProbIndices(probs);
  const confidence = clamp((probs[best[0]] - probs[best[1]]) * 1.6, 0, 1);
  const novelty = clamp(Number(fx.curiosityNovelty) || 0, 0, 1);
  const reward = clamp(Math.abs(Number(fx.reward) || 0) * 1.4, 0, 1);
  const phase = now * .00035;
  const rings = [
    { rx: w * .19, ry: h * .39, color: '89,221,255', speed: 1, width: 1.2 },
    { rx: w * .17, ry: h * .35, color: '244,75,170', speed: -1.3, width: 1.0 },
    { rx: w * .205, ry: h * .32, color: '255,179,74', speed: .72, width: .9 },
  ];
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < rings.length; i++) {
    const r = rings[i];
    const a = phase * r.speed + i * 1.7;
    const span = .75 + confidence * .7 + novelty * .45;
    c.strokeStyle = `rgba(${r.color},${.12 + .20 * confidence + .16 * novelty + .08 * reward})`;
    c.lineWidth = r.width + confidence * .8;
    c.shadowBlur = 8 + 14 * (confidence + novelty) * .5;
    c.shadowColor = `rgba(${r.color},.75)`;
    c.beginPath(); c.ellipse(cx, cy, r.rx, r.ry, i === 2 ? .22 : 0, a, a + span); c.stroke();
    c.beginPath(); c.ellipse(cx, cy, r.rx, r.ry, i === 2 ? .22 : 0, a + Math.PI, a + Math.PI + span * .72); c.stroke();
  }
  if (novelty > .02 || reward > .02) {
    const count = Math.min(10, 2 + Math.floor((novelty + reward) * 8));
    for (let i = 0; i < count; i++) {
      const a = phase * (1.3 + i * .04) + i * 2.17;
      const rx = w * (.17 + .025 * ((i % 3) / 2));
      const ry = h * (.33 + .04 * ((i % 4) / 3));
      const x = cx + Math.cos(a) * rx, y = cy + Math.sin(a) * ry;
      drawSpark(c, x, y, 2 + 4 * clamp(novelty + reward, 0, 1), novelty > reward ? '255,98,184' : '255,194,92');
    }
  }
  c.restore();
}

function drawDecisionBeam(c, outPts, probs, w, now) {
  const idx = dominantIndex(probs), p = clamp(probs[idx], 0, 1), q = outPts[idx];
  const pulse = .76 + .24 * Math.sin(now * .007);
  c.save(); c.globalCompositeOperation = 'lighter';
  const g = c.createLinearGradient(q.x, q.y, w, q.y);
  g.addColorStop(0, `rgba(117,235,255,${.25 + .55 * p})`);
  g.addColorStop(.65, `rgba(226,91,255,${.10 + .28 * p})`);
  g.addColorStop(1, 'rgba(226,91,255,0)');
  c.strokeStyle = g; c.lineWidth = 1 + 2.8 * p * pulse;
  c.shadowBlur = 12 + 12 * p; c.shadowColor = 'rgba(100,225,255,.9)';
  c.beginPath(); c.moveTo(q.x + 7, q.y); c.lineTo(w - 2, q.y); c.stroke();
  c.strokeStyle = `rgba(165,244,255,${.22 + .55 * p})`; c.lineWidth = 1.1;
  c.beginPath(); c.arc(q.x, q.y, 9 + 6 * p * pulse, 0, TAU); c.stroke();
  c.restore();
}

function drawNode(c, p, v, glow = false, emphasis = 0, dominant = false) {
  const a = clamp(Math.abs(v), 0, 1), e = clamp(emphasis, 0, 1);
  c.save();
  if (glow && (e > .12 || dominant)) {
    c.shadowBlur = 5 + 11 * Math.max(e, dominant ? .65 : 0);
    c.shadowColor = v >= 0 ? 'rgba(94,226,255,.9)' : 'rgba(255,82,160,.9)';
  }
  c.beginPath(); c.arc(p.x, p.y, 3.2 + 3.3 * a + (dominant ? 1.2 : 0), 0, TAU);
  c.fillStyle = v >= 0 ? `rgba(111,226,245,${.25 + .75 * a})` : `rgba(239,98,152,${.25 + .75 * a})`;
  c.fill(); c.strokeStyle = dominant ? 'rgba(255,241,176,.85)' : 'rgba(235,250,255,.45)'; c.lineWidth = dominant ? 1.15 : .65; c.stroke();
  c.restore();
}

function drawSpark(c, x, y, r, rgb) {
  c.strokeStyle = `rgba(${rgb},.82)`; c.lineWidth = .8; c.shadowBlur = 10; c.shadowColor = `rgba(${rgb},.95)`;
  c.beginPath(); c.moveTo(x - r, y); c.lineTo(x + r, y); c.moveTo(x, y - r); c.lineTo(x, y + r); c.stroke();
  c.fillStyle = `rgba(${rgb},.95)`; c.beginPath(); c.arc(x, y, 1.2, 0, TAU); c.fill();
}

function cubicPoint(e, t) {
  const u = 1 - t, tt = t * t, uu = u * u;
  return {
    x: uu * u * e.a.x + 3 * uu * t * e.c1.x + 3 * u * tt * e.c2.x + tt * t * e.b.x,
    y: uu * u * e.a.y + 3 * uu * t * e.c1.y + 3 * u * tt * e.c2.y + tt * t * e.b.y,
  };
}

function sortedProbIndices(probs) { return Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]); }
function dominantIndex(probs) { let k = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[k]) k = i; return k; }
function performanceNow() { return globalThis.performance?.now?.() ?? Date.now(); }
