import { ACTIONS } from '../config.js';

const SENSOR_NAMES = ['food x','food y','food dist','danger L','danger F','danger R','speed','turn rate','energy'];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const TAU = Math.PI * 2;

export class CuriosityRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.nodes = [];
    this.telemetry = null;
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

  draw(module, telemetry, now = performanceNow()) {
    this.telemetry = telemetry || null;
    this.nodes = [];
    const { w, h } = this.resize(), c = this.ctx;
    c.clearRect(0, 0, w, h);
    c.fillStyle = '#040910'; c.fillRect(0, 0, w, h);
    if (!telemetry?.obs || !telemetry?.prediction || !telemetry?.nextObs) {
      c.fillStyle = '#78909c'; c.font = '12px system-ui';
      c.fillText('Curiosity starts producing real prediction traces during LEARN.', 14, 24);
      return;
    }

    const input = new Float64Array(module.inputSize);
    for (let i = 0; i < telemetry.obs.length; i++) input[i] = telemetry.obs[i];
    if (Number.isInteger(telemetry.action)) input[module.obsSize + telemetry.action] = 1;
    const hidden = telemetry.hidden, pred = telemetry.prediction, actual = telemetry.nextObs;
    const inPts = layout(input.length, w * .10, h, 28, h - 24);
    const hidPts = layout(hidden.length, w * .49, h, 28, h - 24);
    const outPts = layout(pred.length, w * .82, h, 28, h - 24);
    const novelty = clamp(Number(telemetry.novelty) || 0, 0, 1);
    const meanError = clamp(Number(telemetry.error) || 0, 0, 1);

    drawPredictionHalo(c, w, h, novelty, meanError, now);

    const edges = [];
    for (let i = 0; i < hidden.length; i++) for (let j = 0; j < input.length; j++) {
      const weight = module.params.w1[i * input.length + j];
      edges.push(makeEdge(inPts[j], hidPts[i], weight, Math.abs(input[j] * weight), i * 29 + j));
    }
    for (let k = 0; k < pred.length; k++) for (let i = 0; i < hidden.length; i++) {
      const weight = module.params.w2[k * hidden.length + i];
      edges.push(makeEdge(hidPts[i], outPts[k], weight, Math.abs(hidden[i] * weight), 500 + k * 31 + i));
    }
    edges.sort((a, b) => b.act - a.act);
    c.lineCap = 'round';
    for (const e of edges.slice(0, 180).reverse()) {
      const activity = clamp(e.act, 0, 1), alpha = .025 + .40 * activity;
      c.strokeStyle = e.w >= 0 ? `rgba(92,215,238,${alpha})` : `rgba(236,86,145,${alpha})`;
      c.lineWidth = .35 + 1.35 * activity;
      pathEdge(c, e); c.stroke();
    }
    drawPulses(c, edges.slice(0, 14), now);

    for (let i = 0; i < inPts.length; i++) drawNode(c, inPts[i], input[i]);
    for (let i = 0; i < hidPts.length; i++) drawNode(c, hidPts[i], hidden[i]);
    for (let k = 0; k < outPts.length; k++) {
      const err = Math.abs(pred[k] - actual[k]), delta = actual[k] - pred[k];
      drawNode(c, outPts[k], pred[k]);
      const ghost = { x: outPts[k].x + 10 + 13 * clamp(err * 4, 0, 1), y: outPts[k].y + Math.sign(delta || 1) * 5 * clamp(err * 5, 0, 1) };
      c.save(); c.globalCompositeOperation = 'lighter';
      c.strokeStyle = `rgba(246,187,91,${.10 + .65 * clamp(err * 3, 0, 1)})`; c.lineWidth = .7 + 1.1 * clamp(err * 3, 0, 1);
      c.beginPath(); c.moveTo(outPts[k].x, outPts[k].y); c.lineTo(ghost.x, ghost.y); c.stroke();
      c.fillStyle = `rgba(255,206,116,${.25 + .65 * clamp(err * 3, 0, 1)})`; c.shadowBlur = 7 * clamp(err * 4, 0, 1); c.shadowColor = 'rgba(255,187,91,.9)';
      c.beginPath(); c.arc(ghost.x, ghost.y, 1.8 + 1.5 * clamp(Math.abs(actual[k]), 0, 1), 0, TAU); c.fill();
      c.beginPath(); c.arc(outPts[k].x, outPts[k].y, 7 + 8 * clamp(err * 2, 0, 1), 0, TAU);
      c.strokeStyle = `rgba(246,187,91,${.08 + .65 * clamp(err * 3, 0, 1)})`; c.lineWidth = .8 + 1.4 * clamp(err * 3, 0, 1); c.stroke();
      c.restore();
      this.nodes.push({ ...outPts[k], index: k, pred: pred[k], actual: actual[k], error: err });
    }

    c.font = '10px system-ui'; c.fillStyle = 'rgba(175,210,224,.55)';
    c.fillText('STATE + ACTION', w * .035, 14); c.fillText('PREDICTOR', w * .43, 14); c.fillText('PREDICTED → ACTUAL', w * .72, 14);
    c.fillStyle = 'rgba(220,240,247,.75)'; c.textAlign = 'right';
    for (let k = 0; k < outPts.length; k++) c.fillText(`${SENSOR_NAMES[k]} ${pred[k].toFixed(2)}→${actual[k].toFixed(2)}`, w - 7, outPts[k].y + 3);
    c.textAlign = 'left'; c.fillStyle = 'rgba(144,207,224,.72)';
    c.fillText(`action ${ACTIONS[telemetry.action] || telemetry.action}`, 8, h - 7);
  }

  inspectAt(clientX, clientY) {
    if (!this.nodes.length) return null;
    const r = this.canvas.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    let best = null, dist = 18;
    for (const n of this.nodes) { const d = Math.hypot(x - n.x, y - n.y); if (d < dist) { best = n; dist = d; } }
    if (!best) return null;
    return `${SENSOR_NAMES[best.index]} predicted ${best.pred.toFixed(4)} • actual ${best.actual.toFixed(4)} • |error| ${best.error.toFixed(4)}`;
  }
}

function layout(n, x, h, top, bottom) {
  const span = Math.max(1, bottom - top);
  return Array.from({ length: n }, (_, i) => ({ x, y: top + (n === 1 ? span / 2 : (i / (n - 1)) * span) }));
}
function drawNode(c, p, v) {
  const a = clamp(Math.abs(v), 0, 1);
  c.save(); if (a > .4) { c.shadowBlur = 5 + 7 * a; c.shadowColor = v >= 0 ? 'rgba(90,225,255,.75)' : 'rgba(255,82,160,.75)'; }
  c.beginPath(); c.arc(p.x, p.y, 2.8 + 2.4 * a, 0, TAU);
  c.fillStyle = v >= 0 ? `rgba(111,226,245,${.24 + .72 * a})` : `rgba(239,98,152,${.24 + .72 * a})`;
  c.fill(); c.strokeStyle = 'rgba(235,250,255,.38)'; c.lineWidth = .6; c.stroke(); c.restore();
}
function makeEdge(a, b, w, act, phase) {
  const mx = (a.x + b.x) / 2;
  return { a, b, c1: { x: mx, y: a.y }, c2: { x: mx, y: b.y }, w, act, phase };
}
function pathEdge(c, e) { c.beginPath(); c.moveTo(e.a.x, e.a.y); c.bezierCurveTo(e.c1.x, e.c1.y, e.c2.x, e.c2.y, e.b.x, e.b.y); }
function cubicPoint(e, t) {
  const u = 1 - t, tt = t * t, uu = u * u;
  return { x: uu * u * e.a.x + 3 * uu * t * e.c1.x + 3 * u * tt * e.c2.x + tt * t * e.b.x, y: uu * u * e.a.y + 3 * uu * t * e.c1.y + 3 * u * tt * e.c2.y + tt * t * e.b.y };
}
function drawPulses(c, edges, now) {
  c.save(); c.globalCompositeOperation = 'lighter';
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i], s = clamp(e.act * 1.7, 0, 1); if (s < .05) continue;
    const t = (now * (.00018 + (i % 4) * .000025) + (e.phase % 83) / 83) % 1, q = cubicPoint(e, t);
    c.fillStyle = e.w >= 0 ? `rgba(126,240,255,${.35 + .55 * s})` : `rgba(255,123,190,${.35 + .55 * s})`;
    c.shadowBlur = 7 + 9 * s; c.shadowColor = e.w >= 0 ? 'rgba(80,225,255,.9)' : 'rgba(255,84,169,.9)';
    c.beginPath(); c.arc(q.x, q.y, 1.1 + 2 * s, 0, TAU); c.fill();
  }
  c.restore();
}
function drawPredictionHalo(c, w, h, novelty, error, now) {
  const strength = clamp(novelty * .8 + error * 4, 0, 1); if (strength <= .01) return;
  c.save(); c.globalCompositeOperation = 'lighter';
  const cx = w * .49, cy = h * .5, a = now * .0012;
  c.strokeStyle = `rgba(255,92,177,${.08 + .35 * strength})`; c.lineWidth = .8 + 1.4 * strength; c.shadowBlur = 10 + 12 * strength; c.shadowColor = 'rgba(255,79,175,.85)';
  c.beginPath(); c.ellipse(cx, cy, w * .09, h * .39, 0, a, a + Math.PI * (1.1 + strength * .5)); c.stroke();
  c.strokeStyle = `rgba(87,221,255,${.07 + .30 * strength})`; c.shadowColor = 'rgba(87,221,255,.85)';
  c.beginPath(); c.ellipse(cx, cy, w * .07, h * .35, 0, -a * .8, -a * .8 + Math.PI * 1.3); c.stroke();
  c.restore();
}
function performanceNow() { return globalThis.performance?.now?.() ?? Date.now(); }
