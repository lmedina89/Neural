import { ACTIONS } from '../config.js';

const SENSOR_NAMES = ['food x','food y','food dist','danger L','danger F','danger R','speed','turn rate','energy'];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

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

  draw(module, telemetry) {
    this.telemetry = telemetry || null;
    this.nodes = [];
    const { w, h } = this.resize(), c = this.ctx;
    c.clearRect(0, 0, w, h);
    c.fillStyle = '#050a10'; c.fillRect(0, 0, w, h);
    if (!telemetry?.obs || !telemetry?.prediction || !telemetry?.nextObs) {
      c.fillStyle = '#78909c'; c.font = '12px system-ui';
      c.fillText('Curiosity starts producing real prediction traces during LEARN.', 14, 24);
      return;
    }

    const input = new Float64Array(module.inputSize);
    for (let i = 0; i < telemetry.obs.length; i++) input[i] = telemetry.obs[i];
    if (Number.isInteger(telemetry.action)) input[module.obsSize + telemetry.action] = 1;
    const hidden = telemetry.hidden;
    const pred = telemetry.prediction;
    const actual = telemetry.nextObs;
    const inPts = layout(input.length, w * .10, h, 28, h - 24);
    const hidPts = layout(hidden.length, w * .49, h, 28, h - 24);
    const outPts = layout(pred.length, w * .82, h, 28, h - 24);
    const edges = [];
    for (let i = 0; i < hidden.length; i++) for (let j = 0; j < input.length; j++) {
      const weight = module.params.w1[i * input.length + j];
      edges.push({ a: inPts[j], b: hidPts[i], w: weight, act: Math.abs(input[j] * weight) });
    }
    for (let k = 0; k < pred.length; k++) for (let i = 0; i < hidden.length; i++) {
      const weight = module.params.w2[k * hidden.length + i];
      edges.push({ a: hidPts[i], b: outPts[k], w: weight, act: Math.abs(hidden[i] * weight) });
    }
    edges.sort((a, b) => b.act - a.act);
    c.lineCap = 'round';
    for (const e of edges.slice(0, 180)) {
      const activity = clamp(e.act, 0, 1);
      const alpha = .035 + .42 * activity;
      c.strokeStyle = e.w >= 0 ? `rgba(92,215,238,${alpha})` : `rgba(236,86,145,${alpha})`;
      c.lineWidth = .35 + 1.35 * activity;
      c.beginPath(); c.moveTo(e.a.x, e.a.y);
      const mx = (e.a.x + e.b.x) / 2;
      c.bezierCurveTo(mx, e.a.y, mx, e.b.y, e.b.x, e.b.y); c.stroke();
    }

    for (let i = 0; i < inPts.length; i++) drawNode(c, inPts[i], input[i]);
    for (let i = 0; i < hidPts.length; i++) drawNode(c, hidPts[i], hidden[i]);
    for (let k = 0; k < outPts.length; k++) {
      const err = Math.abs(pred[k] - actual[k]);
      drawNode(c, outPts[k], pred[k]);
      c.beginPath(); c.arc(outPts[k].x, outPts[k].y, 7 + 7 * clamp(err, 0, 1), 0, Math.PI * 2);
      c.strokeStyle = `rgba(246,187,91,${.08 + .65 * clamp(err, 0, 1)})`;
      c.lineWidth = .8 + 1.4 * clamp(err, 0, 1); c.stroke();
      this.nodes.push({ ...outPts[k], index: k, pred: pred[k], actual: actual[k], error: err });
    }

    c.font = '10px system-ui'; c.fillStyle = 'rgba(175,210,224,.55)';
    c.fillText('STATE + ACTION', w * .035, 14); c.fillText('PREDICTOR', w * .43, 14); c.fillText('NEXT SENSORY', w * .75, 14);
    c.fillStyle = 'rgba(220,240,247,.75)'; c.textAlign = 'right';
    for (let k = 0; k < outPts.length; k++) {
      c.fillText(`${SENSOR_NAMES[k]} ${pred[k].toFixed(2)}→${actual[k].toFixed(2)}`, w - 7, outPts[k].y + 3);
    }
    c.textAlign = 'left';
    c.fillStyle = 'rgba(144,207,224,.72)';
    c.fillText(`action ${ACTIONS[telemetry.action] || telemetry.action}`, 8, h - 7);
  }

  inspectAt(clientX, clientY) {
    if (!this.nodes.length) return null;
    const r = this.canvas.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    let best = null, dist = 18;
    for (const n of this.nodes) {
      const d = Math.hypot(x - n.x, y - n.y);
      if (d < dist) { best = n; dist = d; }
    }
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
  c.beginPath(); c.arc(p.x, p.y, 2.8 + 2.4 * a, 0, Math.PI * 2);
  c.fillStyle = v >= 0 ? `rgba(111,226,245,${.24 + .72 * a})` : `rgba(239,98,152,${.24 + .72 * a})`;
  c.fill(); c.strokeStyle = 'rgba(235,250,255,.38)'; c.lineWidth = .6; c.stroke();
}
