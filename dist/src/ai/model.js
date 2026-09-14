import { CONFIG } from '../config.js';
import { PRNG } from '../utils/prng.js';

const { obsSize, hiddenSize, actionSize } = CONFIG.model;

function randnArray(n, rng, scale) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}
function zeros(n) { return new Float64Array(n); }
function softmax(logits) {
  let m = -Infinity;
  for (const x of logits) if (x > m) m = x;
  const out = new Float64Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - m); s += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}
function cloneArray(a) { return new Float64Array(a); }

export class RecurrentActorCritic {
  constructor(seed = 1234) {
    const rng = new PRNG(seed);
    const sx = Math.sqrt(2 / (obsSize + hiddenSize));
    const sh = Math.sqrt(1 / hiddenSize);
    const so = Math.sqrt(1 / hiddenSize);
    this.params = {
      wx: randnArray(hiddenSize * obsSize, rng, sx),
      wh: randnArray(hiddenSize * hiddenSize, rng, sh * 0.5),
      bh: zeros(hiddenSize),
      wp: randnArray(actionSize * hiddenSize, rng, so),
      bp: zeros(actionSize),
      wv: randnArray(hiddenSize, rng, so),
      bv: new Float64Array([0]),
    };
    this.last = null;
  }

  zeroHidden() { return zeros(hiddenSize); }

  forward(obs, hPrev, capture = true) {
    const p = this.params;
    const h = new Float64Array(hiddenSize);
    const pre = capture ? new Float64Array(hiddenSize) : null;
    for (let i = 0; i < hiddenSize; i++) {
      let z = p.bh[i];
      const xo = i * obsSize;
      const ho = i * hiddenSize;
      for (let j = 0; j < obsSize; j++) z += p.wx[xo + j] * obs[j];
      for (let j = 0; j < hiddenSize; j++) z += p.wh[ho + j] * hPrev[j];
      if (pre) pre[i] = z;
      h[i] = Math.tanh(z);
    }
    const logits = new Float64Array(actionSize);
    for (let a = 0; a < actionSize; a++) {
      let z = p.bp[a];
      const o = a * hiddenSize;
      for (let i = 0; i < hiddenSize; i++) z += p.wp[o + i] * h[i];
      logits[a] = z;
    }
    const probs = softmax(logits);
    let value = p.bv[0];
    for (let i = 0; i < hiddenSize; i++) value += p.wv[i] * h[i];
    if (!capture) return { h, logits, probs, value };
    this.last = { obs: cloneArray(obs), hPrev: cloneArray(hPrev), pre, h, logits, probs, value };
    return this.last;
  }

  act(obs, hPrev, rng, deterministic = false, capture = true) {
    const f = this.forward(obs, hPrev, capture);
    let action = 0;
    if (deterministic) {
      for (let i = 1; i < f.probs.length; i++) if (f.probs[i] > f.probs[action]) action = i;
    } else {
      const r = rng.next(); let c = 0;
      for (let i = 0; i < f.probs.length; i++) { c += f.probs[i]; if (r <= c) { action = i; break; } }
    }
    return {
      action,
      logProb: Math.log(Math.max(1e-12, f.probs[action])),
      value: f.value,
      hidden: capture ? cloneArray(f.h) : f.h,
      snapshot: capture ? f : null,
    };
  }

  paramCount() {
    return Object.values(this.params).reduce((s, a) => s + a.length, 0);
  }

  serialize() {
    const params = {};
    for (const [k, v] of Object.entries(this.params)) params[k] = Array.from(v);
    return { version: 1, architecture: { obsSize, hiddenSize, actionSize }, params };
  }

  restore(data) {
    if (!data?.params) throw new Error('Invalid model data');
    for (const k of Object.keys(this.params)) {
      const src = data.params[k];
      if (!Array.isArray(src) || src.length !== this.params[k].length) throw new Error(`Parameter shape mismatch: ${k}`);
      this.params[k].set(src);
    }
  }

  clone() {
    const m = new RecurrentActorCritic(1);
    m.restore(this.serialize());
    return m;
  }
}

export function modelDimensions() { return { obsSize, hiddenSize, actionSize }; }
