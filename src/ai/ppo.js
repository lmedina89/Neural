import { CONFIG } from '../config.js';
import { PRNG } from '../utils/prng.js';

function zerosLike(params) {
  const o = {};
  for (const [k, v] of Object.entries(params)) o[k] = new Float64Array(v.length);
  return o;
}
function zeroGrad(g) { for (const v of Object.values(g)) v.fill(0); }
function gradNorm(g) {
  let s = 0;
  for (const arr of Object.values(g)) for (const x of arr) s += x * x;
  return Math.sqrt(s);
}
function shuffleIndices(n, rng) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.int(0, i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export class PPOTrainer {
  constructor(model, seed = 9876) {
    this.model = model;
    this.rng = new PRNG(seed);
    this.m = zerosLike(model.params);
    this.v = zerosLike(model.params);
    this.t = 0;
    this.lastStats = {};
  }

  update(transitions, advantages, returns) {
    if (!transitions.length) return {};
    const cfg = CONFIG.ppo;
    let mean = 0;
    for (const x of advantages) mean += x;
    mean /= advantages.length;
    let variance = 0;
    for (const x of advantages) variance += (x - mean) ** 2;
    const sd = Math.sqrt(variance / advantages.length + 1e-8);
    const normAdv = new Float64Array(advantages.length);
    for (let i = 0; i < advantages.length; i++) normAdv[i] = (advantages[i] - mean) / sd;

    const g = zerosLike(this.model.params);
    let accPolicy = 0, accValue = 0, accEntropy = 0, accKL = 0, clippedCount = 0, sampleCount = 0;

    for (let epoch = 0; epoch < cfg.epochs; epoch++) {
      const idx = shuffleIndices(transitions.length, this.rng);
      for (let start = 0; start < idx.length; start += cfg.minibatchSize) {
        zeroGrad(g);
        const end = Math.min(idx.length, start + cfg.minibatchSize);
        let batchCount = 0;
        for (let pos = start; pos < end; pos++) {
          const n = idx[pos];
          const tr = transitions[n];
          const f = this.model.forward(tr.obs, tr.hPrev);
          const pAct = Math.max(1e-12, f.probs[tr.action]);
          const logp = Math.log(pAct);
          const ratio = Math.exp(logp - tr.logProb);
          const a = normAdv[n];
          const unclipped = ratio * a;
          const clippedRatio = Math.max(1 - cfg.clip, Math.min(1 + cfg.clip, ratio));
          const clippedObj = clippedRatio * a;
          const policyLoss = -Math.min(unclipped, clippedObj);
          const isClipped = (a >= 0 && ratio > 1 + cfg.clip) || (a < 0 && ratio < 1 - cfg.clip);
          if (isClipped) clippedCount++;

          let entropy = 0;
          for (const pr of f.probs) entropy -= pr * Math.log(Math.max(1e-12, pr));
          const valueErr = f.value - returns[n];
          const valueLoss = 0.5 * valueErr * valueErr;
          const dLogP = isClipped ? 0 : -ratio * a;
          const dLogits = new Float64Array(f.probs.length);
          for (let k = 0; k < dLogits.length; k++) {
            dLogits[k] = dLogP * ((k === tr.action ? 1 : 0) - f.probs[k]);
            dLogits[k] += cfg.entropyCoef * f.probs[k] * (Math.log(Math.max(1e-12, f.probs[k])) + entropy);
          }
          const dValue = cfg.valueCoef * valueErr;

          const p = this.model.params;
          const h = f.h;
          const dh = new Float64Array(h.length);
          for (let k = 0; k < dLogits.length; k++) {
            const off = k * h.length;
            g.bp[k] += dLogits[k];
            for (let i = 0; i < h.length; i++) {
              g.wp[off + i] += dLogits[k] * h[i];
              dh[i] += p.wp[off + i] * dLogits[k];
            }
          }
          g.bv[0] += dValue;
          for (let i = 0; i < h.length; i++) {
            g.wv[i] += dValue * h[i];
            dh[i] += p.wv[i] * dValue;
          }
          for (let i = 0; i < h.length; i++) {
            const dz = dh[i] * (1 - h[i] * h[i]);
            g.bh[i] += dz;
            const xo = i * tr.obs.length, ho = i * h.length;
            for (let j = 0; j < tr.obs.length; j++) g.wx[xo + j] += dz * tr.obs[j];
            for (let j = 0; j < h.length; j++) g.wh[ho + j] += dz * tr.hPrev[j];
          }

          accPolicy += policyLoss;
          accValue += valueLoss;
          accEntropy += entropy;
          accKL += tr.logProb - logp;
          sampleCount++;
          batchCount++;
        }
        this.applyAdam(g, 1 / Math.max(1, batchCount));
      }
    }

    const denom = Math.max(1, sampleCount);
    this.lastStats = {
      policyLoss: accPolicy / denom,
      valueLoss: accValue / denom,
      entropy: accEntropy / denom,
      approxKL: accKL / denom,
      clipFraction: clippedCount / denom,
      advantageMean: mean,
      advantageStd: sd,
    };
    return this.lastStats;
  }

  applyAdam(g, gradientScale = 1) {
    const cfg = CONFIG.ppo;
    let normSq = 0;
    for (const arr of Object.values(g)) for (const x of arr) normSq += (x * gradientScale) ** 2;
    const norm = Math.sqrt(normSq);
    const clipScale = norm > cfg.maxGradNorm ? cfg.maxGradNorm / (norm + 1e-12) : 1;
    const scale = gradientScale * clipScale;
    this.t++;
    for (const k of Object.keys(this.model.params)) {
      const p = this.model.params[k], m = this.m[k], v = this.v[k], gg = g[k];
      for (let i = 0; i < p.length; i++) {
        const grad = gg[i] * scale;
        m[i] = cfg.adamBeta1 * m[i] + (1 - cfg.adamBeta1) * grad;
        v[i] = cfg.adamBeta2 * v[i] + (1 - cfg.adamBeta2) * grad * grad;
        const mh = m[i] / (1 - Math.pow(cfg.adamBeta1, this.t));
        const vh = v[i] / (1 - Math.pow(cfg.adamBeta2, this.t));
        p[i] -= cfg.learningRate * mh / (Math.sqrt(vh) + cfg.adamEps);
        if (!Number.isFinite(p[i])) throw new Error(`Non-finite parameter ${k}[${i}]`);
      }
    }
  }

  serialize() {
    const pack = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.from(v)]));
    return { t: this.t, m: pack(this.m), v: pack(this.v), lastStats: this.lastStats };
  }

  restore(data) {
    if (!data) return;
    this.t = data.t || 0;
    this.lastStats = data.lastStats || {};
    for (const k of Object.keys(this.m)) {
      if (data.m?.[k]?.length === this.m[k].length) this.m[k].set(data.m[k]);
      if (data.v?.[k]?.length === this.v[k].length) this.v[k].set(data.v[k]);
    }
  }
}
