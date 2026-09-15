import { CONFIG } from '../config.js';
import { PRNG } from '../utils/prng.js';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const sq = x => x * x;

function zeros(n) { return new Float64Array(n); }
function randnArray(n, rng, scale) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.normal() * scale;
  return a;
}
function cloneArrays(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[k] = new Float64Array(v);
  return out;
}

/**
 * Tiny learned forward dynamics model used only to create an intrinsic novelty
 * signal during training. It predicts the next dynamic observation from the
 * current observation + chosen action. The policy network remains unchanged.
 */
export class CuriosityModule {
  constructor(seed = 0x51c0ffee) {
    const cfg = CONFIG.curiosity;
    this.obsSize = CONFIG.model.obsSize;
    this.actionSize = CONFIG.model.actionSize;
    this.outputSize = cfg.predictFeatures;
    this.hiddenSize = cfg.hiddenSize;
    this.inputSize = this.obsSize + this.actionSize;
    const rng = new PRNG(seed >>> 0);
    const s1 = Math.sqrt(2 / Math.max(1, this.inputSize + this.hiddenSize));
    const s2 = Math.sqrt(2 / Math.max(1, this.hiddenSize + this.outputSize));
    this.params = {
      w1: randnArray(this.hiddenSize * this.inputSize, rng, s1),
      b1: zeros(this.hiddenSize),
      w2: randnArray(this.outputSize * this.hiddenSize, rng, s2),
      b2: zeros(this.outputSize),
    };
    this.m = Object.fromEntries(Object.entries(this.params).map(([k, v]) => [k, zeros(v.length)]));
    this.v = Object.fromEntries(Object.entries(this.params).map(([k, v]) => [k, zeros(v.length)]));
    this.t = 0;
    this.errorMean = cfg.initialErrorMean;
    this.errorVariance = cfg.initialErrorVariance;
    this.errorSamples = 0;
    this.lastStats = defaultStats();
    this._input = zeros(this.inputSize);
    this._hidden = zeros(this.hiddenSize);
    this._pred = zeros(this.outputSize);
  }

  paramCount() { return Object.values(this.params).reduce((n, a) => n + a.length, 0); }

  fillInput(obs, action) {
    const x = this._input;
    x.fill(0);
    for (let i = 0; i < this.obsSize; i++) x[i] = Number(obs[i]) || 0;
    const a = Math.max(0, Math.min(this.actionSize - 1, Number(action) | 0));
    x[this.obsSize + a] = 1;
    return x;
  }

  forward(obs, action, capture = false) {
    const x = this.fillInput(obs, action);
    const h = this._hidden;
    const pred = this._pred;
    const p = this.params;
    for (let i = 0; i < this.hiddenSize; i++) {
      let z = p.b1[i];
      const off = i * this.inputSize;
      for (let j = 0; j < this.inputSize; j++) z += p.w1[off + j] * x[j];
      h[i] = Math.tanh(z);
    }
    for (let k = 0; k < this.outputSize; k++) {
      let z = p.b2[k];
      const off = k * this.hiddenSize;
      for (let i = 0; i < this.hiddenSize; i++) z += p.w2[off + i] * h[i];
      pred[k] = Math.tanh(z);
    }
    if (!capture) return { input: x, hidden: h, prediction: pred };
    return {
      input: new Float64Array(x),
      hidden: new Float64Array(h),
      prediction: new Float64Array(pred),
    };
  }

  predictionError(nextObs, pred = this._pred) {
    let sum = 0;
    for (let k = 0; k < this.outputSize; k++) sum += sq(pred[k] - (Number(nextObs[k]) || 0));
    return sum / Math.max(1, this.outputSize);
  }

  noveltyFromError(error) {
    const cfg = CONFIG.curiosity;
    if (this.errorSamples < cfg.normalizationWarmupSamples) {
      return clamp(error / Math.max(cfg.minErrorScale, this.errorMean * 2), 0, 1);
    }
    const sd = Math.sqrt(Math.max(cfg.minErrorVariance, this.errorVariance));
    const z = (error - this.errorMean) / Math.max(1e-8, sd);
    return clamp(cfg.noveltyBaseline + z / cfg.noveltySigmaSpan, 0, 1);
  }

  updateErrorStats(error) {
    const beta = CONFIG.curiosity.errorEmaBeta;
    const priorMean = this.errorMean;
    this.errorMean = beta * this.errorMean + (1 - beta) * error;
    const delta = error - priorMean;
    this.errorVariance = beta * this.errorVariance + (1 - beta) * delta * delta;
    this.errorSamples++;
  }

  scoreTransition(obs, action, nextObs, {
    remainingBudget = Infinity,
    terminal = false,
    capture = false,
  } = {}) {
    const f = this.forward(obs, action, capture);
    const error = this.predictionError(nextObs, f.prediction);
    const novelty = this.noveltyFromError(error);
    const cfg = CONFIG.curiosity;
    const unclamped = cfg.rewardScale * novelty;
    // Terminal/death novelty is learned by the predictor but never rewarded; this
    // avoids incentivizing the agent to seek surprising deaths.
    const bonus = terminal ? 0 : Math.min(cfg.maxStepBonus, Math.max(0, remainingBudget), unclamped);
    this.updateErrorStats(error);

    let mostSurprisingIndex = 0;
    let mostSurprisingError = -1;
    for (let k = 0; k < this.outputSize; k++) {
      const e = Math.abs(f.prediction[k] - (Number(nextObs[k]) || 0));
      if (e > mostSurprisingError) { mostSurprisingError = e; mostSurprisingIndex = k; }
    }
    const telemetry = {
      error,
      novelty,
      bonus,
      meanError: this.errorMean,
      errorStd: Math.sqrt(Math.max(cfg.minErrorVariance, this.errorVariance)),
      mostSurprisingIndex,
      mostSurprisingError,
    };
    if (capture) {
      telemetry.obs = new Float64Array(obs);
      telemetry.nextObs = new Float64Array(nextObs);
      telemetry.action = Number(action) | 0;
      telemetry.hidden = f.hidden;
      telemetry.prediction = f.prediction;
    }
    return telemetry;
  }

  trainBatch(transitions) {
    if (!transitions?.length) return this.lastStats;
    const cfg = CONFIG.curiosity;
    const grads = Object.fromEntries(Object.entries(this.params).map(([k, v]) => [k, zeros(v.length)]));
    const dh = zeros(this.hiddenSize);
    const batches = [];
    let lossSum = 0;
    let samples = 0;

    for (let start = 0; start < transitions.length; start += cfg.minibatchSize) {
      for (const g of Object.values(grads)) g.fill(0);
      const end = Math.min(transitions.length, start + cfg.minibatchSize);
      let batchCount = 0;
      let batchLoss = 0;
      for (let n = start; n < end; n++) {
        const tr = transitions[n];
        if (!tr?.nextObs) continue;
        const { input, hidden, prediction } = this.forward(tr.obs, tr.action, false);
        dh.fill(0);
        for (let k = 0; k < this.outputSize; k++) {
          const target = Number(tr.nextObs[k]) || 0;
          const diff = prediction[k] - target;
          batchLoss += 0.5 * diff * diff;
          // Mean squared error with tanh output.
          const dz = (diff / this.outputSize) * (1 - prediction[k] * prediction[k]);
          grads.b2[k] += dz;
          const off = k * this.hiddenSize;
          for (let i = 0; i < this.hiddenSize; i++) {
            grads.w2[off + i] += dz * hidden[i];
            dh[i] += this.params.w2[off + i] * dz;
          }
        }
        for (let i = 0; i < this.hiddenSize; i++) {
          const dz = dh[i] * (1 - hidden[i] * hidden[i]);
          grads.b1[i] += dz;
          const off = i * this.inputSize;
          for (let j = 0; j < this.inputSize; j++) grads.w1[off + j] += dz * input[j];
        }
        batchCount++;
      }
      if (batchCount) {
        this.applyAdam(grads, 1 / batchCount);
        const meanBatchLoss = batchLoss / (batchCount * this.outputSize);
        batches.push(meanBatchLoss);
        lossSum += batchLoss;
        samples += batchCount;
      }
    }

    const loss = samples ? lossSum / (samples * this.outputSize) : 0;
    this.lastStats = {
      ...this.lastStats,
      loss,
      batches: batches.length,
      samples,
      meanPredictionError: this.errorMean,
      predictionErrorStd: Math.sqrt(Math.max(cfg.minErrorVariance, this.errorVariance)),
      errorSamples: this.errorSamples,
    };
    return this.lastStats;
  }

  applyAdam(grads, scale = 1) {
    const cfg = CONFIG.curiosity;
    this.t++;
    const b1c = 1 - Math.pow(cfg.adamBeta1, this.t);
    const b2c = 1 - Math.pow(cfg.adamBeta2, this.t);
    for (const k of Object.keys(this.params)) {
      const p = this.params[k], m = this.m[k], v = this.v[k], g = grads[k];
      for (let i = 0; i < p.length; i++) {
        const grad = g[i] * scale;
        m[i] = cfg.adamBeta1 * m[i] + (1 - cfg.adamBeta1) * grad;
        v[i] = cfg.adamBeta2 * v[i] + (1 - cfg.adamBeta2) * grad * grad;
        const mh = m[i] / b1c;
        const vh = v[i] / b2c;
        p[i] -= cfg.learningRate * mh / (Math.sqrt(vh) + cfg.adamEps);
        if (!Number.isFinite(p[i])) throw new Error(`Non-finite curiosity parameter ${k}[${i}]`);
      }
    }
  }

  serialize() {
    const pack = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.from(v)]));
    return {
      version: 1,
      architecture: { inputSize: this.inputSize, hiddenSize: this.hiddenSize, outputSize: this.outputSize },
      params: pack(this.params),
      m: pack(this.m),
      v: pack(this.v),
      t: this.t,
      errorMean: this.errorMean,
      errorVariance: this.errorVariance,
      errorSamples: this.errorSamples,
      lastStats: { ...this.lastStats },
    };
  }

  restore(data) {
    if (!data?.params) return false;
    const arch = data.architecture || {};
    if (Number(arch.inputSize) !== this.inputSize || Number(arch.hiddenSize) !== this.hiddenSize || Number(arch.outputSize) !== this.outputSize) {
      throw new Error('Curiosity architecture mismatch');
    }
    for (const k of Object.keys(this.params)) {
      if (!Array.isArray(data.params[k]) || data.params[k].length !== this.params[k].length) throw new Error(`Curiosity parameter shape mismatch: ${k}`);
      this.params[k].set(data.params[k]);
      if (Array.isArray(data.m?.[k]) && data.m[k].length === this.m[k].length) this.m[k].set(data.m[k]);
      if (Array.isArray(data.v?.[k]) && data.v[k].length === this.v[k].length) this.v[k].set(data.v[k]);
    }
    this.t = Math.max(0, Number(data.t) || 0);
    this.errorMean = Math.max(CONFIG.curiosity.minErrorScale, Number(data.errorMean) || CONFIG.curiosity.initialErrorMean);
    this.errorVariance = Math.max(CONFIG.curiosity.minErrorVariance, Number(data.errorVariance) || CONFIG.curiosity.initialErrorVariance);
    this.errorSamples = Math.max(0, Number(data.errorSamples) || 0);
    this.lastStats = { ...defaultStats(), ...(data.lastStats || {}) };
    return true;
  }

  clone() {
    const c = new CuriosityModule(1);
    c.restore(this.serialize());
    return c;
  }
}

function defaultStats() {
  return {
    loss: 0,
    batches: 0,
    samples: 0,
    meanPredictionError: CONFIG.curiosity.initialErrorMean,
    predictionErrorStd: Math.sqrt(CONFIG.curiosity.initialErrorVariance),
    errorSamples: 0,
  };
}
