import { CONFIG } from '../config.js';
import { PRNG } from '../utils/prng.js';

function zerosLike(params) {
  const o = {};
  for (const [k, v] of Object.entries(params)) o[k] = new Float64Array(v.length);
  return o;
}
function zeroGrad(g) { for (const v of Object.values(g)) v.fill(0); }
function shuffleIndices(n, rng) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.int(0, i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function cloneArrays(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[k] = new Float64Array(v);
  return out;
}
function restoreArrays(target, source) {
  for (const k of Object.keys(target)) if (source[k]?.length === target[k].length) target[k].set(source[k]);
}
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

export class PPOTrainer {
  constructor(model, seed = 9876) {
    this.model = model;
    this.rng = new PRNG(seed);
    this.m = zerosLike(model.params);
    this.v = zerosLike(model.params);
    this.t = 0;
    this.learningRate = CONFIG.ppo.learningRate;
    this.lastStats = {};
    this.rejectedUpdates = 0;
    this.lrCooldownUpdates = 0;
  }

  update(transitions, advantages, returns, { trainingStep = 0 } = {}) {
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

    const baseEntropyCoef = entropyCoefficient(trainingStep);
    const priorEntropy = Number(this.lastStats.entropy);
    const rescueMultiplier = Number.isFinite(priorEntropy) && priorEntropy < cfg.entropyRescueCritical
      ? cfg.entropyRescueCriticalMultiplier
      : Number.isFinite(priorEntropy) && priorEntropy < cfg.entropyRescueLow
        ? cfg.entropyRescueLowMultiplier
        : 1;
    const entropyCoef = Math.min(cfg.maxEntropyCoef, baseEntropyCoef * rescueMultiplier);
    const modelBefore = cloneArrays(this.model.params);
    const mBefore = cloneArrays(this.m);
    const vBefore = cloneArrays(this.v);
    const tBefore = this.t;
    const lrBefore = this.learningRate;

    const g = zerosLike(this.model.params);
    const dLogits = new Float64Array(CONFIG.model.actionSize);
    const dh = new Float64Array(CONFIG.model.hiddenSize);
    let accPolicy = 0, accValue = 0, accEntropy = 0, accKL = 0, clippedCount = 0, sampleCount = 0;
    let epochsRun = 0;
    let earlyStopped = false;
    let updateRejected = false;
    let maxEpochKL = 0;

    for (let epoch = 0; epoch < cfg.epochs; epoch++) {
      const idx = shuffleIndices(transitions.length, this.rng);
      let epochKL = 0;
      let epochSamples = 0;
      for (let start = 0; start < idx.length; start += cfg.minibatchSize) {
        zeroGrad(g);
        const end = Math.min(idx.length, start + cfg.minibatchSize);
        let batchCount = 0;
        for (let pos = start; pos < end; pos++) {
          const n = idx[pos];
          const tr = transitions[n];
          const f = this.model.forward(tr.obs, tr.hPrev, false);
          const pAct = Math.max(1e-12, f.probs[tr.action]);
          const logp = Math.log(pAct);
          const logRatio = logp - tr.logProb;
          const ratio = Math.exp(logRatio);
          const approxKL = Math.max(0, (ratio - 1) - logRatio);
          const a = normAdv[n];
          const unclipped = ratio * a;
          const clippedRatio = clamp(ratio, 1 - cfg.clip, 1 + cfg.clip);
          const clippedObj = clippedRatio * a;
          const policyLoss = -Math.min(unclipped, clippedObj);
          const isClipped = (a >= 0 && ratio > 1 + cfg.clip) || (a < 0 && ratio < 1 - cfg.clip);
          if (isClipped) clippedCount++;

          let entropy = 0;
          for (const pr of f.probs) entropy -= pr * Math.log(Math.max(1e-12, pr));
          const valueErr = f.value - returns[n];
          const valueLoss = 0.5 * valueErr * valueErr;
          const dLogP = isClipped ? 0 : -ratio * a;
          dLogits.fill(0);
          for (let k = 0; k < dLogits.length; k++) {
            dLogits[k] = dLogP * ((k === tr.action ? 1 : 0) - f.probs[k]);
            dLogits[k] += entropyCoef * f.probs[k] * (Math.log(Math.max(1e-12, f.probs[k])) + entropy);
          }
          const dValue = cfg.valueCoef * valueErr;

          const p = this.model.params;
          const h = f.h;
          dh.fill(0);
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
          accKL += approxKL;
          epochKL += approxKL;
          sampleCount++;
          epochSamples++;
          batchCount++;
        }
        this.applyAdam(g, 1 / Math.max(1, batchCount));
      }

      epochsRun++;
      const meanEpochKL = epochKL / Math.max(1, epochSamples);
      maxEpochKL = Math.max(maxEpochKL, meanEpochKL);
      if (meanEpochKL > cfg.hardKL) {
        restoreArrays(this.model.params, modelBefore);
        restoreArrays(this.m, mBefore);
        restoreArrays(this.v, vBefore);
        this.t = tBefore;
        this.learningRate = Math.max(cfg.minLearningRate, lrBefore * 0.5);
        this.rejectedUpdates++;
        updateRejected = true;
        earlyStopped = true;
        break;
      }
      if (meanEpochKL > cfg.targetKL) {
        earlyStopped = true;
        break;
      }
    }

    if (!updateRejected) {
      if (maxEpochKL > cfg.targetKL * 1.25) {
        this.learningRate = Math.max(cfg.minLearningRate, this.learningRate * cfg.lrDecrease);
      } else if (this.lrCooldownUpdates <= 0 && maxEpochKL < cfg.targetKL * 0.35) {
        this.learningRate = Math.min(cfg.maxLearningRate, this.learningRate * cfg.lrIncrease);
      }
    }
    if (this.lrCooldownUpdates > 0) this.lrCooldownUpdates--;

    const denom = Math.max(1, sampleCount);
    this.lastStats = {
      policyLoss: accPolicy / denom,
      valueLoss: accValue / denom,
      entropy: accEntropy / denom,
      entropyCoef,
      entropyRescueMultiplier: rescueMultiplier,
      lrCooldownUpdates: this.lrCooldownUpdates,
      approxKL: accKL / denom,
      maxEpochKL,
      clipFraction: clippedCount / denom,
      advantageMean: mean,
      advantageStd: sd,
      learningRate: this.learningRate,
      epochsRun,
      earlyStopped,
      updateRejected,
      rejectedUpdates: this.rejectedUpdates,
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
    const beta1Correction = 1 - Math.pow(cfg.adamBeta1, this.t);
    const beta2Correction = 1 - Math.pow(cfg.adamBeta2, this.t);
    for (const k of Object.keys(this.model.params)) {
      const p = this.model.params[k], m = this.m[k], v = this.v[k], gg = g[k];
      for (let i = 0; i < p.length; i++) {
        const grad = gg[i] * scale;
        m[i] = cfg.adamBeta1 * m[i] + (1 - cfg.adamBeta1) * grad;
        v[i] = cfg.adamBeta2 * v[i] + (1 - cfg.adamBeta2) * grad * grad;
        const mh = m[i] / beta1Correction;
        const vh = v[i] / beta2Correction;
        p[i] -= this.learningRate * mh / (Math.sqrt(vh) + cfg.adamEps);
        if (!Number.isFinite(p[i])) throw new Error(`Non-finite parameter ${k}[${i}]`);
      }
    }
  }

  enterRecoveryCooldown(updates = CONFIG.ppo.lrRecoveryCooldownUpdates) {
    this.lrCooldownUpdates = Math.max(this.lrCooldownUpdates, Math.max(0, Number(updates) || 0));
  }

  serialize() {
    const pack = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.from(v)]));
    return {
      t: this.t,
      m: pack(this.m),
      v: pack(this.v),
      learningRate: this.learningRate,
      rejectedUpdates: this.rejectedUpdates,
      lrCooldownUpdates: this.lrCooldownUpdates,
      lastStats: this.lastStats,
      rngState: this.rng.state >>> 0,
    };
  }

  restore(data) {
    if (!data) return;
    this.t = data.t || 0;
    this.lastStats = data.lastStats || {};
    this.learningRate = clamp(Number(data.learningRate) || CONFIG.ppo.learningRate, CONFIG.ppo.minLearningRate, CONFIG.ppo.maxLearningRate);
    this.rejectedUpdates = Math.max(0, Number(data.rejectedUpdates) || 0);
    this.lrCooldownUpdates = Math.max(0, Number(data.lrCooldownUpdates) || 0);
    if (Number.isFinite(Number(data.rngState))) this.rng.state = Number(data.rngState) >>> 0;
    for (const k of Object.keys(this.m)) {
      if (data.m?.[k]?.length === this.m[k].length) this.m[k].set(data.m[k]);
      if (data.v?.[k]?.length === this.v[k].length) this.v[k].set(data.v[k]);
    }
  }
}

function entropyCoefficient(trainingStep) {
  const cfg = CONFIG.ppo;
  const t = clamp((Number(trainingStep) || 0) / Math.max(1, cfg.entropyDecaySteps), 0, 1);
  return cfg.entropyStart + (cfg.entropyEnd - cfg.entropyStart) * t;
}
