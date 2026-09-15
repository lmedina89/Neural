export class RollingStepRate {
  constructor({ windowsMs = [5000, 30000], sampleIntervalMs = 120 } = {}) {
    this.windowsMs = [...windowsMs].sort((a, b) => a - b);
    this.sampleIntervalMs = Math.max(16, Number(sampleIntervalMs) || 120);
    this.samples = [];
    this.lastSampleAt = -Infinity;
  }

  reset(now = 0, steps = 0) {
    const t = Number(now) || 0;
    const s = Number(steps) || 0;
    this.samples = [{ t, steps: s }];
    this.lastSampleAt = t;
  }

  sample(now, steps) {
    const t = Number(now);
    const s = Number(steps);
    if (!Number.isFinite(t) || !Number.isFinite(s)) return;
    const last = this.samples.at(-1);
    if (!last || t < last.t || s < last.steps) {
      this.reset(t, s);
      return;
    }
    if (t - this.lastSampleAt < this.sampleIntervalMs && s === last.steps) return;
    if (t - this.lastSampleAt < this.sampleIntervalMs) return;
    this.samples.push({ t, steps: s });
    this.lastSampleAt = t;
    const maxWindow = this.windowsMs.at(-1) || 30000;
    const cutoff = t - maxWindow - 1500;
    while (this.samples.length > 2 && this.samples[1].t < cutoff) this.samples.shift();
  }

  stats(windowMs) {
    if (this.samples.length < 2) return { rate: null, elapsedMs: 0, ready: false };
    const latest = this.samples.at(-1);
    const target = latest.t - windowMs;
    let base = this.samples[0];
    for (const sample of this.samples) {
      if (sample.t <= target) base = sample;
      else break;
    }
    const elapsedMs = latest.t - base.t;
    if (elapsedMs <= 0) return { rate: null, elapsedMs: 0, ready: false };
    const deltaSteps = Math.max(0, latest.steps - base.steps);
    return {
      rate: deltaSteps / (elapsedMs / 1000),
      elapsedMs,
      ready: elapsedMs >= windowMs * 0.9,
    };
  }
}

export function formatRollingRate(stat) {
  if (!stat || !Number.isFinite(stat.rate)) return '—';
  const value = `${Math.round(stat.rate).toLocaleString()}/s`;
  return stat.ready ? value : `~${value}`;
}
