export function hashString(input) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
}

export class PRNG {
  constructor(seed = 1) {
    this.state = (seed >>> 0) || 0x6d2b79f5;
  }
  nextUint() {
    let t = (this.state += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }
  next() { return this.nextUint() / 4294967296; }
  range(min, max) { return min + (max - min) * this.next(); }
  int(min, maxExclusive) { return min + Math.floor(this.next() * (maxExclusive - min)); }
  pick(arr) { return arr[this.int(0, arr.length)]; }
  normal() {
    const u = Math.max(1e-12, this.next());
    const v = Math.max(1e-12, this.next());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  clone() { const p = new PRNG(1); p.state = this.state; return p; }
}

export function domainSeed(domain, index) {
  return hashString(`micromind:${domain}:${index}`);
}
