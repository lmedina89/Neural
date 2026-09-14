import { CONFIG } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { World } from '../sim/world.js';

export function evaluateModel(model, stage, { episodes = CONFIG.runtime.evalEpisodes, seedBase = 'heldout:v1', deterministic = false } = {}) {
  const rows = [];
  for (let i = 0; i < episodes; i++) {
    const seed = domainSeed(seedBase, i);
    const env = new World(seed, stage);
    let obs = env.observe();
    let h = model.zeroHidden();
    const rng = new PRNG(seed ^ 0x51ed270b);
    while (!env.done) {
      const a = model.act(obs, h, rng, deterministic);
      const r = env.step(a.action);
      obs = r.obs;
      h = a.hidden;
    }
    rows.push(env.info());
  }
  const mean = k => rows.reduce((s, x) => s + (x[k] || 0), 0) / rows.length;
  return {
    episodes,
    meanReturn: mean('totalReward'),
    meanFood: mean('food'),
    meanEnergy: mean('energy'),
    meanSteps: mean('steps'),
    hazardHits: mean('hazardHits'),
    wallHits: mean('wallHits'),
    survivalRate: rows.filter(x => x.survived).length / rows.length,
    seeds: rows.map(x => x.seed),
    rows,
  };
}
