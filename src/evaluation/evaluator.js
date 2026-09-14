import { CONFIG } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM } from '../sim/curriculum.js';
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
  return summarizeRows(rows);
}

export function evaluateCurriculumSuite(model, maxStageIndex, {
  episodesPerStage = CONFIG.validation.episodesPerStage,
  seedBase = CONFIG.validation.seedBase,
  deterministic = false,
} = {}) {
  const maxStage = Math.max(0, Math.min(CURRICULUM.length - 1, Number(maxStageIndex) || 0));
  const stageResults = [];
  const allRows = [];
  for (let stageIndex = 0; stageIndex <= maxStage; stageIndex++) {
    const stage = CURRICULUM[stageIndex];
    const result = evaluateModel(model, stage, {
      episodes: episodesPerStage,
      seedBase: `${seedBase}:stage:${stageIndex}`,
      deterministic,
    });
    stageResults.push({ stage: stageIndex, name: stage.name, ...withoutRows(result) });
    allRows.push(...result.rows);
  }
  const summary = summarizeRows(allRows);
  return {
    ...withoutRows(summary),
    score: summary.meanReturn,
    maxStage,
    protocol: `${seedBase}|0-${maxStage}|${episodesPerStage}|${deterministic ? 'det' : 'seeded-stochastic'}`,
    stageResults,
  };
}

function summarizeRows(rows) {
  if (!rows.length) return { episodes: 0, meanReturn: 0, meanFood: 0, meanEnergy: 0, meanSteps: 0, hazardHits: 0, wallHits: 0, survivalRate: 0, seeds: [], rows: [] };
  const mean = k => rows.reduce((s, x) => s + (Number(x[k]) || 0), 0) / rows.length;
  return {
    episodes: rows.length,
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

function withoutRows(result) {
  const { rows, ...rest } = result;
  return rest;
}
