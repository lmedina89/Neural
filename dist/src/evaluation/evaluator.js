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

export function evaluateCurriculumSuite(model, maxStageIndex = CURRICULUM.length - 1, {
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
    const skillScore = skillScoreFor(result, stage);
    stageResults.push({ stage: stageIndex, name: stage.name, skillScore, ...withoutRows(result) });
    allRows.push(...result.rows);
  }
  const summary = summarizeRows(allRows);
  const balancedScore = stageResults.length ? stageResults.reduce((s, x) => s + x.skillScore, 0) / stageResults.length : 0;
  const categoryScores = {
    overall: summary.meanReturn,
    forager: summary.meanFood,
    survivor: summary.survivalRate,
    efficiency: summary.meanEnergy,
    balanced: balancedScore,
  };
  return {
    ...withoutRows(summary),
    score: balancedScore,
    balancedScore,
    categoryScores,
    maxStage,
    protocol: `${seedBase}|0-${maxStage}|${episodesPerStage}|${deterministic ? 'det' : 'seeded-stochastic'}|retention-v2`,
    stageResults,
  };
}

export function evaluateFullRetentionSuite(model, options = {}) {
  return evaluateCurriculumSuite(model, CURRICULUM.length - 1, options);
}

function skillScoreFor(result, stage) {
  // Bounded competence score: resource acquisition is primary, but survival and
  // remaining energy matter so "do nothing" and reckless foraging cannot dominate.
  const foodScale = Math.max(0.75, Number(stage.foods) || 1);
  const foodScore = 1 - Math.exp(-Math.max(0, result.meanFood) / foodScale);
  const survival = clamp01(result.survivalRate);
  const energy = clamp01(result.meanEnergy);
  return clamp01(0.58 * foodScore + 0.30 * survival + 0.12 * energy);
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

function clamp01(x) { return Math.max(0, Math.min(1, Number(x) || 0)); }
