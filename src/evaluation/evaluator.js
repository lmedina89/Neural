import { CONFIG } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM } from '../sim/curriculum.js';
import { World } from '../sim/world.js';

export function evaluateModel(model, stage, { episodes = CONFIG.runtime.evalEpisodes, seedBase = 'heldout:legacy', deterministic = false } = {}) {
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
  protocolTag = 'retention-v3-ci',
} = {}) {
  const maxStage = Math.max(0, Math.min(CURRICULUM.length - 1, Number(maxStageIndex) || 0));
  const stageResults = [];
  const allRows = [];
  const allSkillSamples = [];
  for (let stageIndex = 0; stageIndex <= maxStage; stageIndex++) {
    const stage = CURRICULUM[stageIndex];
    const result = evaluateModel(model, stage, {
      episodes: episodesPerStage,
      seedBase: `${seedBase}:stage:${stageIndex}`,
      deterministic,
    });
    const skillSamples = result.rows.map(row => episodeSkillScore(row, stage));
    const skillStats = summarizeSamples(skillSamples);
    stageResults.push({
      stage: stageIndex,
      name: stage.name,
      skillScore: skillStats.mean,
      skillCiLow: skillStats.ciLow,
      skillCiHigh: skillStats.ciHigh,
      skillStdErr: skillStats.stdErr,
      ...withoutRows(result),
    });
    allRows.push(...result.rows);
    allSkillSamples.push(...skillSamples);
  }
  const summary = summarizeRows(allRows);
  const balancedStats = summarizeSamples(allSkillSamples);
  const balancedScore = balancedStats.mean;
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
    balancedCiLow: balancedStats.ciLow,
    balancedCiHigh: balancedStats.ciHigh,
    balancedStdErr: balancedStats.stdErr,
    categoryScores,
    maxStage,
    episodesPerStage,
    protocol: `${seedBase}|0-${maxStage}|${episodesPerStage}|${deterministic ? 'det' : 'seeded-stochastic'}|${protocolTag}`,
    stageResults,
  };
}

export function evaluateFullRetentionSuite(model, options = {}) {
  return evaluateCurriculumSuite(model, CURRICULUM.length - 1, options);
}

export function evaluateHeldoutGeneralizationSuite(model, {
  episodesPerStage = CONFIG.generalization.episodesPerStage,
  seedBase = CONFIG.generalization.seedBase,
  deterministic = false,
} = {}) {
  return evaluateFullRetentionSuite(model, {
    episodesPerStage,
    seedBase,
    deterministic,
    protocolTag: 'heldout-generalization-v2',
  });
}

export function generalizationDiagnostic(latestValidation, protectedValidation, latestHeldout, protectedHeldout, {
  margin = CONFIG.generalization.conflictMargin,
} = {}) {
  if (!latestHeldout) return { status: 'unavailable', conflict: false, champion: null, text: 'No held-out result.' };
  const latestHeld = Number(latestHeldout.balancedScore);
  const protectedHeld = Number(protectedHeldout?.balancedScore);
  const latestVal = Number(latestValidation?.balancedScore ?? latestValidation?.score);
  const protectedVal = Number(protectedValidation?.balancedScore ?? protectedValidation?.score);
  const hasProtectedHeld = Number.isFinite(protectedHeld);
  const hasValidationPair = Number.isFinite(latestVal) && Number.isFinite(protectedVal);

  let champion = null;
  if (hasProtectedHeld) {
    if (latestHeld > protectedHeld + margin) champion = 'latest';
    else if (protectedHeld > latestHeld + margin) champion = 'protected';
    else champion = 'tie';
  }

  const validationPref = hasValidationPair
    ? (latestVal > protectedVal + margin ? 'latest' : protectedVal > latestVal + margin ? 'protected' : 'tie')
    : null;
  const heldoutPref = hasProtectedHeld
    ? (latestHeld > protectedHeld + margin ? 'latest' : protectedHeld > latestHeld + margin ? 'protected' : 'tie')
    : null;
  const conflict = Boolean(validationPref && heldoutPref && validationPref !== 'tie' && heldoutPref !== 'tie' && validationPref !== heldoutPref);

  if (conflict) {
    return {
      status: 'validation-heldout-conflict',
      conflict: true,
      champion,
      validationPref,
      heldoutPref,
      text: `VALIDATION / HELD-OUT CONFLICT: validation prefers ${validationPref}, final holdout prefers ${heldoutPref}. Diagnostic only; no archive or rollback state changed.`,
    };
  }
  if (champion === 'latest') {
    return {
      status: 'latest-generalization-champion',
      conflict: false,
      champion,
      validationPref,
      heldoutPref,
      text: 'HELD-OUT OBSERVATION: Latest is the stronger generalizer on this final holdout. Diagnostic only; it is not added to the protected archive.',
    };
  }
  if (champion === 'protected') {
    return {
      status: 'protected-generalization-champion',
      conflict: false,
      champion,
      validationPref,
      heldoutPref,
      text: 'HELD-OUT OBSERVATION: Protected Balanced is stronger on this final holdout. Diagnostic only; no training state changed.',
    };
  }
  return {
    status: hasProtectedHeld ? 'no-material-difference' : 'latest-only',
    conflict: false,
    champion,
    validationPref,
    heldoutPref,
    text: hasProtectedHeld
      ? 'HELD-OUT OBSERVATION: no material generalization-score difference at the configured margin.'
      : 'HELD-OUT OBSERVATION: Latest evaluated; no compatible protected brain was available for comparison.',
  };
}

function episodeSkillScore(row, stage) {
  // Equal per-episode competence samples make confidence intervals meaningful.
  // Resource acquisition is primary, while survival and remaining energy prevent
  // both reckless foraging and "do nothing" survival from dominating.
  const foodScale = Math.max(0.75, Number(stage.foods) || 1);
  const foodScore = 1 - Math.exp(-Math.max(0, Number(row.food) || 0) / foodScale);
  const survival = row.survived ? 1 : 0;
  const energy = clamp01(row.energy);
  return clamp01(0.58 * foodScore + 0.30 * survival + 0.12 * energy);
}

function summarizeSamples(values) {
  const xs = values.map(Number).filter(Number.isFinite);
  if (!xs.length) return { mean: 0, stdErr: 0, ciLow: 0, ciHigh: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  if (xs.length === 1) return { mean, stdErr: 0, ciLow: mean, ciHigh: mean };
  const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (xs.length - 1);
  const stdErr = Math.sqrt(Math.max(0, variance) / xs.length);
  const delta = CONFIG.validation.confidenceZ * stdErr;
  return { mean, stdErr, ciLow: clamp01(mean - delta), ciHigh: clamp01(mean + delta) };
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
