import { TrainingSession } from '../src/ai/session.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { evaluateFullRetentionSuite } from '../src/evaluation/evaluator.js';

const session = new TrainingSession({ seed: 424242, envCount: 8, autoCurriculum: false });
const evalOptions = { episodesPerStage: 8, seedBase: 'benchmark:all-skills:v2', deterministic: false, protocolTag: 'release-benchmark-v2' };
const initial = evaluateFullRetentionSuite(session.model, evalOptions);
console.log('initial', summary(initial));

const target = Number(process.env.STEPS || 60000);
let nextPrint = 10000;
while (session.totalSteps < target) {
  const { metric, validation } = session.trainRollout(32);
  if (session.totalSteps >= nextPrint) {
    console.log('train', session.totalSteps, {
      return: metric.meanReturn.toFixed(3),
      food: metric.meanFood.toFixed(3),
      entropy: metric.entropy.toFixed(3),
      lr: metric.learningRate.toExponential(2),
      kl: metric.maxEpochKL.toFixed(4),
      epochs: metric.epochsRun,
      rejected: metric.updateRejected,
    });
    nextPrint += 10000;
  }
  if (validation) console.log('validation', {
    steps: validation.steps,
    balanced: validation.validation.score.toFixed(3),
    interpretation: validation.interpretation,
    balancedEvidence: validation.balancedEvidence,
    balancedConfirmed: validation.balancedConfirmed,
    alerts: validation.forgetting.map(x => `${x.name}:${x.severity}:x${x.streak}`),
    autoRollback: validation.autoRollback?.sourceSteps ?? null,
    best: Number(validation.bestScore ?? 0).toFixed(3),
  });
}

const latest = evaluateFullRetentionSuite(session.model, evalOptions);
let protectedBest = null;
if (session.bestBrain?.model) {
  const model = new RecurrentActorCritic(1);
  model.restore(session.bestBrain.model);
  protectedBest = evaluateFullRetentionSuite(model, evalOptions);
}

console.log('latest', summary(latest));
console.log('protected balanced', { checkpointSteps: session.bestBrain?.savedAtSteps ?? null, ...summary(protectedBest) });

const report = {
  trainingSteps: session.totalSteps,
  initial: summaryNumbers(initial),
  latest: summaryNumbers(latest),
  protectedBalanced: protectedBest ? { steps: session.bestBrain.savedAtSteps, ...summaryNumbers(protectedBest) } : null,
  rollbacks: session.rollbackHistory,
  archive: session.archiveSummary(),
  validations: session.validationHistory.map(v => ({
    steps: v.steps,
    balanced: v.validation.score,
    interpretation: v.interpretation,
    balancedEvidence: v.balancedEvidence,
    balancedConfirmed: v.balancedConfirmed,
    alerts: v.forgetting.map(x => ({ stage: x.stage, severity: x.severity, streak: x.streak, confirmed: x.confirmed })),
    autoRollback: v.autoRollback,
    bestScore: v.bestScore,
  })),
};
console.log('summary-json', JSON.stringify(report));

if (!protectedBest || protectedBest.balancedScore <= initial.balancedScore) {
  console.error('Benchmark failed: protected balanced brain did not improve all-skills benchmark score over the initial policy.');
  process.exitCode = 1;
}
if (latest.balancedScore <= initial.balancedScore) {
  console.error('Benchmark failed: latest policy did not improve all-skills benchmark score over the initial policy.');
  process.exitCode = 1;
}

function summary(r) {
  if (!r) return { gen: '—', return: '—', food: '—', survival: '—' };
  return {
    gen: `${(r.balancedScore * 100).toFixed(1)}% [${(r.balancedCiLow * 100).toFixed(1)}–${(r.balancedCiHigh * 100).toFixed(1)}]`,
    return: r.meanReturn.toFixed(3),
    food: r.meanFood.toFixed(3),
    survival: `${(r.survivalRate * 100).toFixed(1)}%`,
    steps: r.meanSteps.toFixed(1),
  };
}
function summaryNumbers(r) {
  return { balancedScore: r.balancedScore, meanReturn: r.meanReturn, meanFood: r.meanFood, survivalRate: r.survivalRate, meanSteps: r.meanSteps };
}
