import { TrainingSession } from '../src/ai/session.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { evaluateModel } from '../src/evaluation/evaluator.js';
import { CURRICULUM } from '../src/sim/curriculum.js';

const session = new TrainingSession({ seed: 424242, envCount: 8, autoCurriculum: false });
const stage = CURRICULUM[0];
const evalOptions = { episodes: 32, seedBase: 'heldout:v1', deterministic: false };
const initial = evaluateModel(session.model, stage, evalOptions);
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
    improved: validation.improved,
    regression: validation.regression,
    forgetting: validation.forgetting.map(x => x.name),
    autoRollback: validation.autoRollback?.sourceSteps ?? null,
    best: validation.bestScore.toFixed(3),
  });
}

const latest = evaluateModel(session.model, stage, evalOptions);
let protectedBest = null;
if (session.bestBrain?.model) {
  const model = new RecurrentActorCritic(1);
  model.restore(session.bestBrain.model);
  protectedBest = evaluateModel(model, stage, evalOptions);
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
    improved: v.improved,
    regression: v.regression,
    forgetting: v.forgetting.map(x => x.stage),
    autoRollback: v.autoRollback,
    bestScore: v.bestScore,
  })),
};
console.log('summary-json', JSON.stringify(report));

if (!protectedBest || protectedBest.meanReturn <= initial.meanReturn) {
  console.error('Benchmark failed: protected balanced brain did not improve held-out return over the initial policy.');
  process.exitCode = 1;
}
if (latest.meanReturn <= initial.meanReturn) {
  console.error('Benchmark failed: latest policy did not improve over initial held-out return after stability recovery.');
  process.exitCode = 1;
}

function summary(r) {
  if (!r) return { return: '—', food: '—', survival: '—' };
  return { return: r.meanReturn.toFixed(3), food: r.meanFood.toFixed(3), survival: `${(r.survivalRate * 100).toFixed(1)}%`, steps: r.meanSteps.toFixed(1) };
}
function summaryNumbers(r) {
  return { meanReturn: r.meanReturn, meanFood: r.meanFood, survivalRate: r.survivalRate, meanSteps: r.meanSteps };
}
