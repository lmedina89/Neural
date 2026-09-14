import { TrainingSession } from '../src/ai/session.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { evaluateFullRetentionSuite } from '../src/evaluation/evaluator.js';

// Release benchmark deliberately starts the Learner at the hardest curriculum stage
// while continual rehearsal keeps all earlier skills in the experience stream.
const session = new TrainingSession({ seed: 424242, envCount: 8, autoCurriculum: false });
session.curriculum.stage = 3;
for (let i = 0; i < session.envs.length; i++) session.resetEnv(i);

const evalOptions = { episodesPerStage: 8, seedBase: 'benchmark:all-skills:v3', deterministic: false, protocolTag: 'release-benchmark-v3' };
const initial = evaluateFullRetentionSuite(session.model, evalOptions);
console.log('initial', summary(initial));
console.log('rehearsal target', session.trainingMix().map(x => Number(x.toFixed(3))));

const target = Number(process.env.STEPS || 80000);
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
      rehearsal: metric.rehearsalMix.fractions.map(x => Number(x.toFixed(2))),
    });
    nextPrint += 10000;
  }
  if (validation) console.log('validation', {
    steps: validation.steps,
    learner: validation.validation.score.toFixed(3),
    interpretation: validation.interpretation,
    alerts: validation.forgetting.map(x => `${x.name}:${x.severity}:x${x.streak}`),
    promotions: validation.archiveUpdates,
    pending: validation.promotionPending.map(x => `${x.category}:${x.streak}/${x.required}`),
    champion: Number(validation.bestScore ?? 0).toFixed(3),
    autoRollback: validation.autoRollback,
  });
}

const learner = evaluateFullRetentionSuite(session.model, evalOptions);
let champion = null;
if (session.bestBrain?.model) {
  const model = new RecurrentActorCritic(1);
  model.restore(session.bestBrain.model);
  champion = evaluateFullRetentionSuite(model, evalOptions);
}

console.log('learner', summary(learner));
console.log('champion balanced', { checkpointSteps: session.bestBrain?.savedAtSteps ?? null, ...summary(champion) });

const report = {
  trainingSteps: session.totalSteps,
  initial: summaryNumbers(initial),
  learner: summaryNumbers(learner),
  championBalanced: champion ? { steps: session.bestBrain.savedAtSteps, ...summaryNumbers(champion) } : null,
  rehearsalTarget: session.trainingMix(),
  rehearsalObserved: session.recentRehearsalMix(),
  lineage: session.learnerLineage,
  archive: session.archiveSummary(),
  validations: session.validationHistory.map(v => ({
    steps: v.steps,
    learner: v.validation.score,
    interpretation: v.interpretation,
    alerts: v.forgetting.map(x => ({ stage: x.stage, severity: x.severity, streak: x.streak, confirmed: x.confirmed })),
    promotions: v.archiveUpdates,
    pending: v.promotionPending,
    autoRollback: v.autoRollback,
    championScore: v.bestScore,
  })),
};
console.log('summary-json', JSON.stringify(report));

if (!champion || champion.balancedScore <= initial.balancedScore) {
  console.error('Benchmark failed: repeat-confirmed Champion did not improve all-skills benchmark score over the initial policy.');
  process.exitCode = 1;
}
if (session.rollbackHistory.some(x => x?.automatic)) {
  console.error('Benchmark failed: behavioral automatic rollback occurred in autonomous-learning mode.');
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
