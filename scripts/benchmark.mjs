import { TrainingSession } from '../src/ai/session.js';
import { evaluateModel } from '../src/evaluation/evaluator.js';
import { CURRICULUM } from '../src/sim/curriculum.js';

const session = new TrainingSession({ seed: 424242, envCount: 8, autoCurriculum: false });
const stage = CURRICULUM[0];
const initial = evaluateModel(session.model, stage, { episodes: 24, seedBase: 'heldout:v1', deterministic: false });
console.log('initial', {return: initial.meanReturn.toFixed(3), food: initial.meanFood.toFixed(3), steps: initial.meanSteps.toFixed(1)});
const target = Number(process.env.STEPS || 60000);
let nextPrint = 10000;
while (session.totalSteps < target) {
  const { metric } = session.trainRollout(32);
  if (session.totalSteps >= nextPrint) {
    console.log('train', session.totalSteps, {return: metric.meanReturn.toFixed(3), food: metric.meanFood.toFixed(3), entropy: metric.entropy.toFixed(3)});
    nextPrint += 10000;
  }
}
const final = evaluateModel(session.model, stage, { episodes: 24, seedBase: 'heldout:v1', deterministic: false });
console.log('final', {return: final.meanReturn.toFixed(3), food: final.meanFood.toFixed(3), steps: final.meanSteps.toFixed(1)});
console.log(JSON.stringify({initial, final, steps: session.totalSteps}, null, 2));
