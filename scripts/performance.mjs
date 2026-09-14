import { TrainingSession } from '../src/ai/session.js';

const seed = 539;
const envCount = 8;
const rollout = 36;
const warmup = 10;
const measured = 120;
const session = new TrainingSession({ seed, envCount, autoCurriculum: false });
session.curriculum.stage = 3;
session.nextValidationStep = Number.MAX_SAFE_INTEGER;
for (let i = 0; i < session.envs.length; i++) session.resetEnv(i);
for (let i = 0; i < warmup; i++) session.trainRollout(rollout);
let transitions = 0;
const t0 = performance.now();
for (let i = 0; i < measured; i++) transitions += session.trainRollout(rollout).transitions;
const ms = performance.now() - t0;
const last = session.metrics.at(-1)?.profile || {};
console.log(JSON.stringify({
  version: '0.1.2.1',
  seed,
  envCount,
  curriculum: 'Scarcity',
  rollout,
  transitions,
  wallClockStepsPerSec: transitions / (ms / 1000),
  lastProfile: last,
}, null, 2));
