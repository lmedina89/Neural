import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { CONFIG, VERSION, BUILD_MARKER } from '../src/config.js';
import { PRNG, domainSeed } from '../src/utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../src/sim/curriculum.js';
import { World } from '../src/sim/world.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { computeGAE } from '../src/ai/rollout.js';
import { TrainingSession, nextHistoricalMilestoneAfter } from '../src/ai/session.js';
import { evaluateCurriculumSuite } from '../src/evaluation/evaluator.js';

test('release identity is v0.1.0.1.2 recovery-safe save hotfix',()=>{assert.equal(VERSION,'0.1.0.1.2');assert.equal(BUILD_MARKER,'SAVEREC-01012')});
test('PRNG is repeatable',()=>{const a=new PRNG(123),b=new PRNG(123);for(let i=0;i<100;i++)assert.equal(a.nextUint(),b.nextUint())});
test('seed domains are separated',()=>{assert.notEqual(domainSeed('train',0),domainSeed('heldout:v1',0));assert.notEqual(domainSeed('validation:v1',0),domainSeed('heldout:v1',0));assert.notEqual(domainSeed('validation:v1',0),domainSeed('train',0))});
test('world generation is deterministic',()=>{const a=new World(77,CURRICULUM[2]),b=new World(77,CURRICULUM[2]);assert.deepEqual(a.food,b.food);assert.deepEqual(a.hazards,b.hazards);assert.deepEqual(a.walls,b.walls);assert.deepEqual(Array.from(a.observe()),Array.from(b.observe()))});
test('world info carries curriculum identity',()=>{const w=new World(7,CURRICULUM[2]);assert.equal(w.info().stageId,2);assert.equal(w.info().stageName,'Obstacle Avoidance')});
test('observations are finite and correct shape',()=>{const w=new World(5,CURRICULUM[0]);const o=w.observe();assert.equal(o.length,10);for(const x of o)assert.ok(Number.isFinite(x))});
test('all actions produce finite rewards',()=>{for(let a=0;a<7;a++){const w=new World(10+a,CURRICULUM[1]);const r=w.step(a);assert.ok(Number.isFinite(r.reward));assert.equal(r.obs.length,10)}});
test('GAE returns finite aligned arrays',()=>{const t=[{env:0,reward:1,value:.2,done:false},{env:0,reward:.5,value:.3,done:true}];const g=computeGAE(t,new Map([[0,0]]));assert.equal(g.advantages.length,2);assert.ok([...g.advantages,...g.returns].every(Number.isFinite))});
test('model checkpoint roundtrip is exact',()=>{const m=new RecurrentActorCritic(9),n=new RecurrentActorCritic(10);n.restore(m.serialize());for(const k of Object.keys(m.params))assert.deepEqual(Array.from(m.params[k]),Array.from(n.params[k]))});
test('training changes parameters without non-finite values',()=>{const s=new TrainingSession({seed:99,envCount:2,autoCurriculum:false});const before=Array.from(s.model.params.wp);s.trainRollout(8);const after=Array.from(s.model.params.wp);assert.notDeepEqual(after,before);for(const arr of Object.values(s.model.params))for(const x of arr)assert.ok(Number.isFinite(x))});
test('curriculum promotion uses sustained rolling performance',()=>{const c=new CurriculumManager(0);let event=null;for(let i=1;i<=CONFIG.curriculum.minSamples;i++)event=c.noteEpisode({food:3,survived:true},i)||event;assert.ok(event);assert.equal(event.reason,'promotion');assert.equal(c.stage,1);assert.equal(c.cooldownRemaining,CONFIG.curriculum.transitionCooldownEpisodes)});
test('curriculum can demote after sustained collapse',()=>{const c=new CurriculumManager(2);c.cooldownRemaining=0;let event=null;for(let i=1;i<=CONFIG.curriculum.minSamples;i++)event=c.noteEpisode({food:0,survived:false},i)||event;assert.ok(event);assert.equal(event.reason,'demotion');assert.equal(c.stage,1)});
test('validation suite is deterministic for an unchanged model and fixed protocol',()=>{const m=new RecurrentActorCritic(42);const a=evaluateCurriculumSuite(m,1,{episodesPerStage:3,seedBase:'validation:test'});const b=evaluateCurriculumSuite(m,1,{episodesPerStage:3,seedBase:'validation:test'});assert.equal(a.protocol,b.protocol);assert.equal(a.score,b.score);assert.deepEqual(a.stageResults,b.stageResults)});
test('automatic best protection establishes and preserves a fixed-protocol best',()=>{const s=new TrainingSession({seed:12,envCount:2,autoCurriculum:false});const first=s.runValidation();assert.ok(first.improved);assert.ok(s.bestBrain);const bestSteps=s.bestBrain.savedAtSteps;const bestScore=s.bestBrain.validation.score;const second=s.runValidation();assert.equal(second.improved,false);assert.equal(s.bestBrain.savedAtSteps,bestSteps);assert.equal(s.bestBrain.validation.score,bestScore)});
test('regression detector flags a sufficiently worse validation and preserves best',()=>{const s=new TrainingSession({seed:13,envCount:2,autoCurriculum:false});s.runValidation();const protocol=s.bestBrain.validation.protocol;const protectedModel=JSON.stringify(s.bestBrain.model);s.bestBrains[protocol].validation.score += CONFIG.validation.regressionTolerance + 1;const r=s.runValidation();assert.equal(r.regression,true);assert.equal(JSON.stringify(s.bestBrain.model),protectedModel)});
test('restore best rolls model parameters back without rewinding experience counter',()=>{const s=new TrainingSession({seed:14,envCount:2,autoCurriculum:false});s.runValidation();const protectedBp=Array.from(s.bestBrain.model.params.bp);s.totalSteps=12345;s.model.params.bp[0]+=99;s.restoreBest();assert.deepEqual(Array.from(s.model.params.bp),protectedBp);assert.equal(s.totalSteps,12345);assert.equal(s.rollbackHistory.at(-1).sourceSteps,s.bestBrain.savedAtSteps)});
test('schema-2 checkpoint restore preserves protected best state',()=>{const a=new TrainingSession({seed:3,envCount:2,autoCurriculum:false});a.trainRollout(4);a.runValidation();const snap=a.snapshot();const b=new TrainingSession({seed:4,envCount:2,autoCurriculum:false});b.restore(snap);assert.equal(b.totalSteps,a.totalSteps);assert.deepEqual(Array.from(b.model.params.bp),Array.from(a.model.params.bp));assert.equal(b.bestBrain.validation.protocol,a.bestBrain.validation.protocol);assert.equal(b.bestBrain.savedAtSteps,a.bestBrain.savedAtSteps)});
test('v0.1.0 schema-1 checkpoint migrates without losing model and schedules validation',()=>{const a=new TrainingSession({seed:23,envCount:2,autoCurriculum:false});a.trainRollout(4);const modern=a.snapshot();const legacy={schema:1,seed:modern.seed,totalSteps:modern.totalSteps,totalEpisodes:modern.totalEpisodes,envSeedCursor:modern.envSeedCursor,curriculum:{stage:modern.curriculum.stage,history:modern.curriculum.history},model:modern.model,optimizer:modern.optimizer,metrics:modern.metrics,milestones:modern.milestones};const b=new TrainingSession({seed:99,envCount:2,autoCurriculum:false});b.restore(legacy);assert.equal(b.totalSteps,a.totalSteps);assert.deepEqual(Array.from(b.model.params.bp),Array.from(a.model.params.bp));assert.equal(b.bestBrain,null);assert.equal(b.nextValidationStep,b.totalSteps);assert.equal(b.curriculum.cooldownRemaining,CONFIG.curriculum.transitionCooldownEpisodes)});

test('migrated legacy run validates its existing curriculum before further training',()=>{const a=new TrainingSession({seed:31,envCount:2,autoCurriculum:false});const snap=a.snapshot();const legacy={schema:1,seed:snap.seed,totalSteps:602496,totalEpisodes:1034,envSeedCursor:snap.envSeedCursor,curriculum:{stage:2,history:Array(40).fill(0.1)},model:snap.model,optimizer:snap.optimizer,metrics:[],milestones:snap.milestones};const b=new TrainingSession({seed:99,envCount:2,autoCurriculum:true});b.restore(legacy);b.trainRollout(1);assert.ok(b.validationHistory.length>=1);assert.equal(b.validationHistory[0].validation.maxStage,2);assert.ok(b.bestBrains[b.validationHistory[0].validation.protocol]);});
test('reward is decomposed into finite named components',()=>{const w=new World(12345,CURRICULUM[1]);const r=w.step(1);const parts=r.info.rewardParts;for(const k of ['survival','energy','wall','food','hazard','approach','death'])assert.ok(Number.isFinite(parts[k]),k)});
test('mobile help and best-brain controls are present in static UI',async()=>{const html=await readFile(new URL('../index.html',import.meta.url),'utf8');for(const id of ['brainSource','bestOption','restoreBestBtn','latestValidation','bestValidation','manualSlotInfo','autosaveSlotInfo','loadManualBtn','loadAutosaveBtn'])assert.match(html,new RegExp(`id="${id}"`));assert.match(html,/What do the controls and numbers mean\?/)});

test('historical milestone schedule continues beyond one million steps',()=>{
  assert.equal(nextHistoricalMilestoneAfter(1_000_000),1_250_000);
  assert.equal(nextHistoricalMilestoneAfter(2_122_848),2_250_000);
  assert.equal(nextHistoricalMilestoneAfter(4_999_999),5_000_000);
  assert.equal(nextHistoricalMilestoneAfter(5_000_000),5_500_000);
  assert.equal(nextHistoricalMilestoneAfter(20_000_000),21_000_000);
  assert.equal(nextHistoricalMilestoneAfter(100_000_000),105_000_000);
});

test('legacy long-run migration snapshots the exact loaded brain and does not backfill fake milestones',()=>{
  const a=new TrainingSession({seed:44,envCount:2,autoCurriculum:false});
  const snap=a.snapshot();
  const steps=2_122_848;
  const legacy={schema:1,seed:snap.seed,totalSteps:steps,totalEpisodes:3000,envSeedCursor:snap.envSeedCursor,curriculum:{stage:2,history:[]},model:snap.model,optimizer:snap.optimizer,metrics:[],milestones:snap.milestones};
  const b=new TrainingSession({seed:45,envCount:2,autoCurriculum:false});
  b.restore(legacy);
  assert.equal(b.nextMilestoneStep,2_250_000);
  assert.ok(b.milestones.has(steps));
  assert.equal(b.milestones.has(1_250_000),false);
  assert.equal(b.milestones.has(1_500_000),false);
});


test('recovery-safe UI exposes explicit manual and autosave loading instead of newest-wins loading',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(html,/Load Manual/);
  assert.match(html,/Load Autosave/);
  assert.doesNotMatch(main,/loadNewestCheckpoint/);
  assert.match(main,/loadCheckpointRecord\('latest'\)/);
  assert.match(main,/loadCheckpointRecord\('autosave'\)/);
});

test('manual save recovery guard refuses lower-step overwrite',async()=>{
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(main,/storedSteps>session\.totalSteps/);
  assert.match(main,/this lower-step brain was NOT allowed to overwrite it/);
});

test('explicit checkpoint restore pauses training for verification',async()=>{
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(main,/paused=true;el\.pause\.textContent='Resume'/);
  assert.match(main,/Training is PAUSED so you can verify it/);
});
