import test from 'node:test';
import assert from 'node:assert/strict';
import { PRNG, domainSeed } from '../src/utils/prng.js';
import { CURRICULUM } from '../src/sim/curriculum.js';
import { World } from '../src/sim/world.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { computeGAE } from '../src/ai/rollout.js';
import { TrainingSession } from '../src/ai/session.js';

test('PRNG is repeatable',()=>{const a=new PRNG(123),b=new PRNG(123);for(let i=0;i<100;i++)assert.equal(a.nextUint(),b.nextUint())});
test('seed domains are separated',()=>{assert.notEqual(domainSeed('train',0),domainSeed('heldout',0))});
test('world generation is deterministic',()=>{const a=new World(77,CURRICULUM[2]),b=new World(77,CURRICULUM[2]);assert.deepEqual(a.food,b.food);assert.deepEqual(a.hazards,b.hazards);assert.deepEqual(a.walls,b.walls);assert.deepEqual(Array.from(a.observe()),Array.from(b.observe()))});
test('observations are finite and correct shape',()=>{const w=new World(5,CURRICULUM[0]);const o=w.observe();assert.equal(o.length,10);for(const x of o)assert.ok(Number.isFinite(x))});
test('all actions produce finite rewards',()=>{for(let a=0;a<7;a++){const w=new World(10+a,CURRICULUM[1]);const r=w.step(a);assert.ok(Number.isFinite(r.reward));assert.equal(r.obs.length,10)}});
test('GAE returns finite aligned arrays',()=>{const t=[{env:0,reward:1,value:.2,done:false},{env:0,reward:.5,value:.3,done:true}];const g=computeGAE(t,new Map([[0,0]]));assert.equal(g.advantages.length,2);assert.ok([...g.advantages,...g.returns].every(Number.isFinite))});
test('model checkpoint roundtrip is exact',()=>{const m=new RecurrentActorCritic(9),n=new RecurrentActorCritic(10);n.restore(m.serialize());for(const k of Object.keys(m.params))assert.deepEqual(Array.from(m.params[k]),Array.from(n.params[k]))});
test('training changes parameters without non-finite values',()=>{const s=new TrainingSession({seed:99,envCount:2,autoCurriculum:false});const before=Array.from(s.model.params.wp);s.trainRollout(8);const after=Array.from(s.model.params.wp);assert.notDeepEqual(after,before);for(const arr of Object.values(s.model.params))for(const x of arr)assert.ok(Number.isFinite(x))});
test('session checkpoint restore preserves step/model state',()=>{const a=new TrainingSession({seed:3,envCount:2,autoCurriculum:false});a.trainRollout(4);const snap=a.snapshot();const b=new TrainingSession({seed:4,envCount:2,autoCurriculum:false});b.restore(snap);assert.equal(b.totalSteps,a.totalSteps);assert.deepEqual(Array.from(b.model.params.bp),Array.from(a.model.params.bp))});
test('reward is decomposed into finite named components',()=>{const w=new World(12345,CURRICULUM[1]);const r=w.step(1);const parts=r.info.rewardParts;for(const k of ['survival','energy','wall','food','hazard','approach','death'])assert.ok(Number.isFinite(parts[k]),k)});
