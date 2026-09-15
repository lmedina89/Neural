import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { CONFIG, VERSION, BUILD_MARKER } from '../src/config.js';
import { PRNG, domainSeed } from '../src/utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../src/sim/curriculum.js';
import { World } from '../src/sim/world.js';
import { RecurrentActorCritic } from '../src/ai/model.js';
import { PPOTrainer } from '../src/ai/ppo.js';
import { CuriosityModule } from '../src/ai/curiosity.js';
import { computeGAE } from '../src/ai/rollout.js';
import { TrainingSession, nextHistoricalMilestoneAfter, assessSkillRetention } from '../src/ai/session.js';
import { evaluateCurriculumSuite, evaluateFullRetentionSuite, evaluateHeldoutGeneralizationSuite, generalizationDiagnostic } from '../src/evaluation/evaluator.js';
import { projectHiddenState } from '../src/visualization/memoryRenderer.js';
import { buildHistoryScene } from '../src/visualization/historyRenderer.js';

test('release identity is v0.1.4.0.2 Spawn Clearance & World Validity Hotfix',()=>{assert.equal(VERSION,'0.1.4.0.2');assert.equal(BUILD_MARKER,'SPAWNCLR-01402')});
test('PRNG is repeatable',()=>{const a=new PRNG(123),b=new PRNG(123);for(let i=0;i<100;i++)assert.equal(a.nextUint(),b.nextUint())});
test('training, validation, comparison, curiosity-audit and final-holdout seed domains are separated',()=>{const domains=['train',CONFIG.validation.seedBase,CONFIG.generalization.compareSeedBase,CONFIG.curiosityAudit.seedBase,CONFIG.generalization.seedBase];const seeds=domains.map(x=>domainSeed(x,0));assert.equal(new Set(seeds).size,seeds.length)});
test('world generation is deterministic',()=>{const a=new World(77,CURRICULUM[2]),b=new World(77,CURRICULUM[2]);assert.deepEqual(a.food,b.food);assert.deepEqual(a.hazards,b.hazards);assert.deepEqual(a.walls,b.walls);assert.deepEqual(Array.from(a.observe()),Array.from(b.observe()))});
test('legacy-valid seed geometry is unchanged by corrective spawn clearance',()=>{
  const w=new World(77,CURRICULUM[2]);
  assert.deepEqual({agent:w.agent,food:w.food,hazards:w.hazards,walls:w.walls},{
    agent:{x:0.43750513881444936,y:0.11546459655277432,vx:0,vy:0,angle:1.2315466043679555,omega:0,energy:1},
    food:[{x:0.7739308758825064,y:0.8218447378464043,r:0.035},{x:0.7385559266898781,y:0.12701161314733328,r:0.035}],
    hazards:[{x:0.23182738048024476,y:0.8437525554653258,r:0.05638465300633106},{x:0.49656747081317015,y:0.6590669780503958,r:0.053714828744414266}],
    walls:[{x:0.42567128468602977,y:0.42951874125662093,w:0.03825735151418485,h:0.14823154198238628},{x:0.2881268523869061,y:0.23953033883429078,w:0.14632758107502014,h:0.042206191745353866}],
  });
});

test('known trapped spawn seed is deterministically corrected away from walls',()=>{
  const a=new World(2,CURRICULUM[3]),b=new World(2,CURRICULUM[3]);
  assert.deepEqual(a.agent,b.agent);
  const margin=CONFIG.world.agentRadius+CONFIG.world.wallMargin;
  for(const wall of a.walls){
    const cx=Math.max(wall.x,Math.min(a.agent.x,wall.x+wall.w));
    const cy=Math.max(wall.y,Math.min(a.agent.y,wall.y+wall.h));
    assert.ok(Math.hypot(a.agent.x-cx,a.agent.y-cy)>=margin-1e-12);
  }
});

test('thousands of generated worlds keep agent, food and hazards clear of walls and boundaries',()=>{
  const rectDistance=(p,w)=>{const cx=Math.max(w.x,Math.min(p.x,w.x+w.w));const cy=Math.max(w.y,Math.min(p.y,w.y+w.h));return Math.hypot(p.x-cx,p.y-cy)};
  const clear=(p,r,w)=>{const c=r+CONFIG.world.wallMargin;assert.ok(p.x>=c-1e-12&&p.x<=1-c+1e-12&&p.y>=c-1e-12&&p.y<=1-c+1e-12);for(const wall of w.walls)assert.ok(rectDistance(p,wall)>=c-1e-12)};
  for(let seed=0;seed<2000;seed++)for(const stage of CURRICULUM){const w=new World(seed,stage);clear(w.agent,CONFIG.world.agentRadius,w);for(const f of w.food)clear(f,f.r,w);for(const h of w.hazards)clear(h,h.r,w)}
});

test('food respawn also remains clear of walls',()=>{
  const rectDistance=(p,w)=>{const cx=Math.max(w.x,Math.min(p.x,w.x+w.w));const cy=Math.max(w.y,Math.min(p.y,w.y+w.h));return Math.hypot(p.x-cx,p.y-cy)};
  for(let seed=0;seed<160;seed++){
    const w=new World(seed,CURRICULUM[3]);
    const f=w.food[0]; w.agent.x=f.x; w.agent.y=f.y; w.agent.vx=0; w.agent.vy=0; w.step(0);
    const c=CONFIG.world.foodRadius+CONFIG.world.wallMargin;
    for(const food of w.food)for(const wall of w.walls)assert.ok(rectDistance(food,wall)>=c-1e-12);
  }
});
test('world info carries curriculum identity',()=>{const w=new World(7,CURRICULUM[2]);assert.equal(w.info().stageId,2);assert.equal(w.info().stageName,'Obstacle Avoidance')});
test('observations are finite and correct shape',()=>{const w=new World(5,CURRICULUM[0]);const o=w.observe();assert.equal(o.length,10);for(const x of o)assert.ok(Number.isFinite(x))});
test('all actions produce finite rewards',()=>{for(let a=0;a<7;a++){const w=new World(10+a,CURRICULUM[1]);const r=w.step(a);assert.ok(Number.isFinite(r.reward));assert.equal(r.obs.length,10)}});
test('GAE returns finite aligned arrays',()=>{const t=[{env:0,reward:1,value:.2,done:false},{env:0,reward:.5,value:.3,done:true}];const g=computeGAE(t,new Map([[0,0]]));assert.equal(g.advantages.length,2);assert.ok([...g.advantages,...g.returns].every(Number.isFinite))});
test('model checkpoint roundtrip is exact',()=>{const m=new RecurrentActorCritic(9),n=new RecurrentActorCritic(10);n.restore(m.serialize());for(const k of Object.keys(m.params))assert.deepEqual(Array.from(m.params[k]),Array.from(n.params[k]))});
test('training changes parameters without non-finite values',()=>{const s=new TrainingSession({seed:99,envCount:2,autoCurriculum:false});const before=Array.from(s.model.params.wp);s.trainRollout(8);const after=Array.from(s.model.params.wp);assert.notDeepEqual(after,before);for(const arr of Object.values(s.model.params))for(const x of arr)assert.ok(Number.isFinite(x))});

test('curriculum promotion uses sustained rolling performance when retention gate permits',()=>{const c=new CurriculumManager(0);let event=null;for(let i=1;i<=CONFIG.curriculum.minSamples;i++)event=c.noteEpisode({food:3,survived:true},i,{promotionAllowed:true})||event;assert.ok(event);assert.equal(event.reason,'promotion');assert.equal(c.stage,1)});
test('curriculum promotion can be blocked without changing stage',()=>{const c=new CurriculumManager(1);let event=null;for(let i=1;i<=CONFIG.curriculum.minSamples;i++)event=c.noteEpisode({food:3,survived:true},i,{promotionAllowed:false,gateReason:'Foraging retention low'})||event;assert.ok(event);assert.equal(event.reason,'promotion-blocked');assert.equal(c.stage,1);assert.match(event.gateReason,/retention/)});
test('curriculum can demote after sustained collapse',()=>{const c=new CurriculumManager(2);c.cooldownRemaining=0;let event=null;for(let i=1;i<=CONFIG.curriculum.minSamples;i++)event=c.noteEpisode({food:0,survived:false},i)||event;assert.ok(event);assert.equal(event.reason,'demotion');assert.equal(c.stage,1)});

test('validation suite is deterministic for unchanged model and fixed protocol',()=>{const m=new RecurrentActorCritic(42);const a=evaluateCurriculumSuite(m,3,{episodesPerStage:3,seedBase:'validation:test'});const b=evaluateCurriculumSuite(m,3,{episodesPerStage:3,seedBase:'validation:test'});assert.equal(a.protocol,b.protocol);assert.equal(a.score,b.score);assert.deepEqual(a.stageResults,b.stageResults)});
test('full retention validation covers every skill with confidence ranges',()=>{const m=new RecurrentActorCritic(43);const r=evaluateFullRetentionSuite(m,{episodesPerStage:4,seedBase:'validation:full-test'});assert.equal(r.maxStage,CURRICULUM.length-1);assert.equal(r.stageResults.length,CURRICULUM.length);assert.ok(r.stageResults.every(x=>Number.isFinite(x.skillScore)&&x.skillScore>=0&&x.skillScore<=1&&x.skillCiLow<=x.skillScore&&x.skillScore<=x.skillCiHigh));assert.ok(r.balancedCiLow<=r.balancedScore&&r.balancedScore<=r.balancedCiHigh);for(const k of ['balanced','overall','forager','survivor','efficiency'])assert.ok(Number.isFinite(r.categoryScores[k]))});
test('automatic archive establishes all specialist categories',()=>{const s=new TrainingSession({seed:12,envCount:2,autoCurriculum:false});const first=s.runValidation();assert.ok(first.improved);for(const k of ['balanced','overall','forager','survivor','efficiency'])assert.ok(s.getArchiveBrain(k)?.model,k);assert.equal(s.bestBrain,s.getArchiveBrain('balanced'))});
test('confirmed behavioral regression is observational and never auto-restores learner weights',()=>{const s=new TrainingSession({seed:13,envCount:2,autoCurriculum:false});s.runValidation();const learnerBefore=JSON.stringify(s.model.serialize());const beforeSteps=123456;s.totalSteps=beforeSteps;const prior=s.bestArchive.balanced.validation;prior.categoryScores.balanced=Math.min(1,prior.categoryScores.balanced+0.6);prior.balancedCiLow=Math.min(1,prior.categoryScores.balanced-0.01);prior.balancedCiHigh=prior.categoryScores.balanced;const r1=s.runValidation();assert.equal(r1.balancedEvidence,true);assert.equal(r1.balancedConfirmed,false);assert.equal(r1.autoRollback,null);const r2=s.runValidation();assert.equal(r2.balancedConfirmed,true);assert.equal(r2.autoRollback,null);assert.equal(s.totalSteps,beforeSteps);assert.equal(JSON.stringify(s.model.serialize()),learnerBefore);assert.equal(s.retentionStatus.intervention,'observe-only')});
test('manual Champion fork creates a new lineage without rewinding experience',()=>{const s=new TrainingSession({seed:14,envCount:2,autoCurriculum:false});s.runValidation();const brain=s.getArchiveBrain('forager');const protectedBp=Array.from(brain.model.params.bp);const oldLineage=s.learnerLineage.id;s.totalSteps=12345;s.model.params.bp[0]+=99;const event=s.forkFromChampion('forager');assert.deepEqual(Array.from(s.model.params.bp),protectedBp);assert.equal(s.totalSteps,12345);assert.equal(event.category,'forager');assert.notEqual(s.learnerLineage.id,oldLineage);assert.equal(s.learnerLineage.parent.sourceSteps,brain.savedAtSteps)});

test('PPO uses scheduled entropy, bounded adaptive LR and finite KL stats',()=>{const s=new TrainingSession({seed:19,envCount:2,autoCurriculum:false});s.totalSteps=4_000_000;const {metric}=s.trainRollout(8);assert.ok(metric.entropyCoef<CONFIG.ppo.entropyStart);assert.ok(metric.entropyCoef>=CONFIG.ppo.entropyEnd);assert.ok(metric.learningRate>=CONFIG.ppo.minLearningRate&&metric.learningRate<=CONFIG.ppo.maxLearningRate);assert.ok(Number.isFinite(metric.maxEpochKL));assert.ok(metric.epochsRun>=1&&metric.epochsRun<=CONFIG.ppo.epochs)});
test('PPO hard-KL guard rejects an obviously destructive update and restores model',()=>{const model=new RecurrentActorCritic(55);const trainer=new PPOTrainer(model,56);const obs=new Float64Array(CONFIG.model.obsSize);obs[9]=1;const h=model.zeroHidden();const before=model.serialize();const transitions=[0,1,2,3].map((action,i)=>({obs,hPrev:h,action:action%CONFIG.model.actionSize,logProb:-20-i,value:0,reward:0,done:false}));const advantages=new Float64Array([1,-1,1,-1]);const returns=new Float64Array([1,-1,1,-1]);const stats=trainer.update(transitions,advantages,returns,{trainingStep:0});assert.equal(stats.updateRejected,true);assert.deepEqual(model.serialize(),before);assert.ok(trainer.learningRate<CONFIG.ppo.learningRate)});

test('schema-9 checkpoint roundtrip preserves policy, curiosity, Champions, lineage, Hall, branches and rehearsal state',()=>{const a=new TrainingSession({seed:3,envCount:2,autoCurriculum:false});a.curriculum.stage=2;for(let i=0;i<24;i++)a.resetEnv(i%2);a.trainRollout(4);a.runValidation();a.pinChampion('balanced');const oldLineage=a.learnerLineage.id;a.forkFromChampion('balanced');const snap=a.snapshot();assert.equal(snap.schema,9);assert.ok(snap.hallOfFame.entries.length>=1);assert.ok(snap.frozenLearners.some(x=>x.id===oldLineage));const b=new TrainingSession({seed:4,envCount:2,autoCurriculum:false});b.restore(snap);assert.equal(b.totalSteps,a.totalSteps);assert.deepEqual(Array.from(b.model.params.bp),Array.from(a.model.params.bp));assert.equal(b.bestBrain.savedAtSteps,a.bestBrain.savedAtSteps);assert.deepEqual(b.skillBestRecords,a.skillBestRecords);assert.deepEqual(b.promotionStreaks,a.promotionStreaks);assert.deepEqual(b.learnerLineage,a.learnerLineage);assert.deepEqual(b.rehearsalEpisodeHistory,a.rehearsalEpisodeHistory);assert.equal(b.hallOfFame.length,a.hallOfFame.length);assert.equal(b.frozenLearners.length,a.frozenLearners.length);assert.equal(b.learnerExperienceSteps,a.learnerExperienceSteps);assert.deepEqual(b.curiosity.serialize(),a.curiosity.serialize())});
test('schema-3 v0.1.1 archive is preserved for inspection and scheduled for v3 recalibration',()=>{const a=new TrainingSession({seed:31,envCount:2,autoCurriculum:false});a.runValidation();const snap=a.snapshot();const legacy={...snap,schema:3};delete legacy.skillBestRecords;delete legacy.skillRegressionStreaks;delete legacy.balancedRegressionStreak;delete legacy.pendingArchiveMigration;delete legacy.archiveNeedsRebaseline;const b=new TrainingSession({seed:32,envCount:2,autoCurriculum:false});b.restore(legacy);assert.equal(b.archiveNeedsRebaseline,true);assert.ok(b.bestBrain?.model);assert.ok(b.pendingArchiveMigration.length>=1);assert.equal(b.nextValidationStep,b.totalSteps);assert.ok(b.legacyValidationHistory.length>=1);const v=b.runValidation();assert.equal(b.archiveNeedsRebaseline,false);assert.ok(v.migratedArchive.length>=1);assert.match(v.validation.protocol,/retention-v3-ci/)});
test('schema-2 save migration preserves old protected best and schedules immediate rebaseline',()=>{const a=new TrainingSession({seed:23,envCount:2,autoCurriculum:false});a.trainRollout(4);a.runValidation();const snap=a.snapshot();const legacy={schema:2,seed:snap.seed,totalSteps:3_758_688,totalEpisodes:6000,envSeedCursor:snap.envSeedCursor,curriculum:snap.curriculum,model:snap.model,optimizer:snap.optimizer,metrics:snap.metrics,milestones:snap.milestones,bestBrain:snap.bestBrain,bestBrains:{legacy:snap.bestBrain},validationHistory:[],lastValidationStep:3_500_000,nextValidationStep:3_750_000,rollbackHistory:[]};const b=new TrainingSession({seed:99,envCount:2,autoCurriculum:false});b.restore(legacy);assert.equal(b.totalSteps,3_758_688);assert.ok(b.pendingLegacyBest?.model);assert.equal(b.bestBrain.savedAtSteps,snap.bestBrain.savedAtSteps);assert.equal(b.nextValidationStep,b.totalSteps);const v=b.runValidation();assert.equal(v.validation.maxStage,CURRICULUM.length-1);assert.equal(b.pendingLegacyBest,null);assert.ok(b.getArchiveBrain('balanced'))});
test('schema-1 long-run migration snapshots exact loaded brain and does not backfill fake milestones',()=>{const a=new TrainingSession({seed:44,envCount:2,autoCurriculum:false});const snap=a.snapshot();const steps=2_122_848;const legacy={schema:1,seed:snap.seed,totalSteps:steps,totalEpisodes:3000,envSeedCursor:snap.envSeedCursor,curriculum:{stage:2,history:[]},model:snap.model,optimizer:snap.optimizer,metrics:[],milestones:snap.milestones};const b=new TrainingSession({seed:45,envCount:2,autoCurriculum:false});b.restore(legacy);assert.equal(b.nextMilestoneStep,2_250_000);assert.ok(b.milestones.has(steps));assert.equal(b.milestones.has(1_250_000),false);assert.equal(b.nextValidationStep,b.totalSteps)});

test('reward is decomposed into finite named components',()=>{const w=new World(12345,CURRICULUM[1]);const r=w.step(1);const parts=r.info.rewardParts;for(const k of ['survival','energy','wall','food','hazard','approach','death'])assert.ok(Number.isFinite(parts[k]),k)});
test('v0.1.4.0.2 UI preserves Hall, branching, performance, lineage, rehearsal and curiosity-audit diagnostics',async()=>{const html=await readFile(new URL('../index.html',import.meta.url),'utf8');for(const id of ['brainSource','hallBrainOptions','bestOption','overallOption','foragerOption','survivorOption','efficiencyOption','pinChampionBtn','restoreBestBtn','branchSelect','switchBranchBtn','researchLineageVal','policyOriginVal','hallVal','branchesVal','hallList','skillRetention','lrVal','klVal','epochsVal','retentionAlert','lineageVal','rehearsalVal','promotionVal','simSpeedVal','ppoMsVal','fpsVal','uiMsVal','validationMsVal','storageMsVal','curiosityInfluenceVal','curiosityAppliedVal','curiosityShareVal','curiosityResetsVal','curiosityBudgetUseVal','curiosityExhaustVal','curiosityModeSelect','startCuriosityAuditBtn','switchCuriosityAuditBtn','endCuriosityAuditBtn','curiosityAuditResults'])assert.match(html,new RegExp(`id="${id}"`));assert.match(html,/v0\.1\.4\.0\.2/);assert.match(html,/Fork New Learner/);assert.match(html,/Hall of Fame/);assert.match(html,/Adaptive/);assert.match(html,/observational only/);assert.match(html,/final holdout only diagnoses/)});

test('skill-regression evidence is watch-first and confirmed only on repeat',()=>{const stages=[{stage:0,name:'Motor Nursery',skillScore:.33,skillCiLow:.25,skillCiHigh:.41}];const best=[{stage:0,name:'Motor Nursery',score:.74,ciLow:.66,ciHigh:.82,atSteps:1000},null,null,null];const a=assessSkillRetention(stages,best,[0,0,0,0]);assert.equal(a.alerts.length,1);assert.equal(a.alerts[0].severity,'catastrophic');assert.equal(a.alerts[0].confirmed,false);const b=assessSkillRetention(stages,best,a.streaks);assert.equal(b.alerts[0].confirmed,true);assert.equal(b.alerts[0].streak,2)});
test('final held-out generalization suite is all-skills and isolated from validation',()=>{const m=new RecurrentActorCritic(77);const r=evaluateHeldoutGeneralizationSuite(m,{episodesPerStage:2,seedBase:'heldout:final:test'});assert.equal(r.stageResults.length,CURRICULUM.length);assert.equal(r.episodes,2*CURRICULUM.length);assert.match(r.protocol,/heldout-generalization-v2/);assert.doesNotMatch(r.protocol,/validation:v3/)});
test('generalization diagnostic identifies validation/heldout disagreement without selecting a brain',()=>{const lv={balancedScore:.40},pv={balancedScore:.60},lh={balancedScore:.70},ph={balancedScore:.50};const d=generalizationDiagnostic(lv,pv,lh,ph,{margin:.04});assert.equal(d.conflict,true);assert.equal(d.validationPref,'protected');assert.equal(d.heldoutPref,'latest');assert.equal(d.champion,'latest')});
test('final Unseen Test path is diagnostic-only and does not call Learner or Champion mutators',async()=>{const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');const block=main.slice(main.indexOf('async function runUnseen()'),main.indexOf('async function compareBrains()'));for(const forbidden of ['considerCandidate(','restoreBest(','forkFromChampion(','runValidation(','saveCheckpoint('])assert.equal(block.includes(forbidden),false,forbidden);assert.match(block,/FINAL HOLDOUT IS DIAGNOSTIC ONLY/)});
test('legacy archive is recalibrated before current skill regression is assessed',async()=>{const src=await readFile(new URL('../src/ai/session.js',import.meta.url),'utf8');const start=src.indexOf('  runValidation() {');const run=src.slice(start,src.indexOf('\n  makeCandidate(',start));const baseline=run.indexOf('updateSkillBestRecords(this.skillBestRecords, candidate.validation.stageResults');const assess=run.indexOf('assessSkillRetention(');assert.ok(baseline>=0&&assess>=0&&baseline<assess)});
test('continual rehearsal keeps earlier skills in the training distribution',()=>{const s=new TrainingSession({seed:101,envCount:2,autoCurriculum:false});s.curriculum.stage=3;assert.deepEqual(s.trainingMix().map(x=>Number(x.toFixed(2))),[0.10,0.15,0.20,0.55]);const counts=[0,0,0,0];for(let i=0;i<4000;i++)counts[s.chooseTrainingStage(i)]++;assert.ok(counts.every(x=>x>0));const f=counts.map(x=>x/4000);assert.ok(Math.abs(f[0]-.10)<.03);assert.ok(Math.abs(f[1]-.15)<.03);assert.ok(Math.abs(f[2]-.20)<.03);assert.ok(Math.abs(f[3]-.55)<.04)});
test('rehearsal never samples a future curriculum stage',()=>{const s=new TrainingSession({seed:102,envCount:2,autoCurriculum:false});s.curriculum.stage=2;const seen=new Set();for(let i=0;i<500;i++)seen.add(s.chooseTrainingStage(i));assert.ok(seen.has(0)&&seen.has(1)&&seen.has(2));assert.equal(seen.has(3),false)});
test('session curriculum progression is autonomous and not blocked by retention observations',()=>{const s=new TrainingSession({seed:103,envCount:2,autoCurriculum:true});s.retentionStatus={alerts:[{stage:0,name:'Motor Nursery'}],forgetting:[{stage:0}],healthy:false};assert.equal(s.promotionGate().allowed,true);assert.match(s.promotionGate().reason,/autonomous/)});
test('Champion promotion requires repeated challenger evidence after a Champion exists',()=>{const s=new TrainingSession({seed:104,envCount:2,autoCurriculum:false});s.runValidation();const champion=s.getArchiveBrain('balanced');const base=structuredClone(champion.validation);base.categoryScores={...base.categoryScores,balanced:champion.validation.categoryScores.balanced+.10};base.balancedScore=base.categoryScores.balanced;s.totalSteps=1000;let r=s.considerCandidate(s.makeCandidate(base));assert.equal(r.updates.includes('balanced'),false);assert.equal(r.pending.find(x=>x.category==='balanced').streak,1);assert.equal(s.getArchiveBrain('balanced').savedAtSteps,champion.savedAtSteps);s.totalSteps=2000;r=s.considerCandidate(s.makeCandidate(base));assert.equal(r.updates.includes('balanced'),true);assert.equal(s.getArchiveBrain('balanced').savedAtSteps,2000);assert.equal(s.getArchiveBrain('balanced').promotionEvidence.confirmations,2)});
test('schema-4 migration preserves Champion archive and starts a new continual Learner lineage',()=>{const a=new TrainingSession({seed:105,envCount:2,autoCurriculum:false});a.runValidation();const snap=a.snapshot();const legacy={...snap,schema:4};for(const k of ['promotionStreaks','promotionCandidates','learnerLineage','lineageCounter','lineageHistory','rehearsalEpisodeHistory'])delete legacy[k];const b=new TrainingSession({seed:106,envCount:2,autoCurriculum:false});b.restore(legacy);assert.equal(b.totalSteps,a.totalSteps);assert.ok(b.getArchiveBrain('balanced')?.model);assert.equal(b.nextValidationStep,b.totalSteps);assert.match(b.learnerLineage.reason,/schema-4/);assert.deepEqual(b.promotionStreaks,{balanced:0,overall:0,forager:0,survivor:0,efficiency:0})});
test('behavioral validation cannot mutate Learner weights even when regression is confirmed',async()=>{const src=await readFile(new URL('../src/ai/session.js',import.meta.url),'utf8');const start=src.indexOf('  runValidation() {');const run=src.slice(start,src.indexOf('\n  makeCandidate(',start));assert.doesNotMatch(run,/model\.restore\(/);assert.doesNotMatch(run,/trainer\.restore\(/);assert.match(run,/behavioral regression is observed, never auto-restored/)});
test('historical milestone schedule continues beyond one million steps',()=>{assert.equal(nextHistoricalMilestoneAfter(1_000_000),1_250_000);assert.equal(nextHistoricalMilestoneAfter(2_122_848),2_250_000);assert.equal(nextHistoricalMilestoneAfter(4_999_999),5_000_000);assert.equal(nextHistoricalMilestoneAfter(5_000_000),5_500_000);assert.equal(nextHistoricalMilestoneAfter(20_000_000),21_000_000);assert.equal(nextHistoricalMilestoneAfter(100_000_000),105_000_000)});
test('recovery-safe UI keeps explicit manual/autosave loading and lower-step overwrite guard',async()=>{const html=await readFile(new URL('../index.html',import.meta.url),'utf8');const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');assert.match(html,/Load Manual/);assert.match(html,/Load Autosave/);assert.doesNotMatch(main,/loadNewestCheckpoint/);assert.match(main,/storedSteps > session\.totalSteps/);assert.match(main,/lower-step brain was NOT allowed to overwrite it/)});
test('explicit checkpoint restore pauses training for verification',async()=>{const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');assert.match(main,/paused = true/);assert.match(main,/Training is PAUSED so you can verify it/)});


test('Hall of Fame pin is immutable and deduplicates exact Champion policy',()=>{
  const s=new TrainingSession({seed:201,envCount:2,autoCurriculum:false});
  s.runValidation();
  const original=JSON.stringify(s.getArchiveBrain('balanced').model);
  const a=s.pinChampion('balanced');
  const b=s.pinChampion('balanced');
  assert.equal(a.created,true);
  assert.equal(b.created,false);
  assert.equal(a.entry.id,b.entry.id);
  s.model.params.bp[0]+=123;
  assert.equal(JSON.stringify(s.getHallEntry(a.entry.id).model),original);
});

test('Hall of Fame merge deduplicates model fingerprints across persistent and checkpoint archives',()=>{
  const a=new TrainingSession({seed:202,envCount:2,autoCurriculum:false});
  a.runValidation();
  const pinned=a.pinChampion('balanced').entry;
  const payload=a.exportHallOfFame();
  const b=new TrainingSession({seed:203,envCount:2,autoCurriculum:false});
  b.mergeHallOfFame(payload);
  b.mergeHallOfFame(payload);
  assert.equal(b.hallOfFame.length,1);
  assert.equal(b.hallOfFame[0].id,pinned.id);
});

test('manual Champion fork freezes prior Learner and resets branch-local experience without rewinding global age',()=>{
  const s=new TrainingSession({seed:204,envCount:2,autoCurriculum:false});
  s.runValidation();
  s.totalSteps=4_800_240;
  s.learnerExperienceSteps=650_000;
  const prior=s.learnerLineage.id;
  const event=s.forkFromChampion('balanced');
  assert.equal(s.totalSteps,4_800_240);
  assert.equal(s.learnerExperienceSteps,0);
  assert.notEqual(s.learnerLineage.id,prior);
  assert.ok(s.frozenLearners.some(x=>x.id===prior));
  assert.equal(event.category,'balanced');
  assert.equal(event.preservedLineageId,prior);
});

test('frozen Learner switch preserves current branch and restores target lineage/model while global age stays monotonic',()=>{
  const s=new TrainingSession({seed:205,envCount:2,autoCurriculum:false});
  s.runValidation();
  s.totalSteps=100_000;
  const root=s.learnerLineage.id;
  const rootModel=JSON.stringify(s.model.serialize());
  s.forkFromChampion('balanced');
  const branch=s.learnerLineage.id;
  s.model.params.bp[0]+=7;
  s.totalSteps=120_000;
  const e=s.switchToFrozenLearner(root);
  assert.equal(s.totalSteps,120_000);
  assert.equal(s.learnerLineage.id,root);
  assert.equal(JSON.stringify(s.model.serialize()),rootModel);
  assert.ok(s.frozenLearners.some(x=>x.id===branch));
  assert.equal(e.preservedLineageId,branch);
});

test('forking from Hall of Fame creates a new lineage and preserves current Learner branch',()=>{
  const s=new TrainingSession({seed:206,envCount:2,autoCurriculum:false});
  s.runValidation();
  const hall=s.pinChampion('balanced').entry;
  const prior=s.learnerLineage.id;
  const event=s.forkFromHallOfFame(hall.id);
  assert.equal(event.hallId,hall.id);
  assert.notEqual(s.learnerLineage.id,prior);
  assert.ok(s.frozenLearners.some(x=>x.id===prior));
  assert.equal(s.learnerLineage.parent.type,'hall-of-fame');
});

test('capture-free model forward and action paths preserve numerical policy outputs',()=>{
  const m=new RecurrentActorCritic(207);
  const obs=new Float64Array(CONFIG.model.obsSize); for(let i=0;i<obs.length;i++) obs[i]=(i-3)/10;
  const h=m.zeroHidden();
  const a=m.forward(obs,h,true); const b=m.forward(obs,h,false);
  assert.deepEqual(Array.from(a.h),Array.from(b.h));
  assert.deepEqual(Array.from(a.probs),Array.from(b.probs));
  assert.equal(a.value,b.value);
  const r1=new PRNG(555), r2=new PRNG(555);
  const x=m.act(obs,h,r1,false,true), y=m.act(obs,h,r2,false,false);
  assert.equal(x.action,y.action); assert.equal(x.logProb,y.logProb); assert.equal(x.value,y.value); assert.deepEqual(Array.from(x.hidden),Array.from(y.hidden));
});

test('training profiler reports finite hot-path timing and throughput without validation contamination',()=>{
  const s=new TrainingSession({seed:208,envCount:2,autoCurriculum:false});
  s.nextValidationStep=Number.MAX_SAFE_INTEGER;
  const {metric}=s.trainRollout(12);
  for(const k of ['simulationMs','advantageMs','ppoMs','bookkeepingMs','validationMs','coreMs','trainingStepsPerSec','simulationStepsPerSec','ppoUpdatesPerSec']) assert.ok(Number.isFinite(metric.profile[k]),k);
  assert.equal(metric.profile.validationMs,0);
  assert.equal(metric.throughput,metric.profile.trainingStepsPerSec);
});

test('PPO optimizer RNG state survives serialize/restore for reproducible continuation',()=>{
  const a=new TrainingSession({seed:209,envCount:2,autoCurriculum:false});
  a.nextValidationStep=Number.MAX_SAFE_INTEGER;
  a.trainRollout(8);
  const snap=a.snapshot();
  const b=new TrainingSession({seed:999,envCount:2,autoCurriculum:false});
  b.restore(snap);
  assert.equal(b.trainer.rng.state >>> 0,a.trainer.rng.state >>> 0);
  assert.equal(b.actionRng.state >>> 0,a.actionRng.state >>> 0);
});

test('persistent Hall uses a separate IndexedDB slot and pre-curiosity migration auto-pins only in app layer',async()=>{
  const storage=await readFile(new URL('../src/storage/checkpoints.js',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(storage,/HALL_OF_FAME_SLOT = 'hall-of-fame'/);
  assert.match(storage,/saveHallOfFame/);
  assert.match(storage,/loadHallOfFameRecord/);
  assert.match(main,/cp\.schema <= 6/);
  assert.match(main,/pinChampion\('balanced'/);
});

test('training UI is throttled and neural/action render work is not forced every training animation frame',async()=>{
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(main,/maybeUpdateUI/);
  assert.match(main,/learnUiHz/);
  assert.match(main,/learnRenderHz/);
  assert.match(main,/actionUi/);
  assert.match(main,/controlSignature/);
});


test('curiosity predictor is tiny, finite and separate from the 1,040-parameter policy',()=>{
  const c=new CuriosityModule(301);
  assert.equal(new RecurrentActorCritic(1).paramCount(),1040);
  assert.equal(c.paramCount(),441);
  for(const arr of Object.values(c.params))for(const x of arr)assert.ok(Number.isFinite(x));
});

test('curiosity forward model learns a deterministic transition mapping',()=>{
  const c=new CuriosityModule(302);
  const samples=[];
  for(let n=0;n<256;n++){
    const obs=new Float64Array(CONFIG.model.obsSize);
    for(let i=0;i<9;i++)obs[i]=Math.sin((n+1)*(i+2)*0.017)*0.7;
    obs[9]=1;
    const action=n%CONFIG.model.actionSize;
    const nextObs=new Float64Array(obs);
    for(let i=0;i<9;i++)nextObs[i]=Math.tanh(obs[i]*0.82+(action-3)*0.025+(i-4)*0.004);
    samples.push({obs,action,nextObs});
  }
  const meanError=()=>samples.reduce((sum,tr)=>{const f=c.forward(tr.obs,tr.action,false);return sum+c.predictionError(tr.nextObs,f.prediction)},0)/samples.length;
  const before=meanError();
  for(let i=0;i<35;i++)c.trainBatch(samples);
  const after=meanError();
  assert.ok(after<before*0.70,`expected predictor learning: ${before} -> ${after}`);
});

test('intrinsic curiosity reward is positive-only, tightly bounded, budgeted and zero on terminal transitions',()=>{
  const c=new CuriosityModule(303);
  const obs=new Float64Array(CONFIG.model.obsSize);obs[9]=1;
  const next=new Float64Array(obs);next[0]=0.9;next[3]=0.7;
  const a=c.scoreTransition(obs,1,next,{remainingBudget:0.0004,terminal:false});
  assert.ok(a.bonus>=0&&a.bonus<=0.0004&&a.bonus<=CONFIG.curiosity.maxStepBonus);
  const t=c.scoreTransition(obs,1,next,{remainingBudget:CONFIG.curiosity.maxEpisodeBonus,terminal:true});
  assert.equal(t.bonus,0);
});

test('training mixes intrinsic bonus into PPO reward while external episode metrics remain external',()=>{
  const s=new TrainingSession({seed:304,envCount:2,autoCurriculum:false});
  s.nextValidationStep=Number.MAX_SAFE_INTEGER;
  const {metric}=s.trainRollout(16);
  assert.ok(Number.isFinite(metric.curiosity.meanIntrinsicReward));
  assert.ok(metric.curiosity.meanIntrinsicReward>=0&&metric.curiosity.meanIntrinsicReward<=CONFIG.curiosity.maxStepBonus);
  assert.ok(Number.isFinite(metric.profile.curiosityMs));
  assert.ok(s.curiosityEpisodeBudget.every(x=>x>=0&&x<=CONFIG.curiosity.maxEpisodeBonus));
  assert.equal(metric.meanReturn,s.episodeHistory.length?s.episodeHistory.slice(-40).reduce((q,x)=>q+(x.totalReward||0),0)/s.episodeHistory.slice(-40).length:0);
});

test('evaluation code remains curiosity-free and cannot award intrinsic reward',async()=>{
  const src=await readFile(new URL('../src/evaluation/evaluator.js',import.meta.url),'utf8');
  assert.doesNotMatch(src,/CuriosityModule|intrinsicReward|curiosity\.scoreTransition/);
  const model=new RecurrentActorCritic(305);
  const a=evaluateFullRetentionSuite(model,{episodesPerStage:2,seedBase:'curiosity-isolation'});
  const b=evaluateFullRetentionSuite(model,{episodesPerStage:2,seedBase:'curiosity-isolation'});
  assert.deepEqual(a,b);
});

test('schema-6 migration preserves old policy and starts a fresh curiosity model; schema-9 then roundtrips it',()=>{
  const a=new TrainingSession({seed:306,envCount:2,autoCurriculum:false});
  a.trainRollout(8);
  const current=a.snapshot();
  const legacy={...structuredClone(current),schema:6};delete legacy.curiosity;
  const b=new TrainingSession({seed:999,envCount:2,autoCurriculum:false});
  b.restore(legacy);
  assert.deepEqual(b.model.serialize(),a.model.serialize());
  assert.equal(b.curiosity.errorSamples,0);
  b.trainRollout(8);
  assert.ok(b.curiosity.errorSamples>0);
  const c=new TrainingSession({seed:998,envCount:2,autoCurriculum:false});
  c.restore(b.snapshot());
  assert.deepEqual(c.curiosity.serialize(),b.curiosity.serialize());
});

test('curiosity visualization and UI are wired to real predictor telemetry',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  const renderer=await readFile(new URL('../src/visualization/curiosityRenderer.js',import.meta.url),'utf8');
  assert.match(html,/CURIOSITY \/ PREDICTION/);
  assert.match(html,/evaluation curiosity OFF/);
  assert.match(main,/session\.lastCuriosity/);
  assert.match(main,/curiosityRenderer\.draw/);
  assert.match(renderer,/prediction/);
  assert.match(renderer,/actual/);
});


test('observe-only curiosity keeps predictor learning and potential diagnostics while PPO receives exactly zero intrinsic reward',()=>{
  const s=new TrainingSession({seed:401,envCount:2,autoCurriculum:false});
  s.setCuriosityRewardMode('observe');
  s.nextValidationStep=Number.MAX_SAFE_INTEGER;
  const {metric}=s.trainRollout(32);
  assert.equal(metric.curiosity.rewardMode,'observe');
  assert.equal(metric.curiosity.intrinsicRewardSum,0);
  assert.ok(metric.curiosity.potentialIntrinsicRewardSum>=0);
  assert.ok(s.curiosity.errorSamples>0);
  assert.equal(s.curiosityRewardMode,'observe');
});

test('curiosity episode accounting treats displayed budget as remaining and reset restores the full allowance',()=>{
  const s=new TrainingSession({seed:402,envCount:1,autoCurriculum:false});
  assert.equal(s.curiosityEpisodeBudget[0],CONFIG.curiosity.maxEpisodeBonus);
  s.curiosityEpisodeBudget[0]=0.031;
  s.resetEnv(0);
  assert.equal(s.curiosityEpisodeBudget[0],CONFIG.curiosity.maxEpisodeBonus);
  assert.equal(s.curiosityEpisodeStats[0].potentialIntrinsicReward,0);
});

test('curiosity A/B audit creates matched descendants, freezes curriculum adaptation and never promotes a Champion from audit evaluations',()=>{
  const s=new TrainingSession({seed:403,envCount:2,autoCurriculum:true});
  s.curriculum.stage=3;
  s.runValidation();
  const championBefore=s.getArchiveBrain('balanced').savedAtSteps;
  const modelBefore=s.model.serialize();
  const optimizerBefore=s.trainer.serialize();
  const curiosityBefore=s.curiosity.serialize();
  const audit=s.startCuriosityAudit();
  assert.equal(audit.active,true);
  assert.equal(audit.role,'control');
  assert.equal(s.curiosityRewardMode,'observe');
  assert.equal(s.autoCurriculum,false);
  const q=s.frozenLearners.find(x=>x.auditRole==='curiosity');
  assert.ok(q);
  assert.deepEqual(q.model,modelBefore);
  assert.deepEqual(q.optimizer,optimizerBefore);
  assert.deepEqual(q.curiosity,curiosityBefore);
  assert.deepEqual(s.model.serialize(),modelBefore);
  s.learnerExperienceSteps=CONFIG.curiosityAudit.checkpointInterval;
  const record=s.maybeRunCuriosityAuditEvaluation();
  assert.equal(record.role,'control');
  assert.equal(record.checkpointSteps,CONFIG.curiosityAudit.checkpointInterval);
  assert.equal(s.getArchiveBrain('balanced').savedAtSteps,championBefore);
  assert.equal(s.validationHistory.length,1);
});

test('curiosity A/B switch restores branch-specific reward influence and branch-local progress',()=>{
  const s=new TrainingSession({seed:404,envCount:2,autoCurriculum:true});
  s.startCuriosityAudit();
  s.learnerExperienceSteps=123456;
  const e=s.switchCuriosityAuditBranch('curiosity');
  assert.equal(e.role,'curiosity');
  assert.equal(s.auditRole,'curiosity');
  assert.equal(s.curiosityRewardMode,'reward');
  assert.equal(s.learnerExperienceSteps,0);
  const control=s.frozenLearners.find(x=>x.auditRole==='control');
  assert.equal(control.learnerExperienceSteps,123456);
  s.switchCuriosityAuditBranch('control');
  assert.equal(s.auditRole,'control');
  assert.equal(s.curiosityRewardMode,'observe');
  assert.equal(s.learnerExperienceSteps,123456);
});

test('schema-9 roundtrip preserves an active curiosity audit and schema-7 migration defaults safely to reward-on with no audit',()=>{
  const a=new TrainingSession({seed:405,envCount:2,autoCurriculum:true});
  a.startCuriosityAudit();
  a.learnerExperienceSteps=654321;
  const snap=a.snapshot();
  const b=new TrainingSession({seed:406,envCount:2,autoCurriculum:true});
  b.restore(snap);
  assert.equal(b.curiosityAudit.active,true);
  assert.equal(b.auditRole,'control');
  assert.equal(b.curiosityRewardMode,'observe');
  assert.equal(b.autoCurriculum,false);
  assert.equal(b.learnerExperienceSteps,654321);
  const legacy=structuredClone(snap);
  legacy.schema=7;
  for(const k of ['curiosityRewardMode','curiosityEpisodeHistory','curiosityBudgetResets','curiosityAudit','auditRole','autoCurriculum'])delete legacy[k];
  const c=new TrainingSession({seed:407,envCount:2,autoCurriculum:true});
  c.restore(legacy);
  assert.equal(c.curiosityRewardMode,'reward');
  assert.equal(c.curiosityAudit.active,false);
  assert.equal(c.auditRole,null);
});

test('curiosity audit UI exposes observe-only mode, matched branch controls, budget semantics and no-auto-promotion language',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(html,/budget remaining/i);
  assert.match(html,/Observe only \(reward 0\)/);
  assert.match(html,/same starting brain/i);
  assert.match(html,/no Champion promotion/i);
  assert.match(main,/startCuriosityAudit\(/);
  assert.match(main,/switchCuriosityAuditBranch\(/);
  assert.match(main,/intrinsicRewardSum,0|intrinsic reward exactly 0/i);
});

test('curiosity A/B gives both branches the same PPO schedule age at equal branch-local progress',()=>{
  const s=new TrainingSession({seed:408,envCount:2,autoCurriculum:true});
  s.totalSteps=250000;
  s.startCuriosityAudit({targetStepsPerBranch:100000,checkpointInterval:50000});
  const captureStep=()=>{
    const original=s.trainer.update.bind(s.trainer);
    let seen=null;
    s.trainer.update=(...args)=>{seen=args[3]?.trainingStep;return original(...args)};
    s.trainRollout(4);
    return seen;
  };
  const controlStep=captureStep();
  s.switchCuriosityAuditBranch('curiosity');
  const curiosityStep=captureStep();
  assert.equal(controlStep,250008);
  assert.equal(curiosityStep,250008);
});

test('curiosity A/B keeps ordinary milestones and episode return windows branch-isolated',()=>{
  const s=new TrainingSession({seed:409,envCount:1,autoCurriculum:true});
  s.nextMilestoneStep=1;
  const milestoneCount=s.milestones.size;
  s.startCuriosityAudit({targetStepsPerBranch:100000,checkpointInterval:50000});
  s.episodeHistory.push({totalReward:111,food:1,energy:1});
  s.trainRollout(2);
  assert.equal(s.milestones.size,milestoneCount);
  s.switchCuriosityAuditBranch('curiosity');
  assert.equal(s.episodeHistory.length,0);
  s.episodeHistory.push({totalReward:222,food:2,energy:1});
  s.switchCuriosityAuditBranch('control');
  assert.ok(s.episodeHistory.some(x=>x.totalReward===111));
  assert.ok(!s.episodeHistory.some(x=>x.totalReward===222));
  s.endCuriosityAudit();
  assert.ok(s.nextMilestoneStep>s.totalSteps);
});

test('PPO stability observatory exposes finite gradient, critic and parameter-movement telemetry without changing update semantics',()=>{
  const s=new TrainingSession({seed:510,envCount:2,autoCurriculum:false});
  s.nextValidationStep=Number.MAX_SAFE_INTEGER;
  const {metric}=s.trainRollout(16);
  for(const k of ['policyLoss','valueLoss','explainedVariance','gradientNormMean','gradientNormMax','gradientClipFraction','parameterDeltaL2','parameterRelativeDelta','parameterMaxAbsDelta'])assert.ok(Number.isFinite(metric[k]),k);
  assert.ok(metric.gradientNormMax>=0);
  assert.ok(metric.gradientClipFraction>=0&&metric.gradientClipFraction<=1);
  assert.ok(metric.parameterRelativeDelta>=0);
  assert.ok(['STABLE','WATCH','UPDATE SPIKE','GUARD'].includes(s.stabilitySummary().status));
});

test('schema-9 persists stability capture history and regression events while schema-8 migrates with empty telemetry',()=>{
  const a=new TrainingSession({seed:511,envCount:2,autoCurriculum:false});
  a.totalSteps=CONFIG.stability.captureIntervalSteps;
  a.nextValidationStep=Number.MAX_SAFE_INTEGER;
  a.trainRollout(8);
  a.stabilityEvents.push({atSteps:a.totalSteps,trigger:'test-observation',balancedDelta:-0.2});
  const snap=a.snapshot();
  assert.equal(snap.schema,9);
  assert.ok(snap.stabilityHistory.length>=1);
  const b=new TrainingSession({seed:512,envCount:2,autoCurriculum:false});
  b.restore(snap);
  assert.deepEqual(b.stabilityHistory,a.stabilityHistory);
  assert.deepEqual(b.stabilityEvents,a.stabilityEvents);
  const legacy=structuredClone(snap);legacy.schema=8;delete legacy.stabilityHistory;delete legacy.stabilityEvents;delete legacy.nextStabilityCaptureStep;
  const c=new TrainingSession({seed:513,envCount:2,autoCurriculum:false});
  c.restore(legacy);
  assert.deepEqual(c.stabilityHistory,[]);
  assert.deepEqual(c.stabilityEvents,[]);
  assert.ok(c.nextStabilityCaptureStep>c.totalSteps);
});

test('large validation regression creates an observational stability event without mutating policy weights',()=>{
  const s=new TrainingSession({seed:514,envCount:1,autoCurriculum:false});
  const before=JSON.stringify(s.model.serialize());
  const previous={stageResults:[
    {stage:0,name:'Motor Nursery',skillScore:.9},{stage:1,name:'Foraging',skillScore:.8},{stage:2,name:'Obstacle Avoidance',skillScore:.75},{stage:3,name:'Scarcity',skillScore:.7},
  ]};
  const current={categoryScores:{balanced:.58},stageResults:[
    {stage:0,name:'Motor Nursery',skillScore:.88},{stage:1,name:'Foraging',skillScore:.79},{stage:2,name:'Obstacle Avoidance',skillScore:.50},{stage:3,name:'Scarcity',skillScore:.69},
  ]};
  const result=s.noteValidationStability(current,previous,{validation:{categoryScores:{balanced:.76}}});
  assert.ok(result.event);
  assert.equal(result.event.trigger,'balanced+skill-drop');
  assert.equal(s.stabilityEvents.length,1);
  assert.equal(JSON.stringify(s.model.serialize()),before);
});

test('compact research UI collapses long secondary panels and applies iPhone-safe form sizing',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const css=await readFile(new URL('../styles.css',import.meta.url),'utf8');
  for(const id of ['stabilityCompactSummary','stabilityStatusVal','stabilityPolicyLossVal','stabilityValueLossVal','stabilityExplainedVal','stabilityKlVal','stabilityClipVal','stabilityGradVal','stabilityParamDeltaVal','stabilityValidationText','stabilityEventsText','curiosityCompactSummary','skillCompactSummary','resultsCompactSummary'])assert.match(html,new RegExp(`id="${id}"`));
  assert.match(html,/<details class="panel stabilityPanel collapsiblePanel"/);
  assert.match(html,/<details class="panel curiosityPanel collapsiblePanel"/);
  assert.match(html,/<details class="panel skillPanel collapsiblePanel"/);
  assert.match(html,/<details class="panel resultsPanel collapsiblePanel"/);
  assert.match(css,/select,input,textarea\{font-size:16px!important\}/);
  assert.match(css,/overflow-x:hidden/);
});


test('Cognitive Flow visualization exposes real recurrent memory, value, signal-flow and halo paths',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const neural=await readFile(new URL('../src/visualization/neuralRenderer.js',import.meta.url),'utf8');
  const world=await readFile(new URL('../src/visualization/worldRenderer.js',import.meta.url),'utf8');
  const curiosity=await readFile(new URL('../src/visualization/curiosityRenderer.js',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(html,/value="FLOW" selected>Cognitive Flow/);
  assert.match(neural,/p\.wh/);
  assert.match(neural,/p\.wv/);
  assert.match(neural,/drawSignalPulses/);
  assert.match(neural,/drawCognitiveHalo/);
  assert.match(neural,/drawDecisionBeam/);
  assert.match(world,/inputInfluence/);
  assert.match(world,/drawAgentCognitiveFx/);
  assert.match(curiosity,/PREDICTED → ACTUAL/);
  assert.match(curiosity,/drawPulses/);
  assert.match(main,/function cognitiveFx/);
});

test('v0.1.3.3 visual milestone does not change policy, PPO, curiosity, curriculum or save schema constants',async()=>{
  const config=await readFile(new URL('../src/config.js',import.meta.url),'utf8');
  const session=await readFile(new URL('../src/ai/session.js',import.meta.url),'utf8');
  assert.match(config,/obsSize: 10/);assert.match(config,/hiddenSize: 24/);assert.match(config,/actionSize: 7/);
  assert.match(config,/rewardScale: 0\.0048/);assert.match(config,/maxEpisodeBonus: 0\.25/);
  assert.match(config,/clip: 0\.12/);assert.match(config,/learningRate: 0\.00015/);assert.match(config,/maxGradNorm: 0\.5/);
  assert.match(config,/\[0\.10, 0\.15, 0\.20, 0\.55\]/);
  assert.match(session,/schema: 9/);
});


test('Prediction Echo and Attention Fields are real-data overlays with compact world controls',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const world=await readFile(new URL('../src/visualization/worldRenderer.js',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  const css=await readFile(new URL('../styles.css',import.meta.url),'utf8');
  assert.match(html,/id="worldFx"/);
  assert.match(html,/Attention \+ Echo/);
  assert.match(html,/Prediction Echo/);
  assert.match(world,/drawAttentionField/);
  assert.match(world,/drawPredictionEcho/);
  assert.match(world,/predicted \(gold hollow\) versus actual next observation \(cyan\)/i);
  assert.match(world,/Echo Lens: an agent-local sensor-space projection/);
  assert.match(main,/predictionEcho/);
  assert.match(main,/session\.lastCuriosity\.prediction/);
  assert.match(main,/worldRenderer\.overlayMode/);
  assert.match(css,/worldHeadControls/);
});

test('spawn-clearance hotfix keeps the accepted policy, PPO, curiosity, curriculum constants and schema',async()=>{
  const config=await readFile(new URL('../src/config.js',import.meta.url),'utf8');
  const session=await readFile(new URL('../src/ai/session.js',import.meta.url),'utf8');
  assert.match(config,/obsSize: 10/);assert.match(config,/hiddenSize: 24/);assert.match(config,/actionSize: 7/);
  assert.match(config,/rewardScale: 0\.0048/);assert.match(config,/maxEpisodeBonus: 0\.25/);
  assert.match(config,/clip: 0\.12/);assert.match(config,/learningRate: 0\.00015/);assert.match(config,/maxGradNorm: 0\.5/);
  assert.match(config,/\[0\.10, 0\.15, 0\.20, 0\.55\]/);
  assert.match(session,/schema: 9/);
});


test('Cognitive Observatory exposes five compact views and hidden views stop heavy rendering',async()=>{
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  const css=await readFile(new URL('../styles.css',import.meta.url),'utf8');
  for(const view of ['live','predict','memory','history','research'])assert.match(html,new RegExp(`data-view="${view}"`));
  for(const id of ['predictWorldCanvas','memoryCanvas','historyCanvas','memoryInspect','historyInspect'])assert.match(html,new RegExp(`id="${id}"`));
  assert.match(main,/activeObservatoryView === 'live'/);
  assert.match(main,/activeObservatoryView === 'predict'/);
  assert.match(main,/activeObservatoryView === 'memory'/);
  assert.match(main,/activeObservatoryView === 'history'/);
  assert.match(css,/\[data-observatory-view\]\[hidden\]\{display:none!important\}/);
});

test('Memory Constellation projection is deterministic, finite and sensitive to recurrent state',()=>{
  const a=new Float64Array(CONFIG.model.hiddenSize);for(let i=0;i<a.length;i++)a[i]=Math.sin(i*.37);
  const p1=projectHiddenState(a),p2=projectHiddenState(a);
  assert.deepEqual(p1,p2);
  assert.ok(Number.isFinite(p1.x)&&Number.isFinite(p1.y)&&Number.isFinite(p1.radius));
  assert.ok(Math.abs(p1.x)<=1&&Math.abs(p1.y)<=1&&p1.radius>=0);
  const b=new Float64Array(a);b[3]+=.25;const q=projectHiddenState(b);
  assert.notDeepEqual(q,p1);
});

test('History scene is reconstructed from real validation, Champion and lineage metadata without mutating session',()=>{
  const s=new TrainingSession({seed:601,envCount:1,autoCurriculum:false});
  s.runValidation();
  s.pinChampion('balanced');
  const before=JSON.stringify(s.snapshot());
  const scene=buildHistoryScene(s);
  assert.ok(scene.validations.length>=1);
  assert.ok(scene.champions.length>=1);
  assert.ok(scene.halls.length>=1);
  assert.equal(scene.current.lineage,s.learnerLineage.id);
  assert.equal(JSON.stringify(s.snapshot()),before);
});

test('Cognitive Observatory memory and history renderers use real recurrent/history data and bounded runtime buffers',async()=>{
  const memory=await readFile(new URL('../src/visualization/memoryRenderer.js',import.meta.url),'utf8');
  const history=await readFile(new URL('../src/visualization/historyRenderer.js',import.meta.url),'utf8');
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  assert.match(memory,/projectHiddenState/);
  assert.match(memory,/Experience ripples/i);
  assert.match(history,/validationHistory/);
  assert.match(history,/bestArchive/);
  assert.match(history,/hallOfFame/);
  assert.match(history,/frozenLearners/);
  assert.match(main,/MEMORY_MAX_POINTS = 320/);
  assert.match(main,/session\.hidden\?\.\[0\]/);
});


test('off-screen visualization performance hotfix remains intact in v0.1.4.0.2',async()=>{
  const main=await readFile(new URL('../src/app/main.js',import.meta.url),'utf8');
  const config=await readFile(new URL('../src/config.js',import.meta.url),'utf8');
  assert.match(main,/IntersectionObserver/);
  assert.match(main,/canvasVisible\(el\.world\)/);
  assert.match(main,/canvasVisible\(el\.brain\)/);
  assert.match(main,/canvasVisible\(el\.predictWorld\)/);
  assert.match(main,/canvasVisible\(el\.curiosityCanvas\)/);
  assert.match(main,/canvasVisible\(el\.memoryCanvas\)/);
  assert.match(main,/canvasVisible\(el\.historyCanvas\)/);
  assert.match(main,/canvasVisible\(el\.chart\)/);
  assert.match(config,/learnRenderHz: 12/);
  assert.match(config,/observeRenderHz: 30/);
  assert.match(config,/probeRenderHz: 20/);
  assert.match(config,/curiosityRenderHz: 8/);
});

test('Cognitive Flow reuses geometry objects while refreshing live weights and activations',async()=>{
  const neural=await readFile(new URL('../src/visualization/neuralRenderer.js',import.meta.url),'utf8');
  assert.match(neural,/buildNeuralGeometry/);
  assert.match(neural,/this\.activeEdges/);
  assert.match(neural,/edge\.w = weight/);
  assert.match(neural,/edge\.act = Math\.abs/);
  assert.match(neural,/activity < 0\.012/);
  assert.match(neural,/edges\.sort/);
});
