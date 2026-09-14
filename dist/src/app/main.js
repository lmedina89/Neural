import { ACTIONS, BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { TrainingSession } from '../ai/session.js';
import { RecurrentActorCritic } from '../ai/model.js';
import { evaluateModel } from '../evaluation/evaluator.js';
import { CURRICULUM } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { saveCheckpoint, loadCheckpointRecord } from '../storage/checkpoints.js';
import { WorldRenderer } from '../visualization/worldRenderer.js';
import { NeuralRenderer } from '../visualization/neuralRenderer.js';
import { ChartRenderer } from '../visualization/chartRenderer.js';

const $ = id => document.getElementById(id);
const el = {
  newBrain:$('newBrain'), learn:$('learnBtn'), observe:$('observeBtn'), probe:$('probeBtn'), pause:$('pauseBtn'),
  budget:$('budgetSelect'), brainSource:$('brainSource'), bestOption:$('bestOption'), restoreBest:$('restoreBestBtn'),
  save:$('saveBtn'), loadManual:$('loadManualBtn'), loadAutosave:$('loadAutosaveBtn'), test:$('testBtn'), compare:$('compareBtn'),
  mode:$('modeBadge'), status:$('statusText'), world:$('worldCanvas'), brain:$('brainCanvas'), chart:$('chartCanvas'),
  worldMeta:$('worldMeta'), probeTools:$('probeTools'), brainView:$('brainView'), viewBrainBadge:$('viewBrainBadge'), inspect:$('inspectText'),
  steps:$('stepsVal'), episodes:$('episodesVal'), ret:$('returnVal'), food:$('foodVal'), entropy:$('entropyVal'), speed:$('speedVal'),
  paramCount:$('paramCount'), actionBars:$('actionBars'), energy:$('energyBar'), energyText:$('energyText'), curriculum:$('curriculumText'),
  rewardParts:$('rewardParts'), value:$('valueText'), results:$('resultsText'), latestValidation:$('latestValidation'), bestValidation:$('bestValidation'),
  manualSlot:$('manualSlotInfo'), autosaveSlot:$('autosaveSlotInfo')
};
$('buildTag').textContent=`v${VERSION} • ${BUILD_MARKER}`;

const budgets={eco:{rollout:8,delay:30},balanced:{rollout:18,delay:10},max:{rollout:36,delay:0}};
let mode='LEARN', paused=false, trainBusy=false, observeAccum=0, lastFrame=performance.now(), viewSeedIndex=0;
let session=new TrainingSession({seed:1337,envCount:CONFIG.runtime.trainEnvs});
let bestModelCache=null, bestModelCacheSteps=null;
let viewWorld=createViewWorld();
let viewObs=viewWorld.observe(), viewHidden=session.model.zeroHidden(), lastSnapshot=session.model.forward(viewObs,viewHidden);
const viewRng=new PRNG(0xabc123);
const worldRenderer=new WorldRenderer(el.world), neuralRenderer=new NeuralRenderer(el.brain), chartRenderer=new ChartRenderer(el.chart);

function createViewWorld(){return new World(domainSeed('observe:v2',viewSeedIndex++),session.curriculum.current())}
function setStatus(s){el.status.textContent=s}
function formatSavedTime(ms){
  if(!Number.isFinite(ms))return 'unknown time';
  try{return new Date(ms).toLocaleString()}catch{return 'unknown time'}
}
function describeCheckpointRecord(record){
  if(!record?.snapshot)return 'Empty';
  const steps=Number(record.snapshot.totalSteps||0).toLocaleString();
  const episodes=Number(record.snapshot.totalEpisodes||0).toLocaleString();
  const schema=record.snapshot.schema??'?';
  return `${steps} steps • ${episodes} episodes • schema ${schema} • ${formatSavedTime(record.savedAt)}`;
}
async function refreshSaveSlots(){
  try{
    const [manual,autosave]=await Promise.all([loadCheckpointRecord('latest'),loadCheckpointRecord('autosave')]);
    el.manualSlot.textContent=describeCheckpointRecord(manual);
    el.autosaveSlot.textContent=describeCheckpointRecord(autosave);
    el.loadManual.disabled=!manual;
    el.loadAutosave.disabled=!autosave;
    return {manual,autosave};
  }catch(err){
    el.manualSlot.textContent=`Storage error: ${err.message}`;
    el.autosaveSlot.textContent='Unavailable';
    el.loadManual.disabled=true;el.loadAutosave.disabled=true;
    return {manual:null,autosave:null,error:err};
  }
}
async function restoreCheckpointRecord(record,label){
  const cp=record?.snapshot;
  if(!cp){setStatus(`No ${label.toLowerCase()} found.`);return false}
  paused=true;el.pause.textContent='Resume';
  session.restore(cp);bestModelCache=null;bestModelCacheSteps=null;viewSeedIndex=0;el.brainSource.value='latest';
  resetViewState();updateUI();await refreshSaveSlots();
  const migration=cp.schema===1?' Legacy v0.1.0 checkpoint migrated in memory; its exact loaded policy is preserved as a migration milestone.':'';
  setStatus(`${label} restored at ${session.totalSteps.toLocaleString()} steps. Training is PAUSED so you can verify it before pressing Resume/Learn.${migration}`);
  return true;
}
async function saveManualSafely(){
  try{
    const existing=await loadCheckpointRecord('latest');
    const storedSteps=Number(existing?.snapshot?.totalSteps||0);
    if(existing&&storedSteps>session.totalSteps){
      setStatus(`Manual save protected: stored brain has ${storedSteps.toLocaleString()} steps, current brain has ${session.totalSteps.toLocaleString()}. Load Manual first; this lower-step brain was NOT allowed to overwrite it.`);
      await refreshSaveSlots();
      return;
    }
    await saveCheckpoint(session.snapshot(),'latest');
    await refreshSaveSlots();
    setStatus(`Manual checkpoint saved at ${session.totalSteps.toLocaleString()} steps. Protected-best data included.`);
  }catch(e){setStatus(`Save failed: ${e.message}`)}
}
function selectedViewModel(){
  if(mode==='LEARN'||el.brainSource.value!=='best'||!session.bestBrain?.model)return session.model;
  if(!bestModelCache||bestModelCacheSteps!==session.bestBrain.savedAtSteps){
    bestModelCache=new RecurrentActorCritic(1);bestModelCache.restore(session.bestBrain.model);bestModelCacheSteps=session.bestBrain.savedAtSteps;
  }
  return bestModelCache;
}
function selectedSourceLabel(){return mode!=='LEARN'&&el.brainSource.value==='best'&&session.bestBrain?'BEST':'LATEST'}
function resetViewState(){
  const model=selectedViewModel();
  viewWorld=createViewWorld();viewObs=viewWorld.observe();viewHidden=model.zeroHidden();lastSnapshot=model.forward(viewObs,viewHidden);observeAccum=0;
}
function setMode(next){
  mode=next;paused=false;el.pause.textContent='Pause';el.mode.textContent=next;el.probeTools.hidden=next!=='PROBE';
  for(const b of [el.learn,el.observe,el.probe])b.classList.remove('active');
  if(next==='LEARN')el.learn.classList.add('active');if(next==='OBSERVE')el.observe.classList.add('active');if(next==='PROBE')el.probe.classList.add('active');
  if(next!=='LEARN')resetViewState();
  el.viewBrainBadge.textContent=selectedSourceLabel();
  setStatus(next==='LEARN'?'Training active. Automatic validation protects the best generalization brain.':next==='OBSERVE'?`Watching the ${selectedSourceLabel().toLowerCase()} policy in a separate procedural world.`:'World frozen. Tap to manipulate stimuli and inspect the selected policy response.');
}

function resetBrain(){
  session=new TrainingSession({seed:(Date.now()>>>0),envCount:CONFIG.runtime.trainEnvs});bestModelCache=null;bestModelCacheSteps=null;viewSeedIndex=0;el.brainSource.value='latest';resetViewState();syncBestControls();setStatus('New untrained brain created.');updateUI();
}

async function trainTick(){
  if(mode!=='LEARN'||paused){setTimeout(trainTick,40);return}
  if(trainBusy){setTimeout(trainTick,10);return}
  trainBusy=true;
  try{
    const b=budgets[el.budget.value]||budgets.balanced;
    const result=session.trainRollout(b.rollout);
    viewWorld=session.envs[0];viewObs=session.obs[0];viewHidden=session.hidden[0];lastSnapshot=session.model.forward(viewObs,viewHidden);
    if(result.validation){
      bestModelCache=null;bestModelCacheSteps=null;syncBestControls();
      const v=result.validation;
      if(v.regression)setStatus(`Validation regression detected: latest ${v.validation.score.toFixed(2)} vs protected best ${v.bestScore.toFixed(2)}. Best brain preserved.`);
      else if(v.improved)setStatus(`New protected best brain at ${session.bestBrain.savedAtSteps.toLocaleString()} steps • validation ${v.validation.score.toFixed(2)}.`);
      else setStatus(`Validation complete: ${v.validation.score.toFixed(2)} • protected best ${v.bestScore.toFixed(2)}.`);
      try{await saveCheckpoint(session.snapshot(),'autosave');await refreshSaveSlots()}catch(saveErr){console.warn('Validation autosave failed',saveErr);setStatus(`${el.status.textContent} Autosave failed: ${saveErr.message}`)}
    } else if(result.curriculumEvent){
      const e=result.curriculumEvent;setStatus(`Curriculum ${e.reason}: ${CURRICULUM[e.from].name} → ${CURRICULUM[e.to].name}.`);
    }
    updateUI();setTimeout(trainTick,b.delay);
  }catch(err){console.error(err);paused=true;el.pause.textContent='Resume';setStatus(`Training stopped: ${err.message}`);setTimeout(trainTick,100)}finally{trainBusy=false}
}

function stepObserved(){
  const model=selectedViewModel();
  if(viewWorld.done){resetViewState()}
  const act=model.act(viewObs,viewHidden,viewRng,false);const r=viewWorld.step(act.action);viewObs=r.obs;viewHidden=act.hidden;lastSnapshot=act.snapshot;
  if(r.done)setStatus(`Episode complete (${selectedSourceLabel()}): food ${r.info.food}, return ${r.info.totalReward.toFixed(2)}.`)
}

function updateProbe(){const model=selectedViewModel();viewObs=viewWorld.observe();lastSnapshot=model.forward(viewObs,viewHidden);updateDecision()}

function frame(now){
  const dt=Math.min(100,now-lastFrame);lastFrame=now;
  if(mode==='OBSERVE'&&!paused){observeAccum+=dt;while(observeAccum>=55){stepObserved();observeAccum-=55}}
  const model=selectedViewModel();
  worldRenderer.draw(viewWorld,mode+(paused?' • PAUSED':''));neuralRenderer.mode=el.brainView.value;neuralRenderer.draw(model,lastSnapshot);chartRenderer.draw(session.metrics);updateDecision();requestAnimationFrame(frame)
}

function updateDecision(){
  const s=lastSnapshot;if(!s)return;el.actionBars.innerHTML='';
  for(let i=0;i<ACTIONS.length;i++){const row=document.createElement('div');row.className='actionRow';row.innerHTML=`<span>${ACTIONS[i]}</span><div class="barTrack"><div class="barFill" style="width:${(s.probs[i]*100).toFixed(1)}%"></div></div><b>${(s.probs[i]*100).toFixed(0)}%</b>`;el.actionBars.append(row)}
  el.value.textContent=`value ${s.value.toFixed(3)}`;const rp=viewWorld.lastRewardParts||{};
  el.rewardParts.textContent=`reward  food ${(rp.food||0).toFixed(3)}  approach ${(rp.approach||0).toFixed(3)}  energy ${(rp.energy||0).toFixed(3)}  wall ${(rp.wall||0).toFixed(3)}  hazard ${(rp.hazard||0).toFixed(3)}  death ${(rp.death||0).toFixed(3)}`;
  el.energy.value=viewWorld.agent.energy;el.energyText.textContent=`${Math.round(viewWorld.agent.energy*100)}%`;el.worldMeta.textContent=`seed ${viewWorld.seed}`;el.viewBrainBadge.textContent=selectedSourceLabel();
}

function updateUI(){
  const m=session.metrics.at(-1);el.steps.textContent=session.totalSteps.toLocaleString();el.episodes.textContent=session.totalEpisodes.toLocaleString();el.ret.textContent=(m?.meanReturn??0).toFixed(2);el.food.textContent=(m?.meanFood??0).toFixed(2);el.entropy.textContent=m?.entropy?.toFixed(3)??'—';el.speed.textContent=m?.throughput?Math.round(m.throughput).toLocaleString():'—';el.paramCount.textContent=`${session.model.paramCount().toLocaleString()} params`;el.curriculum.textContent=session.curriculum.current().name;
  const lv=session.validationHistory.at(-1);el.latestValidation.textContent=lv?`${lv.validation.score.toFixed(2)} @ ${lv.steps.toLocaleString()}${lv.regression?' REGRESSION':''}`:'—';
  el.bestValidation.textContent=session.bestBrain?`${session.bestBrain.validation.score.toFixed(2)} @ ${session.bestBrain.savedAtSteps.toLocaleString()}`:'—';
  chartRenderer.draw(session.metrics);syncBestControls();
}

function syncBestControls(){
  const hasBest=Boolean(session.bestBrain?.model);el.bestOption.disabled=!hasBest;el.restoreBest.disabled=!hasBest;
  el.bestOption.textContent=hasBest?`Best @ ${session.bestBrain.savedAtSteps.toLocaleString()}`:'Best (not validated yet)';
  if(!hasBest&&el.brainSource.value==='best')el.brainSource.value='latest';
}

function evaluationText(name,r){return `${name}\nheld-out episodes: ${r.episodes}\nmean return: ${r.meanReturn.toFixed(3)}\nmean food: ${r.meanFood.toFixed(3)}\nsurvival: ${(r.survivalRate*100).toFixed(1)}%\nmean energy: ${r.meanEnergy.toFixed(3)}\nmean steps: ${r.meanSteps.toFixed(1)}`}

async function runUnseen(){
  paused=true;el.pause.textContent='Resume';setStatus('Running held-out evaluation…');await yieldUI();
  const r=evaluateModel(session.model,session.curriculum.current(),{episodes:CONFIG.runtime.evalEpisodes,seedBase:'heldout:v1',deterministic:false});
  let text=evaluationText(`UNSEEN TEST • LATEST @ ${session.totalSteps.toLocaleString()} steps`,r);
  if(session.bestBrain?.model){
    await yieldUI();const best=new RecurrentActorCritic(1);best.restore(session.bestBrain.model);const br=evaluateModel(best,session.curriculum.current(),{episodes:CONFIG.runtime.evalEpisodes,seedBase:'heldout:v1',deterministic:false});
    text+=`\n\n${evaluationText(`PROTECTED BEST @ ${session.bestBrain.savedAtSteps.toLocaleString()} steps`,br)}`;
  }
  el.results.textContent=text;setStatus('Held-out evaluation complete. Training and validation seed domains were not used.');
}

async function compareBrains(){
  paused=true;el.pause.textContent='Resume';setStatus('Comparing historical brains on identical held-out seeds…');await yieldUI();
  const historical=[...session.milestones.values()].sort((a,b)=>a.savedAtSteps-b.savedAtSteps);
  const selected=[];const selectedSteps=new Set();
  const addHistorical=(cp)=>{if(cp&&!selectedSteps.has(cp.savedAtSteps)){selectedSteps.add(cp.savedAtSteps);selected.push(cp)}};
  addHistorical(historical[0]);
  for(const target of [100000,500000,1000000]) addHistorical(historical.find(cp=>cp.label===target));
  for(const cp of historical.slice(-8)) addHistorical(cp);
  selected.sort((a,b)=>a.savedAtSteps-b.savedAtSteps);
  const raw=selected.map(cp=>({savedAtSteps:cp.savedAtSteps,model:cp.model,kind:'milestone'}));
  raw.push({savedAtSteps:session.totalSteps,model:session.model.serialize(),kind:'latest'});
  if(session.bestBrain?.model)raw.push({savedAtSteps:session.bestBrain.savedAtSteps,model:session.bestBrain.model,kind:'best'});
  const entries=[];const seen=new Set();
  for(const e of raw){const key=`${e.savedAtSteps}:${e.kind==='latest'?'latest':e.kind==='best'?'best':'milestone'}`;if(!seen.has(key)){seen.add(key);entries.push(e)}}
  if(!entries.length){el.results.textContent='No historical checkpoints yet.';return}
  const rows=[];
  for(const cp of entries){
    const m=new RecurrentActorCritic(1);m.restore(cp.model);const r=evaluateModel(m,CURRICULUM[Math.min(session.curriculum.stage,CURRICULUM.length-1)],{episodes:CONFIG.runtime.compareEpisodes,seedBase:'heldout:v1',deterministic:false});rows.push({cp,r});await yieldUI();
  }
  let html='<table class="resultsTable"><thead><tr><th>brain</th><th>return</th><th>food</th><th>survival</th></tr></thead><tbody>';
  for(const {cp,r} of rows){const isBest=cp.kind==='best',isLatest=cp.kind==='latest';const cls=isBest?'bestRow':'';const label=`${cp.savedAtSteps.toLocaleString()}${isBest?' ★ BEST':''}${isLatest?' LATEST':''}`;html+=`<tr class="${cls}"><td>${label}</td><td>${r.meanReturn.toFixed(2)}</td><td>${r.meanFood.toFixed(2)}</td><td>${(r.survivalRate*100).toFixed(0)}%</td></tr>`}
  html+='</tbody></table>';el.results.innerHTML=html;setStatus(`Checkpoint comparison complete using ${CONFIG.runtime.compareEpisodes} identical held-out episodes per brain.`);
}

function yieldUI(){return new Promise(resolve=>setTimeout(resolve,20))}

el.newBrain.addEventListener('click',()=>{if(confirm('Create a new untrained brain? Current unsaved progress will be replaced.'))resetBrain()});
el.learn.addEventListener('click',()=>setMode('LEARN'));el.observe.addEventListener('click',()=>setMode('OBSERVE'));el.probe.addEventListener('click',()=>setMode('PROBE'));
el.pause.addEventListener('click',()=>{paused=!paused;el.pause.textContent=paused?'Resume':'Pause';setStatus(paused?'Paused. Neural state remains inspectable.':'Resumed.')});
el.brainView.addEventListener('change',()=>{neuralRenderer.mode=el.brainView.value});
el.brainSource.addEventListener('change',()=>{if(el.brainSource.value==='best'&&!session.bestBrain){el.brainSource.value='latest';return}if(mode!=='LEARN')resetViewState();el.viewBrainBadge.textContent=selectedSourceLabel();setStatus(mode==='LEARN'?'View Brain selection applies in Observe/Probe; Learn always shows/trains Latest.':`Now inspecting ${selectedSourceLabel().toLowerCase()} brain.`)});
el.save.addEventListener('click',saveManualSafely);
el.loadManual.addEventListener('click',async()=>{try{await restoreCheckpointRecord(await loadCheckpointRecord('latest'),'Manual Save')}catch(e){setStatus(`Manual load failed: ${e.message}`)}});
el.loadAutosave.addEventListener('click',async()=>{try{await restoreCheckpointRecord(await loadCheckpointRecord('autosave'),'Validation Autosave')}catch(e){setStatus(`Autosave load failed: ${e.message}`)}});
el.restoreBest.addEventListener('click',()=>{if(!session.bestBrain)return;const source=session.bestBrain.savedAtSteps;if(!confirm(`Restore the trainable policy + optimizer from the protected best at ${source.toLocaleString()} steps? Total experience steps will remain ${session.totalSteps.toLocaleString()} for an honest training history.`))return;session.restoreBest();bestModelCache=null;bestModelCacheSteps=null;el.brainSource.value='latest';resetViewState();updateUI();setStatus(`Latest policy restored from protected best @ ${source.toLocaleString()} steps. Training can resume from that brain.`)});
el.test.addEventListener('click',runUnseen);el.compare.addEventListener('click',compareBrains);
el.brain.addEventListener('pointerdown',e=>{const text=neuralRenderer.inspectAt(e.clientX,e.clientY);if(text)el.inspect.textContent=text});
el.world.addEventListener('pointerdown',e=>{if(mode!=='PROBE')return;const r=el.world.getBoundingClientRect(),x=Math.max(.03,Math.min(.97,(e.clientX-r.left)/r.width)),y=Math.max(.03,Math.min(.97,(e.clientY-r.top)/r.height));const tool=document.querySelector('input[name="probeTool"]:checked')?.value||'food';if(tool==='food'&&viewWorld.food.length){viewWorld.food[0].x=x;viewWorld.food[0].y=y;viewWorld.prevFoodDist=viewWorld.nearestFoodDistance()}else if(tool==='hazard'){if(viewWorld.hazards.length){viewWorld.hazards[0].x=x;viewWorld.hazards[0].y=y}else viewWorld.hazards.push({x,y,r:CONFIG.world.hazardRadius})}else{viewWorld.agent.x=x;viewWorld.agent.y=y;viewWorld.agent.vx=0;viewWorld.agent.vy=0}updateProbe();setStatus(`Probe moved ${tool}. ${selectedSourceLabel()} policy outputs updated without taking an action.`)});

updateUI();setMode('LEARN');requestAnimationFrame(frame);refreshSaveSlots().then(({manual})=>{if(manual?.snapshot?.totalSteps>session.totalSteps)setStatus(`Stored Manual Save detected at ${Number(manual.snapshot.totalSteps).toLocaleString()} steps. It is protected from lower-step overwrite; use Load Manual to recover it.`)});trainTick();
