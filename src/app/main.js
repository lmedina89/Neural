import { ACTIONS, BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { TrainingSession } from '../ai/session.js';
import { RecurrentActorCritic } from '../ai/model.js';
import { evaluateModel } from '../evaluation/evaluator.js';
import { CURRICULUM } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { saveCheckpoint, loadCheckpoint } from '../storage/checkpoints.js';
import { WorldRenderer } from '../visualization/worldRenderer.js';
import { NeuralRenderer } from '../visualization/neuralRenderer.js';
import { ChartRenderer } from '../visualization/chartRenderer.js';

const $ = id => document.getElementById(id);
const el = {
  newBrain:$('newBrain'), learn:$('learnBtn'), observe:$('observeBtn'), probe:$('probeBtn'), pause:$('pauseBtn'),
  budget:$('budgetSelect'), save:$('saveBtn'), load:$('loadBtn'), test:$('testBtn'), compare:$('compareBtn'),
  mode:$('modeBadge'), status:$('statusText'), world:$('worldCanvas'), brain:$('brainCanvas'), chart:$('chartCanvas'),
  worldMeta:$('worldMeta'), probeTools:$('probeTools'), brainView:$('brainView'), inspect:$('inspectText'),
  steps:$('stepsVal'), episodes:$('episodesVal'), ret:$('returnVal'), food:$('foodVal'), entropy:$('entropyVal'), speed:$('speedVal'),
  paramCount:$('paramCount'), actionBars:$('actionBars'), energy:$('energyBar'), energyText:$('energyText'), curriculum:$('curriculumText'), rewardParts:$('rewardParts'), value:$('valueText'), results:$('resultsText')
};
$('buildTag').textContent=`v${VERSION} • ${BUILD_MARKER}`;

const budgets={eco:{rollout:8,delay:30},balanced:{rollout:18,delay:10},max:{rollout:36,delay:0}};
let mode='LEARN', paused=false, trainBusy=false, observeAccum=0, lastFrame=performance.now(), viewSeedIndex=0;
let session=new TrainingSession({seed:1337,envCount:CONFIG.runtime.trainEnvs});
let viewWorld=createViewWorld();
let viewObs=viewWorld.observe(), viewHidden=session.model.zeroHidden(), lastSnapshot=session.model.forward(viewObs,viewHidden);
const viewRng=new PRNG(0xabc123);
const worldRenderer=new WorldRenderer(el.world), neuralRenderer=new NeuralRenderer(el.brain), chartRenderer=new ChartRenderer(el.chart);

function createViewWorld(){return new World(domainSeed('observe:v1',viewSeedIndex++),session.curriculum.current())}
function setStatus(s){el.status.textContent=s}
function setMode(next){mode=next;paused=false;el.mode.textContent=next;el.probeTools.hidden=next!=='PROBE';for(const b of [el.learn,el.observe,el.probe])b.classList.remove('active');if(next==='LEARN')el.learn.classList.add('active');if(next==='OBSERVE')el.observe.classList.add('active');if(next==='PROBE')el.probe.classList.add('active');if(next!=='LEARN'){viewWorld=createViewWorld();viewObs=viewWorld.observe();viewHidden=session.model.zeroHidden();lastSnapshot=session.model.forward(viewObs,viewHidden)}setStatus(next==='LEARN'?'Training active. Rendering is throttled separately from learning.':next==='OBSERVE'?'Watching the current policy on an unseen procedural world.':'World frozen. Tap to manipulate stimuli and inspect the policy response.')}

function resetBrain(){session=new TrainingSession({seed:(Date.now()>>>0),envCount:CONFIG.runtime.trainEnvs});viewSeedIndex=0;viewWorld=createViewWorld();viewObs=viewWorld.observe();viewHidden=session.model.zeroHidden();lastSnapshot=session.model.forward(viewObs,viewHidden);setStatus('New untrained brain created.');updateUI()}

async function trainTick(){if(mode!=='LEARN'||paused){setTimeout(trainTick,40);return}if(trainBusy){setTimeout(trainTick,10);return}trainBusy=true;try{const b=budgets[el.budget.value]||budgets.balanced;session.trainRollout(b.rollout);viewWorld=session.envs[0];viewObs=session.obs[0];viewHidden=session.hidden[0];lastSnapshot=session.model.forward(viewObs,viewHidden);updateUI();setTimeout(trainTick,b.delay)}catch(err){console.error(err);paused=true;setStatus(`Training stopped: ${err.message}`);setTimeout(trainTick,100)}finally{trainBusy=false}}

function stepObserved(){if(viewWorld.done){viewWorld=createViewWorld();viewObs=viewWorld.observe();viewHidden=session.model.zeroHidden()}const act=session.model.act(viewObs,viewHidden,viewRng,false);const r=viewWorld.step(act.action);viewObs=r.obs;viewHidden=act.hidden;lastSnapshot=act.snapshot;if(r.done)setStatus(`Episode complete: food ${r.info.food}, return ${r.info.totalReward.toFixed(2)}.`)}

function updateProbe(){viewObs=viewWorld.observe();lastSnapshot=session.model.forward(viewObs,viewHidden);updateDecision()}

function frame(now){const dt=Math.min(100,now-lastFrame);lastFrame=now;if(mode==='OBSERVE'&&!paused){observeAccum+=dt;while(observeAccum>=55){stepObserved();observeAccum-=55}}worldRenderer.draw(viewWorld,mode+(paused?' • PAUSED':''));neuralRenderer.mode=el.brainView.value;neuralRenderer.draw(session.model,lastSnapshot);chartRenderer.draw(session.metrics);updateDecision();requestAnimationFrame(frame)}

function updateDecision(){const s=lastSnapshot;if(!s)return;el.actionBars.innerHTML='';for(let i=0;i<ACTIONS.length;i++){const row=document.createElement('div');row.className='actionRow';row.innerHTML=`<span>${ACTIONS[i]}</span><div class="barTrack"><div class="barFill" style="width:${(s.probs[i]*100).toFixed(1)}%"></div></div><b>${(s.probs[i]*100).toFixed(0)}%</b>`;el.actionBars.append(row)}el.value.textContent=`value ${s.value.toFixed(3)}`;const rp=viewWorld.lastRewardParts||{};el.rewardParts.textContent=`reward  food ${(rp.food||0).toFixed(3)}  approach ${(rp.approach||0).toFixed(3)}  energy ${(rp.energy||0).toFixed(3)}  wall ${(rp.wall||0).toFixed(3)}  hazard ${(rp.hazard||0).toFixed(3)}  death ${(rp.death||0).toFixed(3)}`;el.energy.value=viewWorld.agent.energy;el.energyText.textContent=`${Math.round(viewWorld.agent.energy*100)}%`;el.worldMeta.textContent=`seed ${viewWorld.seed}`}
function updateUI(){const m=session.metrics.at(-1);el.steps.textContent=session.totalSteps.toLocaleString();el.episodes.textContent=session.totalEpisodes.toLocaleString();el.ret.textContent=(m?.meanReturn??0).toFixed(2);el.food.textContent=(m?.meanFood??0).toFixed(2);el.entropy.textContent=m?.entropy?.toFixed(3)??'—';el.speed.textContent=m?.throughput?Math.round(m.throughput).toLocaleString():'—';el.paramCount.textContent=`${session.model.paramCount().toLocaleString()} params`;el.curriculum.textContent=session.curriculum.current().name;chartRenderer.draw(session.metrics)}

function evaluationText(name,r){return `${name}\nheld-out episodes: ${r.episodes}\nmean return: ${r.meanReturn.toFixed(3)}\nmean food: ${r.meanFood.toFixed(3)}\nsurvival: ${(r.survivalRate*100).toFixed(1)}%\nmean energy: ${r.meanEnergy.toFixed(3)}\nmean steps: ${r.meanSteps.toFixed(1)}`}
async function runUnseen(){paused=true;setStatus('Running held-out evaluation…');await new Promise(r=>setTimeout(r,20));const r=evaluateModel(session.model,session.curriculum.current(),{episodes:CONFIG.runtime.evalEpisodes,seedBase:'heldout:v1',deterministic:false});el.results.textContent=evaluationText(`UNSEEN TEST @ ${session.totalSteps.toLocaleString()} steps`,r);setStatus('Held-out evaluation complete. Training seeds were not used.');}
async function compareBrains(){paused=true;setStatus('Comparing historical brains on identical held-out seeds…');await new Promise(r=>setTimeout(r,20));const entries=[...session.milestones.values()].sort((a,b)=>a.savedAtSteps-b.savedAtSteps).slice(-8);if(!entries.length){el.results.textContent='No historical checkpoints yet.';return}let html='<table class="resultsTable"><thead><tr><th>brain</th><th>return</th><th>food</th><th>survival</th></tr></thead><tbody>';for(const cp of entries){const m=new RecurrentActorCritic(1);m.restore(cp.model);const r=evaluateModel(m,CURRICULUM[Math.min(session.curriculum.stage,CURRICULUM.length-1)],{episodes:10,seedBase:'heldout:v1',deterministic:false});html+=`<tr><td>${cp.savedAtSteps.toLocaleString()}</td><td>${r.meanReturn.toFixed(2)}</td><td>${r.meanFood.toFixed(2)}</td><td>${(r.survivalRate*100).toFixed(0)}%</td></tr>`}html+='</tbody></table>';el.results.innerHTML=html;setStatus('Checkpoint comparison complete. Every brain saw the same held-out seeds.')}

el.newBrain.addEventListener('click',()=>{if(confirm('Create a new untrained brain? Current unsaved progress will be replaced.'))resetBrain()});
el.learn.addEventListener('click',()=>setMode('LEARN'));el.observe.addEventListener('click',()=>setMode('OBSERVE'));el.probe.addEventListener('click',()=>setMode('PROBE'));el.pause.addEventListener('click',()=>{paused=!paused;el.pause.textContent=paused?'Resume':'Pause';setStatus(paused?'Paused. Neural state remains inspectable.':'Resumed.')});
el.brainView.addEventListener('change',()=>{neuralRenderer.mode=el.brainView.value});
el.save.addEventListener('click',async()=>{try{await saveCheckpoint(session.snapshot());setStatus('Checkpoint saved to IndexedDB.')}catch(e){setStatus(`Save failed: ${e.message}`)}});
el.load.addEventListener('click',async()=>{try{const cp=await loadCheckpoint();if(!cp){setStatus('No saved checkpoint found.');return}session.restore(cp);viewWorld=createViewWorld();viewObs=viewWorld.observe();viewHidden=session.model.zeroHidden();lastSnapshot=session.model.forward(viewObs,viewHidden);updateUI();setStatus('Checkpoint restored.')}catch(e){setStatus(`Load failed: ${e.message}`)}});
el.test.addEventListener('click',runUnseen);el.compare.addEventListener('click',compareBrains);
el.brain.addEventListener('pointerdown',e=>{const text=neuralRenderer.inspectAt(e.clientX,e.clientY);if(text)el.inspect.textContent=text});
el.world.addEventListener('pointerdown',e=>{if(mode!=='PROBE')return;const r=el.world.getBoundingClientRect(),x=Math.max(.03,Math.min(.97,(e.clientX-r.left)/r.width)),y=Math.max(.03,Math.min(.97,(e.clientY-r.top)/r.height));const tool=document.querySelector('input[name="probeTool"]:checked')?.value||'food';if(tool==='food'&&viewWorld.food.length){viewWorld.food[0].x=x;viewWorld.food[0].y=y;viewWorld.prevFoodDist=viewWorld.nearestFoodDistance()}else if(tool==='hazard'){if(viewWorld.hazards.length){viewWorld.hazards[0].x=x;viewWorld.hazards[0].y=y}else viewWorld.hazards.push({x,y,r:CONFIG.world.hazardRadius})}else{viewWorld.agent.x=x;viewWorld.agent.y=y;viewWorld.agent.vx=0;viewWorld.agent.vy=0}updateProbe();setStatus(`Probe moved ${tool}. Policy outputs updated without taking an action.`)});

updateUI();setMode('LEARN');requestAnimationFrame(frame);trainTick();
