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
  newBrain: $('newBrain'), learn: $('learnBtn'), observe: $('observeBtn'), probe: $('probeBtn'), pause: $('pauseBtn'),
  budget: $('budgetSelect'), brainSource: $('brainSource'), bestOption: $('bestOption'), overallOption: $('overallOption'),
  foragerOption: $('foragerOption'), survivorOption: $('survivorOption'), efficiencyOption: $('efficiencyOption'), restoreBest: $('restoreBestBtn'),
  save: $('saveBtn'), loadManual: $('loadManualBtn'), loadAutosave: $('loadAutosaveBtn'), test: $('testBtn'), compare: $('compareBtn'),
  mode: $('modeBadge'), status: $('statusText'), world: $('worldCanvas'), brain: $('brainCanvas'), chart: $('chartCanvas'),
  worldMeta: $('worldMeta'), probeTools: $('probeTools'), brainView: $('brainView'), viewBrainBadge: $('viewBrainBadge'), inspect: $('inspectText'),
  steps: $('stepsVal'), episodes: $('episodesVal'), ret: $('returnVal'), food: $('foodVal'), entropy: $('entropyVal'), speed: $('speedVal'),
  lr: $('lrVal'), kl: $('klVal'), epochs: $('epochsVal'), paramCount: $('paramCount'), actionBars: $('actionBars'), energy: $('energyBar'),
  energyText: $('energyText'), curriculum: $('curriculumText'), rewardParts: $('rewardParts'), value: $('valueText'), results: $('resultsText'),
  latestValidation: $('latestValidation'), bestValidation: $('bestValidation'), retentionAlert: $('retentionAlert'), skillRetention: $('skillRetention'),
  manualSlot: $('manualSlotInfo'), autosaveSlot: $('autosaveSlotInfo'),
};
$('buildTag').textContent = `v${VERSION} • ${BUILD_MARKER}`;

const budgets = { eco: { rollout: 8, delay: 30 }, balanced: { rollout: 18, delay: 10 }, max: { rollout: 36, delay: 0 } };
const archiveLabels = { balanced: 'BALANCED', overall: 'OVERALL', forager: 'FORAGER', survivor: 'SURVIVOR', efficiency: 'EFFICIENCY' };
let mode = 'LEARN', paused = false, trainBusy = false, observeAccum = 0, lastFrame = performance.now(), viewSeedIndex = 0;
let session = new TrainingSession({ seed: 1337, envCount: CONFIG.runtime.trainEnvs });
let archiveModelCache = null, archiveModelCacheKey = null;
let viewWorld = createViewWorld();
let viewObs = viewWorld.observe(), viewHidden = session.model.zeroHidden(), lastSnapshot = session.model.forward(viewObs, viewHidden);
const viewRng = new PRNG(0xabc123);
const worldRenderer = new WorldRenderer(el.world), neuralRenderer = new NeuralRenderer(el.brain), chartRenderer = new ChartRenderer(el.chart);

function createViewWorld() { return new World(domainSeed('observe:v2', viewSeedIndex++), session.curriculum.current()); }
function setStatus(s) { el.status.textContent = s; }
function formatSavedTime(ms) {
  if (!Number.isFinite(ms)) return 'unknown time';
  try { return new Date(ms).toLocaleString(); } catch { return 'unknown time'; }
}
function describeCheckpointRecord(record) {
  if (!record?.snapshot) return 'Empty';
  const steps = Number(record.snapshot.totalSteps || 0).toLocaleString();
  const episodes = Number(record.snapshot.totalEpisodes || 0).toLocaleString();
  const schema = record.snapshot.schema ?? '?';
  return `${steps} steps • ${episodes} episodes • schema ${schema} • ${formatSavedTime(record.savedAt)}`;
}
async function refreshSaveSlots() {
  try {
    const [manual, autosave] = await Promise.all([loadCheckpointRecord('latest'), loadCheckpointRecord('autosave')]);
    el.manualSlot.textContent = describeCheckpointRecord(manual);
    el.autosaveSlot.textContent = describeCheckpointRecord(autosave);
    el.loadManual.disabled = !manual;
    el.loadAutosave.disabled = !autosave;
    return { manual, autosave };
  } catch (err) {
    el.manualSlot.textContent = `Storage error: ${err.message}`;
    el.autosaveSlot.textContent = 'Unavailable';
    el.loadManual.disabled = true;
    el.loadAutosave.disabled = true;
    return { manual: null, autosave: null, error: err };
  }
}
async function restoreCheckpointRecord(record, label) {
  const cp = record?.snapshot;
  if (!cp) { setStatus(`No ${label.toLowerCase()} found.`); return false; }
  paused = true;
  el.pause.textContent = 'Resume';
  session.restore(cp);
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  updateUI();
  await refreshSaveSlots();
  const migration = cp.schema < 3
    ? ' Legacy checkpoint loaded safely. Its protected Best is preserved and will be re-evaluated on the new all-skills protocol before training advances.'
    : '';
  setStatus(`${label} restored at ${session.totalSteps.toLocaleString()} steps. Training is PAUSED so you can verify it before pressing Resume/Learn.${migration}`);
  return true;
}
async function saveManualSafely() {
  try {
    const existing = await loadCheckpointRecord('latest');
    const storedSteps = Number(existing?.snapshot?.totalSteps || 0);
    if (existing && storedSteps > session.totalSteps) {
      setStatus(`Manual save protected: stored brain has ${storedSteps.toLocaleString()} steps, current brain has ${session.totalSteps.toLocaleString()}. Load Manual first; this lower-step brain was NOT allowed to overwrite it.`);
      await refreshSaveSlots();
      return;
    }
    await saveCheckpoint(session.snapshot(), 'latest');
    await refreshSaveSlots();
    setStatus(`Manual checkpoint saved at ${session.totalSteps.toLocaleString()} steps. Skill archive and protected specialists included.`);
  } catch (e) { setStatus(`Save failed: ${e.message}`); }
}

function selectedArchiveCategory() {
  const value = el.brainSource.value;
  return value === 'latest' ? null : value;
}
function selectedArchiveBrain() {
  const category = selectedArchiveCategory();
  return category ? session.getArchiveBrain(category) : null;
}
function selectedViewModel() {
  const category = selectedArchiveCategory();
  const brain = selectedArchiveBrain();
  if (mode === 'LEARN' || !category || !brain?.model) return session.model;
  const key = `${category}:${brain.savedAtSteps}`;
  if (!archiveModelCache || archiveModelCacheKey !== key) {
    archiveModelCache = new RecurrentActorCritic(1);
    archiveModelCache.restore(brain.model);
    archiveModelCacheKey = key;
  }
  return archiveModelCache;
}
function selectedSourceLabel() {
  const category = selectedArchiveCategory();
  if (mode !== 'LEARN' && category && selectedArchiveBrain()) return archiveLabels[category] || category.toUpperCase();
  return 'LATEST';
}
function resetViewState() {
  const model = selectedViewModel();
  viewWorld = createViewWorld();
  viewObs = viewWorld.observe();
  viewHidden = model.zeroHidden();
  lastSnapshot = model.forward(viewObs, viewHidden);
  observeAccum = 0;
}
function setMode(next) {
  mode = next;
  paused = false;
  el.pause.textContent = 'Pause';
  el.mode.textContent = next;
  el.probeTools.hidden = next !== 'PROBE';
  for (const b of [el.learn, el.observe, el.probe]) b.classList.remove('active');
  if (next === 'LEARN') el.learn.classList.add('active');
  if (next === 'OBSERVE') el.observe.classList.add('active');
  if (next === 'PROBE') el.probe.classList.add('active');
  if (next !== 'LEARN') resetViewState();
  el.viewBrainBadge.textContent = selectedSourceLabel();
  setStatus(next === 'LEARN'
    ? 'Training active. Guarded PPO + all-skills validation protect learning and skill retention.'
    : next === 'OBSERVE'
      ? `Watching the ${selectedSourceLabel().toLowerCase()} policy in a separate procedural world.`
      : 'World frozen. Tap to manipulate stimuli and inspect the selected policy response.');
}
function resetBrain() {
  session = new TrainingSession({ seed: (Date.now() >>> 0), envCount: CONFIG.runtime.trainEnvs });
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  syncArchiveControls();
  setStatus('New untrained brain created.');
  updateUI();
}

async function trainTick() {
  if (mode !== 'LEARN' || paused) { setTimeout(trainTick, 40); return; }
  if (trainBusy) { setTimeout(trainTick, 10); return; }
  trainBusy = true;
  try {
    const b = budgets[el.budget.value] || budgets.balanced;
    const result = session.trainRollout(b.rollout);
    viewWorld = session.envs[0];
    viewObs = session.obs[0];
    viewHidden = session.hidden[0];
    lastSnapshot = session.model.forward(viewObs, viewHidden);

    if (result.metric.updateRejected) {
      setStatus(`PPO safety rejected an oversized update (KL ${result.metric.maxEpochKL.toFixed(3)}). Weights were rolled back; learning rate reduced to ${result.metric.learningRate.toExponential(2)}.`);
    } else if (result.validation) {
      archiveModelCache = null;
      archiveModelCacheKey = null;
      syncArchiveControls();
      const v = result.validation;
      const forgetting = v.forgetting?.length ? ` • forgetting: ${v.forgetting.map(x => x.name).join(', ')}` : '';
      if (v.autoRollback) setStatus(`Skill-retention guard restored Best Balanced @ ${v.autoRollback.sourceSteps.toLocaleString()} and reduced LR after ${v.regression ? 'validation regression' : 'catastrophic forgetting'}.${forgetting}`);
      else if (v.regression) setStatus(`Retention regression: latest balanced ${(v.validation.score * 100).toFixed(1)}% vs protected ${(v.bestScore * 100).toFixed(1)}%. Archive preserved.${forgetting}`);
      else if (v.improved) setStatus(`New Best Balanced @ ${session.bestBrain.savedAtSteps.toLocaleString()} • ${(v.validation.score * 100).toFixed(1)}% skill balance.${forgetting}`);
      else setStatus(`All-skills validation complete: balanced ${(v.validation.score * 100).toFixed(1)}% • protected ${(v.bestScore * 100).toFixed(1)}%.${forgetting}`);
      try {
        await saveCheckpoint(session.snapshot(), 'autosave');
        await refreshSaveSlots();
      } catch (saveErr) {
        console.warn('Validation autosave failed', saveErr);
        setStatus(`${el.status.textContent} Autosave failed: ${saveErr.message}`);
      }
    } else if (result.curriculumEvent) {
      const e = result.curriculumEvent;
      if (e.reason === 'promotion-blocked') setStatus(`Curriculum promotion held at ${CURRICULUM[e.from].name}: ${e.gateReason || 'earlier skills need retention work'}.`);
      else setStatus(`Curriculum ${e.reason}: ${CURRICULUM[e.from].name} → ${CURRICULUM[e.to].name}.`);
    } else if (result.metric.earlyStopped && result.metric.maxEpochKL > CONFIG.ppo.targetKL) {
      setStatus(`PPO epoch stopped early at KL ${result.metric.maxEpochKL.toFixed(3)} to protect the current policy.`);
    }
    updateUI();
    setTimeout(trainTick, b.delay);
  } catch (err) {
    console.error(err);
    paused = true;
    el.pause.textContent = 'Resume';
    setStatus(`Training stopped: ${err.message}`);
    setTimeout(trainTick, 100);
  } finally { trainBusy = false; }
}

function stepObserved() {
  const model = selectedViewModel();
  if (viewWorld.done) resetViewState();
  const act = model.act(viewObs, viewHidden, viewRng, false);
  const r = viewWorld.step(act.action);
  viewObs = r.obs;
  viewHidden = act.hidden;
  lastSnapshot = act.snapshot;
  if (r.done) setStatus(`Episode complete (${selectedSourceLabel()}): food ${r.info.food}, return ${r.info.totalReward.toFixed(2)}.`);
}
function updateProbe() {
  const model = selectedViewModel();
  viewObs = viewWorld.observe();
  lastSnapshot = model.forward(viewObs, viewHidden);
  updateDecision();
}
function frame(now) {
  const dt = Math.min(100, now - lastFrame);
  lastFrame = now;
  if (mode === 'OBSERVE' && !paused) {
    observeAccum += dt;
    while (observeAccum >= 55) { stepObserved(); observeAccum -= 55; }
  }
  const model = selectedViewModel();
  worldRenderer.draw(viewWorld, mode + (paused ? ' • PAUSED' : ''));
  neuralRenderer.mode = el.brainView.value;
  neuralRenderer.draw(model, lastSnapshot);
  chartRenderer.draw(session.metrics);
  updateDecision();
  requestAnimationFrame(frame);
}
function updateDecision() {
  const s = lastSnapshot;
  if (!s) return;
  el.actionBars.innerHTML = '';
  for (let i = 0; i < ACTIONS.length; i++) {
    const row = document.createElement('div');
    row.className = 'actionRow';
    row.innerHTML = `<span>${ACTIONS[i]}</span><div class="barTrack"><div class="barFill" style="width:${(s.probs[i] * 100).toFixed(1)}%"></div></div><b>${(s.probs[i] * 100).toFixed(0)}%</b>`;
    el.actionBars.append(row);
  }
  el.value.textContent = `value ${s.value.toFixed(3)}`;
  const rp = viewWorld.lastRewardParts || {};
  el.rewardParts.textContent = `reward  food ${(rp.food || 0).toFixed(3)}  approach ${(rp.approach || 0).toFixed(3)}  energy ${(rp.energy || 0).toFixed(3)}  wall ${(rp.wall || 0).toFixed(3)}  hazard ${(rp.hazard || 0).toFixed(3)}  death ${(rp.death || 0).toFixed(3)}`;
  el.energy.value = viewWorld.agent.energy;
  el.energyText.textContent = `${Math.round(viewWorld.agent.energy * 100)}%`;
  el.worldMeta.textContent = `seed ${viewWorld.seed}`;
  el.viewBrainBadge.textContent = selectedSourceLabel();
}

function updateUI() {
  const m = session.metrics.at(-1);
  el.steps.textContent = session.totalSteps.toLocaleString();
  el.episodes.textContent = session.totalEpisodes.toLocaleString();
  el.ret.textContent = (m?.meanReturn ?? 0).toFixed(2);
  el.food.textContent = (m?.meanFood ?? 0).toFixed(2);
  el.entropy.textContent = m?.entropy?.toFixed(3) ?? '—';
  el.speed.textContent = m?.throughput ? Math.round(m.throughput).toLocaleString() : '—';
  el.lr.textContent = Number.isFinite(m?.learningRate) ? m.learningRate.toExponential(2) : session.trainer.learningRate.toExponential(2);
  el.kl.textContent = Number.isFinite(m?.maxEpochKL) ? m.maxEpochKL.toFixed(4) : '—';
  el.epochs.textContent = Number.isFinite(m?.epochsRun) ? `${m.epochsRun}${m.earlyStopped ? ' stop' : ''}` : '—';
  el.paramCount.textContent = `${session.model.paramCount().toLocaleString()} params`;
  el.curriculum.textContent = session.curriculum.current().name;
  const lv = session.validationHistory.at(-1);
  el.latestValidation.textContent = lv ? `${(lv.validation.score * 100).toFixed(1)}% @ ${lv.steps.toLocaleString()}${lv.regression ? ' REGRESSION' : ''}` : '—';
  const balancedBrain = session.getArchiveBrain('balanced');
  if (balancedBrain?.validation?.categoryScores) el.bestValidation.textContent = `${(balancedBrain.validation.categoryScores.balanced * 100).toFixed(1)}% @ ${balancedBrain.savedAtSteps.toLocaleString()}`;
  else if (balancedBrain?.validation) el.bestValidation.textContent = `legacy ${Number(balancedBrain.validation.score ?? 0).toFixed(2)} @ ${balancedBrain.savedAtSteps.toLocaleString()}`;
  else el.bestValidation.textContent = '—';
  el.retentionAlert.textContent = session.retentionStatus?.forgetting?.length ? `${session.retentionStatus.forgetting.length} FORGOTTEN` : (session.lastSkillValidation ? 'OK' : '—');
  renderSkillRetention();
  chartRenderer.draw(session.metrics);
  syncArchiveControls();
}
function renderSkillRetention() {
  const validation = session.lastSkillValidation;
  if (!validation?.stageResults?.length) { el.skillRetention.textContent = 'Waiting for v0.1.1 validation…'; return; }
  el.skillRetention.innerHTML = '';
  const forgetting = new Set((session.retentionStatus?.forgetting || []).map(x => x.stage));
  for (const stage of validation.stageResults) {
    const cell = document.createElement('div');
    cell.className = `skillCell${forgetting.has(stage.stage) ? ' forgetting' : ''}`;
    const best = session.skillBestScores[stage.stage] || stage.skillScore;
    cell.innerHTML = `<b>${stage.name}</b><span>now ${(stage.skillScore * 100).toFixed(0)}% • best ${(best * 100).toFixed(0)}%</span>`;
    el.skillRetention.append(cell);
  }
}
function syncArchiveControls() {
  const map = {
    balanced: el.bestOption,
    overall: el.overallOption,
    forager: el.foragerOption,
    survivor: el.survivorOption,
    efficiency: el.efficiencyOption,
  };
  for (const [category, option] of Object.entries(map)) {
    const brain = session.getArchiveBrain(category);
    option.disabled = !brain?.model;
    const isLegacy = Boolean(brain?.model && !brain?.validation?.categoryScores);
    option.textContent = brain
      ? isLegacy
        ? `Legacy Best @ ${brain.savedAtSteps.toLocaleString()} (rebaseline pending)`
        : `Best ${title(category)} @ ${brain.savedAtSteps.toLocaleString()}`
      : `Best ${title(category)} (not validated)`;
  }
  const selected = selectedArchiveCategory();
  if (selected && !session.getArchiveBrain(selected)) el.brainSource.value = 'latest';
  const restoreTarget = selectedArchiveBrain() || session.getArchiveBrain('balanced');
  el.restoreBest.disabled = !restoreTarget?.validation?.categoryScores;
}

function evaluationText(name, r) {
  return `${name}\nheld-out episodes: ${r.episodes}\nmean return: ${r.meanReturn.toFixed(3)}\nmean food: ${r.meanFood.toFixed(3)}\nsurvival: ${(r.survivalRate * 100).toFixed(1)}%\nmean energy: ${r.meanEnergy.toFixed(3)}\nmean steps: ${r.meanSteps.toFixed(1)}`;
}
async function runUnseen() {
  paused = true;
  el.pause.textContent = 'Resume';
  setStatus('Running held-out evaluation…');
  await yieldUI();
  const r = evaluateModel(session.model, session.curriculum.current(), { episodes: CONFIG.runtime.evalEpisodes, seedBase: 'heldout:v1', deterministic: false });
  let text = evaluationText(`UNSEEN TEST • LATEST @ ${session.totalSteps.toLocaleString()} steps`, r);
  const best = session.getArchiveBrain('balanced');
  if (best?.model) {
    await yieldUI();
    const model = new RecurrentActorCritic(1);
    model.restore(best.model);
    const br = evaluateModel(model, session.curriculum.current(), { episodes: CONFIG.runtime.evalEpisodes, seedBase: 'heldout:v1', deterministic: false });
    text += `\n\n${evaluationText(`PROTECTED BALANCED @ ${best.savedAtSteps.toLocaleString()} steps`, br)}`;
  }
  el.results.textContent = text;
  setStatus('Held-out evaluation complete. Training and validation seed domains were not used.');
}
async function compareBrains() {
  paused = true;
  el.pause.textContent = 'Resume';
  setStatus('Comparing historical brains on identical held-out seeds…');
  await yieldUI();
  const historical = [...session.milestones.values()].sort((a, b) => a.savedAtSteps - b.savedAtSteps);
  const selected = [], selectedSteps = new Set();
  const addHistorical = cp => { if (cp && !selectedSteps.has(cp.savedAtSteps)) { selectedSteps.add(cp.savedAtSteps); selected.push(cp); } };
  addHistorical(historical[0]);
  for (const target of [100000, 500000, 1000000]) addHistorical(historical.find(cp => cp.label === target));
  for (const cp of historical.slice(-8)) addHistorical(cp);
  selected.sort((a, b) => a.savedAtSteps - b.savedAtSteps);
  const raw = selected.map(cp => ({ savedAtSteps: cp.savedAtSteps, model: cp.model, kind: 'milestone' }));
  raw.push({ savedAtSteps: session.totalSteps, model: session.model.serialize(), kind: 'latest' });
  const balanced = session.getArchiveBrain('balanced');
  if (balanced?.model) raw.push({ savedAtSteps: balanced.savedAtSteps, model: balanced.model, kind: 'balanced' });
  const entries = [], seen = new Set();
  for (const e of raw) {
    const key = `${e.savedAtSteps}:${e.kind}`;
    if (!seen.has(key)) { seen.add(key); entries.push(e); }
  }
  if (!entries.length) { el.results.textContent = 'No historical checkpoints yet.'; return; }
  const rows = [];
  for (const cp of entries) {
    const model = new RecurrentActorCritic(1);
    model.restore(cp.model);
    const r = evaluateModel(model, CURRICULUM[Math.min(session.curriculum.stage, CURRICULUM.length - 1)], { episodes: CONFIG.runtime.compareEpisodes, seedBase: 'heldout:v1', deterministic: false });
    rows.push({ cp, r });
    await yieldUI();
  }
  let html = '<table class="resultsTable"><thead><tr><th>brain</th><th>return</th><th>food</th><th>survival</th></tr></thead><tbody>';
  for (const { cp, r } of rows) {
    const isBest = cp.kind === 'balanced', isLatest = cp.kind === 'latest';
    const cls = isBest ? 'bestRow' : '';
    const label = `${cp.savedAtSteps.toLocaleString()}${isBest ? ' ★ BALANCED' : ''}${isLatest ? ' LATEST' : ''}`;
    html += `<tr class="${cls}"><td>${label}</td><td>${r.meanReturn.toFixed(2)}</td><td>${r.meanFood.toFixed(2)}</td><td>${(r.survivalRate * 100).toFixed(0)}%</td></tr>`;
  }
  html += '</tbody></table>';
  el.results.innerHTML = html;
  setStatus(`Checkpoint comparison complete using ${CONFIG.runtime.compareEpisodes} identical held-out episodes per brain.`);
}
function yieldUI() { return new Promise(resolve => setTimeout(resolve, 20)); }
function title(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

el.newBrain.addEventListener('click', () => { if (confirm('Create a new untrained brain? Current unsaved progress will be replaced.')) resetBrain(); });
el.learn.addEventListener('click', () => setMode('LEARN'));
el.observe.addEventListener('click', () => setMode('OBSERVE'));
el.probe.addEventListener('click', () => setMode('PROBE'));
el.pause.addEventListener('click', () => { paused = !paused; el.pause.textContent = paused ? 'Resume' : 'Pause'; setStatus(paused ? 'Paused. Neural state remains inspectable.' : 'Resumed.'); });
el.brainView.addEventListener('change', () => { neuralRenderer.mode = el.brainView.value; });
el.brainSource.addEventListener('change', () => {
  const category = selectedArchiveCategory();
  if (category && !session.getArchiveBrain(category)) { el.brainSource.value = 'latest'; return; }
  if (mode !== 'LEARN') resetViewState();
  el.viewBrainBadge.textContent = selectedSourceLabel();
  syncArchiveControls();
  setStatus(mode === 'LEARN' ? 'View Brain selection applies in Observe/Probe; Learn always shows/trains Latest.' : `Now inspecting ${selectedSourceLabel().toLowerCase()} brain.`);
});
el.save.addEventListener('click', saveManualSafely);
el.loadManual.addEventListener('click', async () => { try { await restoreCheckpointRecord(await loadCheckpointRecord('latest'), 'Manual Save'); } catch (e) { setStatus(`Manual load failed: ${e.message}`); } });
el.loadAutosave.addEventListener('click', async () => { try { await restoreCheckpointRecord(await loadCheckpointRecord('autosave'), 'Validation Autosave'); } catch (e) { setStatus(`Autosave load failed: ${e.message}`); } });
el.restoreBest.addEventListener('click', () => {
  const category = selectedArchiveCategory() || 'balanced';
  const brain = session.getArchiveBrain(category);
  if (!brain) return;
  const source = brain.savedAtSteps;
  if (!confirm(`Restore the trainable policy + optimizer from Best ${title(category)} @ ${source.toLocaleString()} steps? Total experience steps will remain ${session.totalSteps.toLocaleString()} for an honest training history.`)) return;
  session.restoreBest(category);
  archiveModelCache = null;
  archiveModelCacheKey = null;
  el.brainSource.value = 'latest';
  resetViewState();
  updateUI();
  setStatus(`Latest policy restored from Best ${title(category)} @ ${source.toLocaleString()} steps. Training can resume from that brain.`);
});
el.test.addEventListener('click', runUnseen);
el.compare.addEventListener('click', compareBrains);
el.brain.addEventListener('pointerdown', e => { const text = neuralRenderer.inspectAt(e.clientX, e.clientY); if (text) el.inspect.textContent = text; });
el.world.addEventListener('pointerdown', e => {
  if (mode !== 'PROBE') return;
  const r = el.world.getBoundingClientRect();
  const x = Math.max(.03, Math.min(.97, (e.clientX - r.left) / r.width));
  const y = Math.max(.03, Math.min(.97, (e.clientY - r.top) / r.height));
  const tool = document.querySelector('input[name="probeTool"]:checked')?.value || 'food';
  if (tool === 'food' && viewWorld.food.length) {
    viewWorld.food[0].x = x; viewWorld.food[0].y = y; viewWorld.prevFoodDist = viewWorld.nearestFoodDistance();
  } else if (tool === 'hazard') {
    if (viewWorld.hazards.length) { viewWorld.hazards[0].x = x; viewWorld.hazards[0].y = y; }
    else viewWorld.hazards.push({ x, y, r: CONFIG.world.hazardRadius });
  } else {
    viewWorld.agent.x = x; viewWorld.agent.y = y; viewWorld.agent.vx = 0; viewWorld.agent.vy = 0;
  }
  updateProbe();
  setStatus(`Probe moved ${tool}. ${selectedSourceLabel()} policy outputs updated without taking an action.`);
});

updateUI();
setMode('LEARN');
requestAnimationFrame(frame);
refreshSaveSlots().then(({ manual }) => {
  if (manual?.snapshot?.totalSteps > session.totalSteps) setStatus(`Stored Manual Save detected at ${Number(manual.snapshot.totalSteps).toLocaleString()} steps. It is protected from lower-step overwrite; use Load Manual to recover it.`);
});
trainTick();
