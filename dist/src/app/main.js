import { ACTIONS, BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { TrainingSession } from '../ai/session.js';
import { RecurrentActorCritic } from '../ai/model.js';
import { evaluateFullRetentionSuite, evaluateHeldoutGeneralizationSuite, generalizationDiagnostic } from '../evaluation/evaluator.js';
import { CURRICULUM } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { loadCheckpointRecord, loadHallOfFameRecord, saveCheckpoint, saveHallOfFame } from '../storage/checkpoints.js';
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
  lineage: $('lineageVal'), rehearsal: $('rehearsalVal'), promotion: $('promotionVal'),
  pinChampion: $('pinChampionBtn'), branchSelect: $('branchSelect'), switchBranch: $('switchBranchBtn'), hallOptions: $('hallBrainOptions'),
  experienceAge: $('experienceAgeVal'), researchLineage: $('researchLineageVal'), policyOrigin: $('policyOriginVal'), activeChampion: $('activeChampionVal'), hall: $('hallVal'), branches: $('branchesVal'), hallList: $('hallList'),
  simSpeed: $('simSpeedVal'), ppoMs: $('ppoMsVal'), fps: $('fpsVal'), uiMs: $('uiMsVal'), validationMs: $('validationMsVal'), storageMs: $('storageMsVal'),
};
$('buildTag').textContent = `v${VERSION} • ${BUILD_MARKER}`;

const budgets = { eco: { rollout: 8, delay: 30 }, balanced: { rollout: 18, delay: 10 }, adaptive: { rollout: 18, delay: 8 }, max: { rollout: 36, delay: 0 } };
const archiveLabels = { balanced: 'CHAMPION BALANCED', overall: 'CHAMPION OVERALL', forager: 'CHAMPION FORAGER', survivor: 'CHAMPION SURVIVOR', efficiency: 'CHAMPION EFFICIENCY' };
let mode = 'LEARN', paused = false, trainBusy = false, observeAccum = 0, lastFrame = performance.now(), viewSeedIndex = 0;
let lastVisualRender = 0, lastUiPaint = 0, lastUiDurationMs = 0, lastStorageDurationMs = 0, browserFps = 60, rafFrames = 0, rafWindowStart = performance.now(), adaptiveDelay = 8, controlSignature = '';
let session = new TrainingSession({ seed: 1337, envCount: CONFIG.runtime.trainEnvs });
let archiveModelCache = null, archiveModelCacheKey = null;
let viewWorld = createViewWorld();
let viewObs = viewWorld.observe(), viewHidden = session.model.zeroHidden(), lastSnapshot = session.model.forward(viewObs, viewHidden);
const viewRng = new PRNG(0xabc123);
const worldRenderer = new WorldRenderer(el.world), neuralRenderer = new NeuralRenderer(el.brain), chartRenderer = new ChartRenderer(el.chart);
const actionUi = ACTIONS.map(action => {
  const row = document.createElement('div');
  row.className = 'actionRow';
  const name = document.createElement('span');
  name.textContent = action;
  const track = document.createElement('div');
  track.className = 'barTrack';
  const fill = document.createElement('div');
  fill.className = 'barFill';
  track.append(fill);
  const value = document.createElement('b');
  value.textContent = '0%';
  row.append(name, track, value);
  el.actionBars.append(row);
  return { fill, value };
});

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
async function persistHallOfFame() {
  const started = performance.now();
  await saveHallOfFame(session.exportHallOfFame());
  lastStorageDurationMs = performance.now() - started;
}

async function hydratePersistentHall({ pinMigrationBaseline = false } = {}) {
  try {
    const record = await loadHallOfFameRecord();
    if (record?.snapshot) session.mergeHallOfFame(record.snapshot);
    if (pinMigrationBaseline && session.hallOfFame.length === 0 && session.getArchiveBrain('balanced')?.model) {
      session.pinChampion('balanced', { reason: 'v0.1.2-migration-baseline' });
    }
    if (session.hallOfFame.length) await persistHallOfFame();
  } catch (err) {
    console.warn('Hall of Fame storage unavailable', err);
  }
}

async function restoreCheckpointRecord(record, label) {
  const cp = record?.snapshot;
  if (!cp) { setStatus(`No ${label.toLowerCase()} found.`); return false; }
  paused = true;
  el.pause.textContent = 'Resume';
  session.restore(cp);
  await hydratePersistentHall({ pinMigrationBaseline: cp.schema === 5 });
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  updateUI();
  await refreshSaveSlots();
  const migration = cp.schema < 6
    ? ' Legacy checkpoint loaded safely. Existing Champions are preserved; v0.1.2 Balanced Champion was pinned into the Hall of Fame when no prior Hall record existed.'
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
    const started = performance.now();
    await saveCheckpoint(session.snapshot(), 'latest');
    await persistHallOfFame();
    lastStorageDurationMs = performance.now() - started;
    await refreshSaveSlots();
    setStatus(`Manual checkpoint saved at ${session.totalSteps.toLocaleString()} steps. Learner branches, rehearsal state, Champions, and Hall of Fame backup included.`);
  } catch (e) { setStatus(`Save failed: ${e.message}`); }
}

function selectedBrainRef() {
  const value = el.brainSource.value;
  if (value === 'latest') return { type: 'learner', key: 'latest', brain: null };
  if (value.startsWith('hof:')) {
    const id = value.slice(4);
    return { type: 'hall', key: value, id, brain: session.getHallEntry(id) };
  }
  return { type: 'champion', key: value, category: value, brain: session.getArchiveBrain(value) };
}
function selectedArchiveCategory() {
  const ref = selectedBrainRef();
  return ref.type === 'champion' ? ref.category : null;
}
function selectedArchiveBrain() { return selectedBrainRef().brain; }
function selectedViewModel() {
  const ref = selectedBrainRef();
  const brain = ref.brain;
  if (mode === 'LEARN' || ref.type === 'learner' || !brain?.model) return session.model;
  const key = `${ref.key}:${brain.savedAtSteps}`;
  if (!archiveModelCache || archiveModelCacheKey !== key) {
    archiveModelCache = new RecurrentActorCritic(1);
    archiveModelCache.restore(brain.model);
    archiveModelCacheKey = key;
  }
  return archiveModelCache;
}
function selectedSourceLabel() {
  const ref = selectedBrainRef();
  if (mode === 'LEARN' || ref.type === 'learner' || !ref.brain) return 'LEARNER';
  if (ref.type === 'hall') return `${ref.brain.id} HALL OF FAME`;
  return archiveLabels[ref.category] || ref.category.toUpperCase();
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
    ? 'Autonomous Learner active. PPO trains continuously across current challenges plus rehearsal of earlier skills; Champions are frozen observers.'
    : next === 'OBSERVE'
      ? `Watching the ${selectedSourceLabel().toLowerCase()} policy in a separate procedural world.`
      : 'World frozen. Tap to manipulate stimuli and inspect the selected policy response.');
}
function resetBrain() {
  const preservedHall = session.exportHallOfFame();
  session = new TrainingSession({ seed: (Date.now() >>> 0), envCount: CONFIG.runtime.trainEnvs });
  session.mergeHallOfFame(preservedHall);
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  syncArchiveControls();
  setStatus('New untrained brain created. Permanent Hall of Fame preserved; prior active learner branches are not part of the new experiment.');
  updateUI();
}

function nextTrainingDelay(budgetKey, baseDelay) {
  if (budgetKey !== 'adaptive') return baseDelay;
  if (browserFps < 46) adaptiveDelay = Math.min(40, adaptiveDelay + 2);
  else if (browserFps > 56) adaptiveDelay = Math.max(0, adaptiveDelay - 1);
  return adaptiveDelay;
}

function maybeUpdateUI(force = false) {
  const now = performance.now();
  const minInterval = mode === 'LEARN' ? 1000 / Math.max(1, CONFIG.runtime.learnUiHz) : 0;
  if (!force && now - lastUiPaint < minInterval) return;
  lastUiPaint = now;
  updateUI();
}

async function trainTick() {
  if (mode !== 'LEARN' || paused) { setTimeout(trainTick, 40); return; }
  if (trainBusy) { setTimeout(trainTick, 10); return; }
  trainBusy = true;
  let forceUi = false;
  try {
    const budgetKey = el.budget.value;
    const b = budgets[budgetKey] || budgets.balanced;
    const result = session.trainRollout(b.rollout);
    viewWorld = session.envs[0];
    viewObs = session.obs[0];
    viewHidden = session.hidden[0];
    lastSnapshot = session.model.forward(viewObs, viewHidden);

    if (result.metric.updateRejected) {
      forceUi = true;
      setStatus(`PPO safety rejected an oversized numerical update (KL ${result.metric.maxEpochKL.toFixed(3)}). That optimizer step was undone; autonomous behavioral exploration remains untouched.`);
    } else if (result.validation) {
      forceUi = true;
      archiveModelCache = null;
      archiveModelCacheKey = null;
      syncArchiveControls();
      const v = result.validation;
      const alerts = v.forgetting?.length ? ` • observed: ${v.forgetting.map(x => `${x.name}${x.confirmed ? ' confirmed' : ''}`).join(', ')}` : '';
      const interp = String(v.interpretation || 'healthy').replaceAll('-', ' ');
      const pendingBalanced = v.promotionPending?.find(x => x.category === 'balanced');
      if (v.improved) setStatus(`New Champion Balanced promoted @ ${session.bestBrain.savedAtSteps.toLocaleString()} after repeat-confirmed validation. Learner continues independently; pin it manually if you want this Champion in the permanent Hall.`);
      else if (pendingBalanced) setStatus(`Learner is challenging Champion Balanced: confirmation ${pendingBalanced.streak}/${pendingBalanced.required}. No weights changed by validation.`);
      else if (v.balancedEvidence || v.forgetting?.length) setStatus(`Validation observed ${interp}: learner ${(v.validation.score * 100).toFixed(1)}% • champion ${(v.bestScore * 100).toFixed(1)}%. Learner keeps learning; no behavioral rollback.${alerts}`);
      else setStatus(`All-skills validation complete: learner ${(v.validation.score * 100).toFixed(1)}% • champion ${(v.bestScore * 100).toFixed(1)}% • retention ${interp}.`);
      try {
        const saveStarted = performance.now();
        await saveCheckpoint(session.snapshot(), 'autosave');
        lastStorageDurationMs = performance.now() - saveStarted;
        await refreshSaveSlots();
      } catch (saveErr) {
        console.warn('Validation autosave failed', saveErr);
        setStatus(`${el.status.textContent} Autosave failed: ${saveErr.message}`);
      }
    } else if (result.curriculumEvent) {
      forceUi = true;
      const e = result.curriculumEvent;
      setStatus(`Autonomous curriculum ${e.reason}: ${CURRICULUM[e.from].name} → ${CURRICULUM[e.to].name}. Earlier skills remain in rehearsal.`);
    } else if (result.metric.earlyStopped && result.metric.maxEpochKL > CONFIG.ppo.targetKL) {
      setStatus(`PPO epoch stopped early at KL ${result.metric.maxEpochKL.toFixed(3)} to avoid a numerically oversized optimizer step.`);
    }
    maybeUpdateUI(forceUi);
    setTimeout(trainTick, nextTrainingDelay(budgetKey, b.delay));
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
  rafFrames++;
  const rafWindow = now - rafWindowStart;
  if (rafWindow >= 1000) {
    browserFps = rafFrames * 1000 / rafWindow;
    rafFrames = 0;
    rafWindowStart = now;
  }
  if (mode === 'OBSERVE' && !paused) {
    observeAccum += dt;
    while (observeAccum >= 55) { stepObserved(); observeAccum -= 55; }
  }
  const renderHz = mode === 'LEARN'
    ? CONFIG.runtime.learnRenderHz
    : mode === 'OBSERVE'
      ? CONFIG.runtime.observeRenderHz
      : CONFIG.runtime.probeRenderHz;
  if (now - lastVisualRender >= 1000 / Math.max(1, renderHz)) {
    lastVisualRender = now;
    const model = selectedViewModel();
    worldRenderer.draw(viewWorld, mode + (paused ? ' • PAUSED' : ''));
    neuralRenderer.mode = el.brainView.value;
    neuralRenderer.draw(model, lastSnapshot);
    updateDecision();
  }
  requestAnimationFrame(frame);
}
function updateDecision() {
  const s = lastSnapshot;
  if (!s) return;
  for (let i = 0; i < ACTIONS.length; i++) {
    const pctValue = s.probs[i] * 100;
    actionUi[i].fill.style.width = `${pctValue.toFixed(1)}%`;
    actionUi[i].value.textContent = `${pctValue.toFixed(0)}%`;
  }
  el.value.textContent = `value ${s.value.toFixed(3)}`;
  const rp = viewWorld.lastRewardParts || {};
  el.rewardParts.textContent = `reward  food ${(rp.food || 0).toFixed(3)}  approach ${(rp.approach || 0).toFixed(3)}  energy ${(rp.energy || 0).toFixed(3)}  wall ${(rp.wall || 0).toFixed(3)}  hazard ${(rp.hazard || 0).toFixed(3)}  death ${(rp.death || 0).toFixed(3)}`;
  el.energy.value = viewWorld.agent.energy;
  el.energyText.textContent = `${Math.round(viewWorld.agent.energy * 100)}%`;
  el.worldMeta.textContent = `seed ${viewWorld.seed}`;
  el.viewBrainBadge.textContent = selectedSourceLabel();
}

function policyOriginText() {
  const lineage = session.learnerLineage;
  const parent = lineage?.parent;
  if (!parent) return 'native learner';
  if (parent.type === 'hall-of-fame') return `${parent.hallId || 'Hall'} @ ${Number(parent.sourceSteps || 0).toLocaleString()}`;
  if (parent.type === 'champion') return `Champion ${title(parent.category || 'balanced')} @ ${Number(parent.sourceSteps || 0).toLocaleString()}`;
  if (parent.legacySchema) return `schema-${parent.legacySchema} migration`;
  return lineage.reason || 'derived learner';
}

function updateUI() {
  const uiStarted = performance.now();
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
  el.latestValidation.textContent = lv ? `${(lv.validation.score * 100).toFixed(1)}% @ ${lv.steps.toLocaleString()}${lv.balancedConfirmed ? ' CONFIRMED' : lv.balancedEvidence ? ' WATCH' : ''}` : '—';
  const balancedBrain = session.getArchiveBrain('balanced');
  if (balancedBrain?.validation?.categoryScores) el.bestValidation.textContent = `${(balancedBrain.validation.categoryScores.balanced * 100).toFixed(1)}% @ ${balancedBrain.savedAtSteps.toLocaleString()}`;
  else if (balancedBrain?.validation) el.bestValidation.textContent = `legacy ${Number(balancedBrain.validation.score ?? 0).toFixed(2)} @ ${balancedBrain.savedAtSteps.toLocaleString()}`;
  else el.bestValidation.textContent = '—';
  el.retentionAlert.textContent = session.lastSkillValidation ? retentionLabel(session.retentionStatus) : '—';
  el.lineage.textContent = `${session.learnerLineage?.id || '—'} @ ${(session.learnerLineage?.startedAtSteps ?? 0).toLocaleString()} • ${Number(session.learnerExperienceSteps || 0).toLocaleString()} trained`;
  const mix = session.recentRehearsalMix();
  const mixValues = mix.total ? mix.fractions : mix.target;
  el.rehearsal.textContent = mixValues.map((x, i) => x >= 0.005 ? `${CURRICULUM[i].name.split(' ')[0]} ${Math.round(x * 100)}%` : null).filter(Boolean).join(' • ');
  const pending = Object.entries(session.promotionCandidates || {}).map(([category, x]) => `${title(category)} ${x.streak}/${CONFIG.validation.championPromotionConfirmations}`);
  el.promotion.textContent = pending.length ? pending.join(' • ') : 'none pending';

  const profile = m?.profile || {};
  el.simSpeed.textContent = Number.isFinite(profile.simulationStepsPerSec) ? `${Math.round(profile.simulationStepsPerSec).toLocaleString()}/s` : '—';
  el.ppoMs.textContent = Number.isFinite(profile.ppoMs) ? `${profile.ppoMs.toFixed(1)} ms` : '—';
  el.fps.textContent = `${browserFps.toFixed(0)}`;
  el.validationMs.textContent = Number.isFinite(profile.validationMs) && profile.validationMs > 0 ? `${profile.validationMs.toFixed(0)} ms` : '—';
  el.storageMs.textContent = lastStorageDurationMs > 0 ? `${lastStorageDurationMs.toFixed(0)} ms` : '—';

  const hall = session.hallOfFameSummary();
  const branches = session.frozenLearnerSummary();
  el.experienceAge.textContent = session.totalSteps.toLocaleString();
  el.researchLineage.textContent = `${session.learnerLineage?.id || '—'} • ${Number(session.learnerExperienceSteps || 0).toLocaleString()} branch steps`;
  el.policyOrigin.textContent = policyOriginText();
  el.activeChampion.textContent = balancedBrain ? `${balancedBrain.savedAtSteps.toLocaleString()} • ${pct(balancedBrain.validation?.categoryScores?.balanced)}` : 'none';
  el.hall.textContent = hall.length ? `${hall.length} pinned` : 'empty';
  el.branches.textContent = `${branches.length}`;
  el.hallList.textContent = hall.length
    ? hall.map(x => `${x.id} • ${Number(x.savedAtSteps || 0).toLocaleString()} • ${title(x.category || 'brain')} • ${x.lineage?.id || 'unknown lineage'}`).join('   |   ')
    : 'No pinned historic brains yet.';

  renderSkillRetention();
  chartRenderer.draw(session.metrics);
  syncArchiveControls();
  lastUiDurationMs = performance.now() - uiStarted;
  el.uiMs.textContent = `${lastUiDurationMs.toFixed(1)} ms`;
}

function renderSkillRetention() {
  const validation = session.lastSkillValidation;
  if (!validation?.stageResults?.length) { el.skillRetention.textContent = 'Waiting for autonomous continual-learning validation…'; return; }
  el.skillRetention.innerHTML = '';
  const alerts = new Map((session.retentionStatus?.alerts || session.retentionStatus?.forgetting || []).map(x => [x.stage, x]));
  for (const stage of validation.stageResults) {
    const cell = document.createElement('div');
    const alert = alerts.get(stage.stage);
    const best = session.skillBestRecords?.[stage.stage];
    const cls = alert ? (alert.confirmed ? ' forgetting confirmed' : ' forgetting watch') : '';
    cell.className = `skillCell${cls}`;
    const nowRange = `${pct(stage.skillCiLow)}–${pct(stage.skillCiHigh)}`;
    const bestScore = best?.score ?? stage.skillScore;
    const bestRange = best ? `${pct(best.ciLow)}–${pct(best.ciHigh)}` : nowRange;
    const tag = alert ? ` • ${alert.confirmed ? 'CONFIRMED' : 'WATCH'} ${alert.severity.toUpperCase()} x${alert.streak}` : '';
    cell.innerHTML = `<b>${stage.name}</b><span>now ${pct(stage.skillScore)} [${nowRange}]</span><span>best ${pct(bestScore)} [${bestRange}]${tag}</span>`;
    el.skillRetention.append(cell);
  }
}
function pct(x) { return `${(Math.max(0, Math.min(1, Number(x) || 0)) * 100).toFixed(0)}%`; }
function retentionLabel(status) {
  const label = String(status?.interpretation || 'unvalidated').replaceAll('-', ' ').toUpperCase();
  const alerts = status?.alerts?.length || status?.forgetting?.length || 0;
  return alerts ? `${label} • ${alerts}` : label === 'HEALTHY' ? 'OK' : label;
}

function syncArchiveControls() {
  const championMap = {
    balanced: el.bestOption,
    overall: el.overallOption,
    forager: el.foragerOption,
    survivor: el.survivorOption,
    efficiency: el.efficiencyOption,
  };
  const hall = session.hallOfFameSummary();
  const branches = session.frozenLearnerSummary();
  const signature = JSON.stringify({
    archiveNeedsRebaseline: session.archiveNeedsRebaseline,
    champions: Object.fromEntries(Object.keys(championMap).map(k => [k, session.getArchiveBrain(k)?.savedAtSteps ?? null])),
    hall: hall.map(x => [x.id, x.savedAtSteps]),
    branches: branches.map(x => [x.id, x.frozenAtSteps]),
    active: session.learnerLineage?.id,
  });
  if (signature !== controlSignature) {
    controlSignature = signature;
    for (const [category, option] of Object.entries(championMap)) {
      const brain = session.getArchiveBrain(category);
      option.disabled = !brain?.model;
      const needsRebaseline = session.archiveNeedsRebaseline;
      const isLegacy = Boolean(brain?.model && !brain?.validation?.categoryScores);
      option.textContent = brain
        ? needsRebaseline
          ? `Prior Champion ${title(category)} @ ${brain.savedAtSteps.toLocaleString()} (calibration pending)`
          : isLegacy
            ? `Legacy Champion @ ${brain.savedAtSteps.toLocaleString()} (calibration pending)`
            : `Champion ${title(category)} @ ${brain.savedAtSteps.toLocaleString()}`
        : `Champion ${title(category)} (not validated)`;
    }
    const priorBrainValue = el.brainSource.value;
    el.hallOptions.replaceChildren();
    for (const entry of hall) {
      const option = document.createElement('option');
      option.value = `hof:${entry.id}`;
      option.textContent = `${entry.id} • ${Number(entry.savedAtSteps || 0).toLocaleString()} • ${title(entry.category || 'brain')}`;
      el.hallOptions.append(option);
    }
    if ([...el.brainSource.options].some(o => o.value === priorBrainValue && !o.disabled)) el.brainSource.value = priorBrainValue;
    else el.brainSource.value = 'latest';

    const priorBranchValue = el.branchSelect.value;
    el.branchSelect.replaceChildren();
    const active = document.createElement('option');
    active.value = 'active';
    active.textContent = `Active ${session.learnerLineage?.id || 'Learner'} • ${Number(session.learnerExperienceSteps || 0).toLocaleString()} branch steps`;
    el.branchSelect.append(active);
    for (const branch of branches) {
      const option = document.createElement('option');
      option.value = branch.id;
      option.textContent = `${branch.id} • frozen @ ${Number(branch.frozenAtSteps || 0).toLocaleString()} • ${Number(branch.learnerExperienceSteps || 0).toLocaleString()} branch steps`;
      el.branchSelect.append(option);
    }
    if ([...el.branchSelect.options].some(o => o.value === priorBranchValue)) el.branchSelect.value = priorBranchValue;
    else el.branchSelect.value = 'active';
  }

  const ref = selectedBrainRef();
  if (ref.type !== 'learner' && !ref.brain?.model) el.brainSource.value = 'latest';
  const currentRef = selectedBrainRef();
  const forkTarget = currentRef.type === 'learner' ? session.getArchiveBrain('balanced') : currentRef.brain;
  el.restoreBest.disabled = session.archiveNeedsRebaseline || !forkTarget?.model;
  el.pinChampion.disabled = session.archiveNeedsRebaseline || currentRef.type !== 'champion' || !currentRef.brain?.model;
  el.switchBranch.disabled = el.branchSelect.value === 'active' || !session.frozenLearners.some(x => x.id === el.branchSelect.value);
}

function suiteText(name, r) {
  const skillLines = r.stageResults.map(x => `  ${x.name}: ${pct(x.skillScore)} [${pct(x.skillCiLow)}–${pct(x.skillCiHigh)}]`).join('\n');
  return `${name}\nprotocol: ${r.protocol}\nepisodes: ${r.episodes} (${r.episodesPerStage}/skill)\ngeneralization score: ${pct(r.balancedScore)} [${pct(r.balancedCiLow)}–${pct(r.balancedCiHigh)}]\nmean return: ${r.meanReturn.toFixed(3)}\nmean food: ${r.meanFood.toFixed(3)}\nsurvival: ${(r.survivalRate * 100).toFixed(1)}%\nmean energy: ${r.meanEnergy.toFixed(3)}\nmean steps: ${r.meanSteps.toFixed(1)}\nskills:\n${skillLines}`;
}
async function runUnseen() {
  paused = true;
  el.pause.textContent = 'Resume';
  setStatus('Running FINAL held-out all-skills evaluation… This diagnostic never updates Champions or promotion state.');
  await yieldUI();
  const latestCalibration = evaluateFullRetentionSuite(session.model, {
    episodesPerStage: CONFIG.validation.episodesPerStage,
    seedBase: CONFIG.validation.seedBase,
    deterministic: false,
    protocolTag: 'retention-v3-ci-readonly',
  });
  await yieldUI();
  const r = evaluateHeldoutGeneralizationSuite(session.model, {
    episodesPerStage: CONFIG.generalization.episodesPerStage,
    seedBase: CONFIG.generalization.seedBase,
    deterministic: false,
  });
  let text = suiteText(`FINAL UNSEEN ALL-SKILLS • LATEST @ ${session.totalSteps.toLocaleString()} steps`, r);
  text += `

READ-ONLY VALIDATION CALIBRATION @ SAME LATEST WEIGHTS
balanced: ${pct(latestCalibration.balancedScore)} [${pct(latestCalibration.balancedCiLow)}–${pct(latestCalibration.balancedCiHigh)}]
This measurement is not committed to validation history or archives.`;
  const best = session.getArchiveBrain('balanced');
  let br = null;
  if (best?.model) {
    await yieldUI();
    const model = new RecurrentActorCritic(1);
    model.restore(best.model);
    br = evaluateHeldoutGeneralizationSuite(model, {
      episodesPerStage: CONFIG.generalization.episodesPerStage,
      seedBase: CONFIG.generalization.seedBase,
      deterministic: false,
    });
    text += `\n\n${suiteText(`CHAMPION BALANCED @ ${best.savedAtSteps.toLocaleString()} steps`, br)}`;
  }
  const compatibleBestValidation = best?.validation?.protocol?.includes('retention-v3-ci') ? best.validation : null;
  const diagnostic = generalizationDiagnostic(latestCalibration, compatibleBestValidation, r, br);
  text += `\n\n${diagnostic.text}\nFINAL HOLDOUT IS DIAGNOSTIC ONLY — no weights, optimizer, Champion archive, curriculum, validation history, or promotion state were changed. Repeatedly consulting this set can still bias human decisions, so use it sparingly.`;
  el.results.textContent = text;
  setStatus(diagnostic.conflict
    ? 'Final holdout found a validation/generalization conflict. No automatic action taken.'
    : 'Final held-out all-skills evaluation complete. No training-selection state changed.');
}
async function compareBrains() {
  paused = true;
  el.pause.textContent = 'Resume';
  setStatus('Comparing historical brains on the separate held-out comparison domain…');
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
  for (const entry of session.hallOfFame || []) {
    if (entry?.model) raw.push({ savedAtSteps: entry.savedAtSteps, model: entry.model, kind: 'hall', hallId: entry.id });
  }
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
    const r = evaluateFullRetentionSuite(model, {
      episodesPerStage: CONFIG.generalization.compareEpisodesPerStage,
      seedBase: CONFIG.generalization.compareSeedBase,
      deterministic: false,
      protocolTag: 'heldout-comparison-v2',
    });
    rows.push({ cp, r });
    await yieldUI();
  }
  let html = '<table class="resultsTable"><thead><tr><th>brain</th><th>gen</th><th>return</th><th>food</th><th>survival</th></tr></thead><tbody>';
  for (const { cp, r } of rows) {
    const isBest = cp.kind === 'balanced', isLatest = cp.kind === 'latest', isHall = cp.kind === 'hall';
    const cls = isBest || isHall ? 'bestRow' : '';
    const label = `${cp.savedAtSteps.toLocaleString()}${isBest ? ' ★ CHAMPION' : ''}${isLatest ? ' LEARNER' : ''}${isHall ? ` ${cp.hallId} HALL` : ''}`;
    html += `<tr class="${cls}"><td>${label}</td><td>${pct(r.balancedScore)}</td><td>${r.meanReturn.toFixed(2)}</td><td>${r.meanFood.toFixed(2)}</td><td>${(r.survivalRate * 100).toFixed(0)}%</td></tr>`;
  }
  html += '</tbody></table><div class="comparisonNote">Comparison uses heldout:compare:v2, not the final Unseen Test domain.</div>';
  el.results.innerHTML = html;
  setStatus(`Checkpoint comparison complete using ${CONFIG.generalization.compareEpisodesPerStage} episodes per skill on a non-final comparison domain.`);
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
  const ref = selectedBrainRef();
  if (ref.type !== 'learner' && !ref.brain?.model) { el.brainSource.value = 'latest'; return; }
  if (mode !== 'LEARN') resetViewState();
  el.viewBrainBadge.textContent = selectedSourceLabel();
  syncArchiveControls();
  setStatus(mode === 'LEARN' ? 'View Brain selection applies in Observe/Probe; Learn always shows/trains the autonomous Learner.' : `Now inspecting ${selectedSourceLabel().toLowerCase()} brain.`);
});
el.save.addEventListener('click', saveManualSafely);
el.loadManual.addEventListener('click', async () => { try { await restoreCheckpointRecord(await loadCheckpointRecord('latest'), 'Manual Save'); } catch (e) { setStatus(`Manual load failed: ${e.message}`); } });
el.loadAutosave.addEventListener('click', async () => { try { await restoreCheckpointRecord(await loadCheckpointRecord('autosave'), 'Validation Autosave'); } catch (e) { setStatus(`Autosave load failed: ${e.message}`); } });
el.pinChampion.addEventListener('click', async () => {
  try {
    const ref = selectedBrainRef();
    if (ref.type !== 'champion' || !ref.brain?.model) { setStatus('Select a validated Champion in View Brain before pinning it.'); return; }
    const result = session.pinChampion(ref.category, { reason: 'manual-pin' });
    await persistHallOfFame();
    controlSignature = '';
    updateUI();
    setStatus(result.created
      ? `${result.entry.id} permanently pinned from Champion ${title(ref.category)} @ ${result.entry.savedAtSteps.toLocaleString()} steps.`
      : `${result.entry.id} already preserves that exact Champion policy; no duplicate was created.`);
  } catch (e) { setStatus(`Hall of Fame pin failed: ${e.message}`); }
});
el.restoreBest.addEventListener('click', () => {
  const ref = selectedBrainRef();
  const fallback = session.getArchiveBrain('balanced');
  const source = ref.type === 'learner' ? fallback : ref.brain;
  if (!source?.model) return;
  const sourceLabel = ref.type === 'hall'
    ? `${ref.brain.id} Hall of Fame @ ${ref.brain.savedAtSteps.toLocaleString()}`
    : ref.type === 'champion'
      ? `Champion ${title(ref.category)} @ ${ref.brain.savedAtSteps.toLocaleString()}`
      : `Champion Balanced @ ${fallback.savedAtSteps.toLocaleString()}`;
  if (!confirm(`Fork a NEW Learner lineage from ${sourceLabel}? The current Learner will be frozen and preserved as a switchable branch. Global experience age stays ${session.totalSteps.toLocaleString()}.`)) return;
  const event = ref.type === 'hall' ? session.forkFromHallOfFame(ref.id) : session.forkFromChampion(ref.type === 'champion' ? ref.category : 'balanced');
  archiveModelCache = null;
  archiveModelCacheKey = null;
  controlSignature = '';
  el.brainSource.value = 'latest';
  resetViewState();
  updateUI();
  setStatus(`New Learner ${event.lineageId} forked from ${sourceLabel}. Prior Learner ${event.preservedLineageId} is frozen and switchable.`);
});
el.branchSelect.addEventListener('change', syncArchiveControls);
el.switchBranch.addEventListener('click', () => {
  const id = el.branchSelect.value;
  if (!id || id === 'active') return;
  if (!confirm(`Switch active training to frozen Learner ${id}? The current Learner will be frozen first. Champions and Hall of Fame are shared; global experience age remains monotonic.`)) return;
  try {
    const event = session.switchToFrozenLearner(id);
    archiveModelCache = null;
    archiveModelCacheKey = null;
    controlSignature = '';
    el.brainSource.value = 'latest';
    resetViewState();
    updateUI();
    setStatus(`Active Learner switched to ${event.lineageId}. Previous Learner ${event.preservedLineageId} is frozen. Immediate validation is scheduled before long training continues.`);
  } catch (e) { setStatus(`Branch switch failed: ${e.message}`); }
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
hydratePersistentHall().then(() => {
  controlSignature = '';
  updateUI();
}).catch(e => setStatus(`Hall of Fame storage check failed: ${e.message}`));
trainTick();
