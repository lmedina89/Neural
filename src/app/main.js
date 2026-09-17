import { ACTIONS, BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { TrainingSession } from '../ai/session.js';
import { RecurrentActorCritic } from '../ai/model.js';
import { evaluateFullRetentionSuite, evaluateHeldoutGeneralizationSuite, generalizationDiagnostic } from '../evaluation/evaluator.js';
import { ORIENTATION_AUDIT, runOrientationAudit } from '../evaluation/orientationAudit.js';
import { LiveSpinRecorder, OCCLUSION_AUDIT, SPIN_TELEMETRY, foodVisibilityDiagnostic, runOcclusionAudit } from '../evaluation/occlusionSpinTelemetry.js';
import { CURRICULUM } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { loadCheckpointRecord, loadHallOfFameRecord, saveCheckpoint, saveHallOfFame } from '../storage/checkpoints.js';
import { WorldRenderer } from '../visualization/worldRenderer.js';
import { NeuralRenderer } from '../visualization/neuralRenderer.js';
import { ChartRenderer } from '../visualization/chartRenderer.js';
import { CuriosityRenderer } from '../visualization/curiosityRenderer.js';
import { MemoryRenderer, projectHiddenState } from '../visualization/memoryRenderer.js';
import { HistoryRenderer } from '../visualization/historyRenderer.js';
import { RollingStepRate, formatRollingRate } from './runtimeDiagnostics.js';

const $ = id => document.getElementById(id);
const el = {
  newBrain: $('newBrain'), learn: $('learnBtn'), observe: $('observeBtn'), probe: $('probeBtn'), pause: $('pauseBtn'),
  budget: $('budgetSelect'), brainSource: $('brainSource'), bestOption: $('bestOption'), overallOption: $('overallOption'),
  foragerOption: $('foragerOption'), survivorOption: $('survivorOption'), efficiencyOption: $('efficiencyOption'), restoreBest: $('restoreBestBtn'),
  save: $('saveBtn'), loadManual: $('loadManualBtn'), loadAutosave: $('loadAutosaveBtn'), test: $('testBtn'), compare: $('compareBtn'),
  mode: $('modeBadge'), status: $('statusText'), world: $('worldCanvas'), brain: $('brainCanvas'), chart: $('chartCanvas'),
  worldMeta: $('worldMeta'), probeTools: $('probeTools'), worldFx: $('worldFx'), worldFxBadge: $('worldFxBadge'), brainView: $('brainView'), viewBrainBadge: $('viewBrainBadge'), inspect: $('inspectText'),
  steps: $('stepsVal'), episodes: $('episodesVal'), ret: $('returnVal'), food: $('foodVal'), entropy: $('entropyVal'), speed: $('speedVal'),
  lr: $('lrVal'), kl: $('klVal'), epochs: $('epochsVal'), paramCount: $('paramCount'), actionBars: $('actionBars'), energy: $('energyBar'),
  energyText: $('energyText'), curriculum: $('curriculumText'), rewardParts: $('rewardParts'), value: $('valueText'), results: $('resultsText'),
  latestValidation: $('latestValidation'), bestValidation: $('bestValidation'), retentionAlert: $('retentionAlert'), skillRetention: $('skillRetention'),
  manualSlot: $('manualSlotInfo'), autosaveSlot: $('autosaveSlotInfo'),
  lineage: $('lineageVal'), rehearsal: $('rehearsalVal'), promotion: $('promotionVal'),
  pinChampion: $('pinChampionBtn'), branchSelect: $('branchSelect'), switchBranch: $('switchBranchBtn'), hallOptions: $('hallBrainOptions'),
  experienceAge: $('experienceAgeVal'), researchLineage: $('researchLineageVal'), policyOrigin: $('policyOriginVal'), activeChampion: $('activeChampionVal'), hall: $('hallVal'), branches: $('branchesVal'), hallList: $('hallList'),
  researchCompactSummary: $('researchCompactSummary'),
  simSpeed: $('simSpeedVal'), speed30: $('speed30Val'), ppoMs: $('ppoMsVal'), fps: $('fpsVal'), uiMs: $('uiMsVal'), validationMs: $('validationMsVal'), storageMs: $('storageMsVal'),
  pageState: $('pageStateVal'), visualState: $('visualStateVal'), meterState: $('meterStateVal'),
  curiosityCanvas: $('curiosityCanvas'), curiosityError: $('curiosityErrorVal'), curiosityNovelty: $('curiosityNoveltyVal'), curiosityBonus: $('curiosityBonusVal'),
  curiosityBudget: $('curiosityBudgetVal'), curiosityLoss: $('curiosityLossVal'), curiosityParams: $('curiosityParamsVal'), curiosityInspect: $('curiosityInspect'),
  curiosityInfluence: $('curiosityInfluenceVal'), curiosityApplied: $('curiosityAppliedVal'), curiosityShare: $('curiosityShareVal'),
  curiosityResets: $('curiosityResetsVal'), curiosityBudgetUse: $('curiosityBudgetUseVal'), curiosityExhaust: $('curiosityExhaustVal'),
  curiosityMode: $('curiosityModeSelect'), startCuriosityAudit: $('startCuriosityAuditBtn'), switchCuriosityAudit: $('switchCuriosityAuditBtn'), endCuriosityAudit: $('endCuriosityAuditBtn'),
  curiosityAuditRole: $('curiosityAuditRoleVal'), curiosityControlProgress: $('curiosityControlProgressVal'), curiosityRewardProgress: $('curiosityRewardProgressVal'),
  curiosityPairCheckpoint: $('curiosityPairCheckpointVal'), curiosityControlScore: $('curiosityControlScoreVal'), curiosityRewardScore: $('curiosityRewardScoreVal'), curiosityAuditResults: $('curiosityAuditResults'),
  curiosityCompactSummary: $('curiosityCompactSummary'), skillCompactSummary: $('skillCompactSummary'), resultsCompactSummary: $('resultsCompactSummary'),
  stabilityCompactSummary: $('stabilityCompactSummary'), stabilityStatus: $('stabilityStatusVal'), stabilityPolicyLoss: $('stabilityPolicyLossVal'), stabilityValueLoss: $('stabilityValueLossVal'),
  stabilityExplained: $('stabilityExplainedVal'), stabilityKl: $('stabilityKlVal'), stabilityClip: $('stabilityClipVal'), stabilityGrad: $('stabilityGradVal'), stabilityGradClip: $('stabilityGradClipVal'),
  stabilityParamDelta: $('stabilityParamDeltaVal'), stabilityParamMax: $('stabilityParamMaxVal'), stabilityAdvantage: $('stabilityAdvantageVal'), stabilityRejected: $('stabilityRejectedVal'),
  stabilityValidation: $('stabilityValidationText'), stabilityEvents: $('stabilityEventsText'),
  confidenceCompactSummary: $('confidenceCompactSummary'), confidenceVerdict: $('confidenceVerdictVal'), confidenceReference: $('confidenceReferenceVal'),
  confidenceSample: $('confidenceSampleVal'), confidenceBalanced: $('confidenceBalancedVal'), confidenceEvidence: $('confidenceEvidenceText'), confidenceHistory: $('confidenceHistoryText'),
  orientationCompactSummary: $('orientationCompactSummary'), orientationRun: $('orientationRunBtn'), orientationSource: $('orientationSourceVal'),
  orientationProtocol: $('orientationProtocolVal'), orientationRear: $('orientationRearVal'), orientationSpin: $('orientationSpinVal'), orientationResults: $('orientationResultsText'),
  spinCompactSummary: $('spinCompactSummary'), spinRecorderStatus: $('spinRecorderStatusVal'), spinEventCount: $('spinEventCountVal'), spinBlockedRate: $('spinBlockedRateVal'), spinLatestCause: $('spinLatestCauseVal'),
  spinResults: $('spinResultsText'), spinClear: $('spinClearBtn'), spinOcclusionRun: $('spinOcclusionRunBtn'), spinOcclusionResults: $('spinOcclusionResultsText'),
  observatoryNav: $('observatoryNav'), observatoryViewNote: $('observatoryViewNote'),
  predictWorld: $('predictWorldCanvas'), predictWorldMeta: $('predictWorldMeta'),
  memoryCanvas: $('memoryCanvas'), memoryMeta: $('memoryMeta'), memoryInspect: $('memoryInspect'),
  historyCanvas: $('historyCanvas'), historyMeta: $('historyMeta'), historyInspect: $('historyInspect'),
};
$('buildTag').textContent = `v${VERSION} • ${BUILD_MARKER}`;

const budgets = { eco: { rollout: 8, delay: 30 }, balanced: { rollout: 18, delay: 10 }, adaptive: { rollout: 18, delay: 8 }, max: { rollout: 36, delay: 0 } };
const archiveLabels = { balanced: 'CHAMPION BALANCED', overall: 'CHAMPION OVERALL', forager: 'CHAMPION FORAGER', survivor: 'CHAMPION SURVIVOR', efficiency: 'CHAMPION EFFICIENCY' };
let mode = 'LEARN', paused = false, trainBusy = false, observeAccum = 0, lastFrame = performance.now(), viewSeedIndex = 0;
let lastVisualRender = 0, lastCuriosityRender = 0, lastMemoryRender = 0, lastHistoryRender = 0, lastMemoryCaptureCheck = 0, lastUiPaint = 0, lastUiDurationMs = 0, lastStorageDurationMs = 0, browserFps = 60, rafFrames = 0, rafWindowStart = performance.now(), adaptiveDelay = 8, controlSignature = '';
let activeObservatoryView = 'live';
let lastOrientationAudit = null;
let lastOcclusionAudit = null;
const liveSpinRecorder = new LiveSpinRecorder();
const trainRate = new RollingStepRate({ windowsMs: [5000, 30000], sampleIntervalMs: 120 });
let memoryPoints = [], memoryLastSampleAt = 0, memoryLastStep = -1, memoryLastSourceKey = '', memoryLineageId = null, memoryLastEpisode = -1;
const MEMORY_MAX_POINTS = 320;

// Rendering is observational, so expensive canvases may sleep while they are
// outside the visual viewport. IntersectionObserver avoids synchronous layout
// reads on every animation frame. The last rendered frame stays on the canvas,
// so scrolling back into view looks identical and rendering resumes immediately.
const canvasVisibility = new WeakMap();
let canvasVisibilityObserver = null;
function canvasVisible(canvas) {
  if (!canvas || canvas.closest?.('[hidden]')) return false;
  const known = canvasVisibility.get(canvas);
  if (typeof known === 'boolean') return known;
  const r = canvas.getBoundingClientRect();
  return r.width > 1 && r.height > 1 && r.bottom >= -8 && r.right >= -8 && r.top <= (globalThis.innerHeight || 0) + 8 && r.left <= (globalThis.innerWidth || 0) + 8;
}
function installCanvasVisibilityObserver(canvases) {
  const nodes = canvases.filter(Boolean);
  if (!nodes.length || typeof IntersectionObserver === 'undefined') return;
  canvasVisibilityObserver?.disconnect?.();
  canvasVisibilityObserver = new IntersectionObserver(entries => {
    for (const entry of entries) canvasVisibility.set(entry.target, entry.isIntersecting && entry.intersectionRect.width > 0 && entry.intersectionRect.height > 0);
  }, { root: null, rootMargin: '48px 0px 48px 0px', threshold: 0 });
  for (const node of nodes) canvasVisibilityObserver.observe(node);
}
function currentVisualState() {
  if (document.hidden) return 'BACKGROUND IDLE';
  if (activeObservatoryView === 'live') {
    const active = canvasVisible(el.world) || canvasVisible(el.brain);
    return active ? 'LIVE ACTIVE' : 'LIVE OFFSCREEN-IDLE';
  }
  if (activeObservatoryView === 'predict') {
    const active = canvasVisible(el.predictWorld) || canvasVisible(el.curiosityCanvas);
    return active ? 'PREDICT ACTIVE' : 'PREDICT OFFSCREEN-IDLE';
  }
  if (activeObservatoryView === 'memory') return canvasVisible(el.memoryCanvas) ? 'MEMORY ACTIVE' : 'MEMORY OFFSCREEN-IDLE';
  if (activeObservatoryView === 'history') return canvasVisible(el.historyCanvas) ? 'HISTORY ACTIVE' : 'HISTORY OFFSCREEN-IDLE';
  return 'RESEARCH IDLE';
}
function resetTrainRateMeter() { trainRate.reset(performance.now(), session.totalSteps); }
let session = new TrainingSession({ seed: 1337, envCount: CONFIG.runtime.trainEnvs });
trainRate.reset(performance.now(), session.totalSteps);
let archiveModelCache = null, archiveModelCacheKey = null;
let viewWorld = createViewWorld();
let viewObs = viewWorld.observe(), viewHidden = session.model.zeroHidden(), lastSnapshot = session.model.forward(viewObs, viewHidden);
const viewRng = new PRNG(0xabc123);
const worldRenderer = new WorldRenderer(el.world), predictWorldRenderer = new WorldRenderer(el.predictWorld), neuralRenderer = new NeuralRenderer(el.brain), chartRenderer = new ChartRenderer(el.chart), curiosityRenderer = new CuriosityRenderer(el.curiosityCanvas), memoryRenderer = new MemoryRenderer(el.memoryCanvas), historyRenderer = new HistoryRenderer(el.historyCanvas);
installCanvasVisibilityObserver([el.world, el.brain, el.chart, el.predictWorld, el.curiosityCanvas, el.memoryCanvas, el.historyCanvas]);
worldRenderer.overlayMode = el.worldFx?.value || 'BOTH';
if (el.worldFxBadge && el.worldFx) el.worldFxBadge.textContent = el.worldFx.options[el.worldFx.selectedIndex]?.textContent?.toUpperCase() || 'ATTN + ECHO';
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

const observatoryNotes = {
  live: 'Live world and real recurrent policy activity. Hidden observatory views stop their heavy canvas rendering.',
  predict: 'Learned forward-model expectations versus actual sampled outcomes, plus the curiosity predictor. Visual only.',
  memory: 'Real recurrent hidden states projected into a deterministic 2D state space. Experience Ripples come from real novelty/reward events.',
  history: 'Persisted validation, Champion, Hall-of-Fame and lineage metadata reconstructed as a learning timeline and branch map.',
  research: 'Detailed PPO stability, retention and evaluation telemetry. These panels remain observational unless a control explicitly says otherwise.',
};
function setObservatoryView(view = 'live') {
  if (!Object.hasOwn(observatoryNotes, view)) view = 'live';
  activeObservatoryView = view;
  for (const node of document.querySelectorAll('[data-observatory-view]')) node.hidden = node.dataset.observatoryView !== view;
  for (const button of el.observatoryNav?.querySelectorAll('[data-view]') || []) {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  if (el.observatoryViewNote) el.observatoryViewNote.textContent = observatoryNotes[view];
  lastVisualRender = 0; lastCuriosityRender = 0; lastMemoryRender = 0; lastHistoryRender = 0;
}
function resetMemoryConstellation() {
  memoryPoints = []; memoryLastSampleAt = 0; memoryLastStep = -1; memoryLastSourceKey = ''; memoryLastEpisode = session.totalEpisodes; memoryLineageId = session.learnerLineage?.id || null;
  if (el.memoryInspect) el.memoryInspect.textContent = 'Each point is a real sampled recurrent hidden state. Similar internal states land near each other under a fixed deterministic projection; recent experience leaves a trail and real novelty/reward events create Experience Ripples.';
}
function argmax(values = []) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if ((Number(values[i]) || 0) > (Number(values[best]) || 0)) best = i;
  return best;
}
function captureMemoryState(now) {
  const lineage = session.learnerLineage?.id || '—';
  if (memoryLineageId !== lineage || session.totalSteps < memoryLastStep) resetMemoryConstellation();
  const minMs = mode === 'LEARN' ? 130 : 220;
  const sourceKey = mode === 'LEARN' ? `learn:${session.totalSteps}` : `${mode}:${viewWorld?.seed ?? '—'}:${viewWorld?.stepCount ?? 0}`;
  if (now - memoryLastSampleAt < minMs || sourceKey === memoryLastSourceKey) return;
  const hidden = mode === 'LEARN' ? session.hidden?.[0] : viewHidden;
  if (!hidden?.length) return;
  const projected = projectHiddenState(hidden);
  const world = mode === 'LEARN' ? session.envs?.[0] : viewWorld;
  const rewardParts = world?.lastRewardParts || {};
  const curiosity = mode === 'LEARN' && session.lastCuriosity && Math.abs(session.totalSteps - Number(session.lastCuriosity.atSteps || 0)) < 50_000 ? session.lastCuriosity : null;
  const novelty = Math.max(0, Math.min(1, Number(curiosity?.novelty) || 0));
  const action = Number.isInteger(curiosity?.action) ? curiosity.action : argmax(lastSnapshot?.probs || []);
  const episodeChanged = session.totalEpisodes !== memoryLastEpisode;
  const latestEpisode = episodeChanged ? session.episodeHistory?.at(-1) : null;
  let eventStrength = novelty;
  let eventType = novelty > .08 ? 'novelty' : 'ordinary state';
  if ((Number(rewardParts.food) || 0) > 0 || (Number(latestEpisode?.food) || 0) > 0) { eventStrength = Math.max(eventStrength, .92); eventType = 'reward'; }
  if ((Number(rewardParts.hazard) || 0) < 0 || (Number(rewardParts.death) || 0) < 0 || (Number(latestEpisode?.hazardHits) || 0) > 0) { eventStrength = Math.max(eventStrength, 1); eventType = 'danger'; }
  if (episodeChanged && eventStrength < .32) { eventStrength = .32; eventType = latestEpisode?.survived ? 'reward' : 'episode'; }
  memoryPoints.push({ ...projected, steps: session.totalSteps, episodes: session.totalEpisodes, createdAt: now, novelty, activity: projected.radius, action, actionLabel: ACTIONS[action] || `action ${action}`, eventStrength, eventType, lineage });
  if (memoryPoints.length > MEMORY_MAX_POINTS) memoryPoints.shift();
  memoryLastSampleAt = now; memoryLastStep = session.totalSteps; memoryLastSourceKey = sourceKey; memoryLastEpisode = session.totalEpisodes; memoryLineageId = lineage;
  if (el.memoryMeta) el.memoryMeta.textContent = `${memoryPoints.length}/${MEMORY_MAX_POINTS} states • ${lineage} • runtime buffer`;
}

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
      session.pinChampion('balanced', { reason: 'pre-curiosity-migration-baseline' });
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
  resetTrainRateMeter();
  resetMemoryConstellation();
  await hydratePersistentHall({ pinMigrationBaseline: cp.schema <= 6 });
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  updateUI();
  await refreshSaveSlots();
  const migration = cp.schema < 7
    ? ' Legacy policy/Champion state loaded safely. The v0.1.3 curiosity predictor starts fresh because older checkpoints did not contain one; when the Hall was empty, the existing Balanced Champion was pinned as a pre-curiosity reference.'
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
    setStatus(`Manual checkpoint saved at ${session.totalSteps.toLocaleString()} steps. Learner, curiosity predictor, branches, rehearsal state, Champions, and Hall of Fame backup included.`);
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
  liveSpinRecorder.beginEpisode();
}
function setMode(next) {
  const previousMode = mode;
  if (next === 'OBSERVE' && previousMode !== 'OBSERVE') liveSpinRecorder.reset();
  mode = next;
  resetTrainRateMeter();
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
  updateSpinTelemetryUI();
  const curiosityModeText = session.curiosityRewardMode === 'reward' ? 'curiosity reward ON' : 'curiosity observe-only (reward 0)';
  setStatus(next === 'LEARN'
    ? `Autonomous Learner active. PPO trains across current challenges plus rehearsal; ${curiosityModeText}. Champions remain frozen observers.`
    : next === 'OBSERVE'
      ? `Watching the ${selectedSourceLabel().toLowerCase()} policy in a separate procedural world.`
      : 'World frozen. Tap to manipulate stimuli and inspect the selected policy response.');
}
function resetBrain() {
  const preservedHall = session.exportHallOfFame();
  session = new TrainingSession({ seed: (Date.now() >>> 0), envCount: CONFIG.runtime.trainEnvs });
  resetTrainRateMeter();
  session.mergeHallOfFame(preservedHall);
  archiveModelCache = null;
  archiveModelCacheKey = null;
  viewSeedIndex = 0;
  el.brainSource.value = 'latest';
  resetViewState();
  resetMemoryConstellation();
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
      const confidence = v.confidenceAudit;
      if (v.improved) setStatus(`New Champion Balanced promoted @ ${session.bestBrain.savedAtSteps.toLocaleString()} after repeat-confirmed validation. Learner continues independently; pin it manually if you want this Champion in the permanent Hall.`);
      else if (confidence?.label === 'confirmed-regression') setStatus(`Validation confidence: CONFIRMED REGRESSION after paired fixed-seed replay. Learner ${(v.validation.score * 100).toFixed(1)}% • Champion ${(v.bestScore * 100).toFixed(1)}%. Observational only; no rollback or learning change was applied.`);
      else if (confidence?.label === 'likely-noise') setStatus(`Validation confidence: LIKELY NOISE. A raw drop triggered confirmation but did not survive the larger paired fixed-seed replay. Learner continues unchanged.`);
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

    if (result.auditEvaluation) {
      forceUi = true;
      const a = result.auditEvaluation;
      try {
        const saveStarted = performance.now();
        await saveCheckpoint(session.snapshot(), 'autosave');
        lastStorageDurationMs = performance.now() - saveStarted;
        await refreshSaveSlots();
      } catch (saveErr) {
        console.warn('Curiosity audit autosave failed', saveErr);
      }
      setStatus(`Curiosity A/B ${String(a.role).toUpperCase()} checkpoint @ ${Number(a.checkpointSteps).toLocaleString()} branch steps: balanced ${(a.validation.balancedScore * 100).toFixed(1)}%. Same fixed audit seeds; Champion promotion remains suspended.`);
    }
    if (result.auditBranchComplete) {
      forceUi = true;
      paused = true;
      el.pause.textContent = 'Resume';
      const audit = session.curiosityAuditSummary();
      const other = audit.role === 'control' ? 'CURIOSITY' : 'CONTROL';
      setStatus(audit.completed
        ? 'Curiosity A/B audit complete on both branches. Training paused. No winner was promoted automatically; review the paired results and End Audit when ready.'
        : `${String(audit.role || '').toUpperCase()} branch reached ${Number(audit.targetStepsPerBranch).toLocaleString()} audit steps. Training paused. Switch to ${other} to run the matched branch.`);
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
  const telemetrySample = liveSpinRecorder.beginSample(viewWorld, act.snapshot, act.action);
  const r = viewWorld.step(act.action);
  const capturedSpin = liveSpinRecorder.endSample(telemetrySample, viewWorld, r);
  viewObs = r.obs;
  viewHidden = act.hidden;
  lastSnapshot = act.snapshot;
  if (capturedSpin) {
    updateSpinTelemetryUI();
    setStatus(`Captured live spin event #${capturedSpin.id}: ${capturedSpin.cause} • LOS blocked ${(capturedSpin.blockedRate * 100).toFixed(0)}% of trigger window • ${capturedSpin.recovery}. Recorder remains read-only.`);
  } else if (r.done) {
    setStatus(`Episode complete (${selectedSourceLabel()}): food ${r.info.food}, return ${r.info.totalReward.toFixed(2)}.`);
  }
}
function updateProbe() {
  const model = selectedViewModel();
  viewObs = viewWorld.observe();
  lastSnapshot = model.forward(viewObs, viewHidden);
  updateDecision();
}

function cognitiveFx(model, snapshot, world) {
  if (!model || !snapshot) return null;
  const obs = snapshot.obs || [];
  const hiddenN = snapshot.h?.length || 0;
  const influence = Array(obs.length).fill(0);
  if (hiddenN && model.params?.wx) {
    for (let j = 0; j < obs.length; j++) {
      let sum = 0;
      for (let i = 0; i < hiddenN; i++) sum += Math.abs((Number(obs[j]) || 0) * model.params.wx[i * obs.length + j]);
      influence[j] = sum / hiddenN;
    }
    const max = Math.max(...influence, 1e-9);
    for (let j = 0; j < influence.length; j++) influence[j] = Math.min(1, influence[j] / max);
  }
  const probs = snapshot.probs || [];
  const ranked = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  const first = ranked[0] ?? 0, second = ranked[1] ?? first;
  const dominantProb = Number(probs[first]) || 0;
  const secondProb = Number(probs[second]) || 0;
  const rp = world?.lastRewardParts || {};
  const reward = Object.values(rp).reduce((a, v) => a + (Number(v) || 0), 0);
  const predictionEcho = mode === 'LEARN' && session.lastCuriosity?.prediction && session.lastCuriosity?.nextObs
    ? {
        prediction: Array.from(session.lastCuriosity.prediction),
        actual: Array.from(session.lastCuriosity.nextObs),
        action: Number(session.lastCuriosity.action),
        error: Number(session.lastCuriosity.error || 0),
        novelty: Number(session.lastCuriosity.novelty || 0),
      }
    : null;
  return {
    inputInfluence: influence,
    dominantAction: first,
    dominantProb,
    decisionConfidence: Math.max(0, dominantProb - secondProb),
    curiosityNovelty: mode === 'LEARN' ? Number(session.lastCuriosity?.novelty || 0) : 0,
    curiosityError: mode === 'LEARN' ? Number(session.lastCuriosity?.error || 0) : 0,
    predictionEcho,
    reward,
  };
}
function frame(now) {
  const dt = Math.min(100, now - lastFrame);
  lastFrame = now;
  if (!document.hidden) trainRate.sample(now, session.totalSteps);
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
  if (now - lastMemoryCaptureCheck >= 30) { lastMemoryCaptureCheck = now; captureMemoryState(now); }
  if (now - lastVisualRender >= 1000 / Math.max(1, renderHz)) {
    lastVisualRender = now;
    if (activeObservatoryView === 'live') {
      const worldOnscreen = canvasVisible(el.world), brainOnscreen = canvasVisible(el.brain);
      if (worldOnscreen || brainOnscreen) {
        const model = selectedViewModel();
        const fx = cognitiveFx(model, lastSnapshot, viewWorld);
        if (worldOnscreen) {
          worldRenderer.overlayMode = el.worldFx?.value || 'BOTH';
          worldRenderer.draw(viewWorld, mode + (paused ? ' • PAUSED' : ''), mode === 'LEARN' ? session.curiosityTrail : [], fx, now);
        }
        if (brainOnscreen) {
          neuralRenderer.mode = el.brainView.value;
          neuralRenderer.draw(model, lastSnapshot, fx, now);
        }
        updateDecision();
      }
    } else if (activeObservatoryView === 'predict') {
      const predictWorldOnscreen = canvasVisible(el.predictWorld);
      if (predictWorldOnscreen) {
        const model = selectedViewModel();
        const fx = cognitiveFx(model, lastSnapshot, viewWorld);
        predictWorldRenderer.overlayMode = el.worldFx?.value || 'BOTH';
        predictWorldRenderer.draw(viewWorld, mode + (paused ? ' • PAUSED' : ''), mode === 'LEARN' ? session.curiosityTrail : [], fx, now);
        if (el.predictWorldMeta) el.predictWorldMeta.textContent = `${el.worldFx?.options[el.worldFx.selectedIndex]?.textContent || 'Attention + Echo'} • ${mode} • visual only`;
      }
      if (canvasVisible(el.curiosityCanvas) && now - lastCuriosityRender >= 1000 / Math.max(1, CONFIG.runtime.curiosityRenderHz)) {
        lastCuriosityRender = now;
        curiosityRenderer.draw(session.curiosity, session.lastCuriosity, now);
      }
    } else if (activeObservatoryView === 'memory' && canvasVisible(el.memoryCanvas) && now - lastMemoryRender >= 80) {
      lastMemoryRender = now;
      memoryRenderer.draw(memoryPoints, now, { mode, steps: session.totalSteps });
    } else if (activeObservatoryView === 'history' && canvasVisible(el.historyCanvas) && now - lastHistoryRender >= 220) {
      lastHistoryRender = now;
      historyRenderer.draw(session, now);
      if (el.historyMeta) el.historyMeta.textContent = `${session.validationHistory?.length || 0} validations • ${session.lineageHistory?.length || 0} lineages • ${session.hallOfFame?.length || 0} Hall`;
    }
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
  const curiosityPotential = mode === 'LEARN' ? Number(session.lastCuriosity?.potentialBonus ?? session.lastCuriosity?.bonus ?? 0) : 0;
  const curiosityApplied = mode === 'LEARN' ? Number(session.lastCuriosity?.appliedBonus ?? (session.curiosityRewardMode === 'reward' ? curiosityPotential : 0)) : 0;
  const curiositySuffix = mode !== 'LEARN'
    ? ' (OFF in evaluation/observation)'
    : session.curiosityRewardMode === 'observe'
      ? ` potential +${curiosityPotential.toFixed(4)} • APPLIED +0.0000`
      : ` +${curiosityApplied.toFixed(4)}`;
  el.rewardParts.textContent = `external  food ${(rp.food || 0).toFixed(3)}  approach ${(rp.approach || 0).toFixed(3)}  energy ${(rp.energy || 0).toFixed(3)}  wall ${(rp.wall || 0).toFixed(3)}  hazard ${(rp.hazard || 0).toFixed(3)}  death ${(rp.death || 0).toFixed(3)}  | curiosity${curiositySuffix}`;
  el.energy.value = viewWorld.agent.energy;
  el.energyText.textContent = `${Math.round(viewWorld.agent.energy * 100)}%`;
  const visibility = foodVisibilityDiagnostic(viewWorld);
  const losLabel = visibility.losBlocked ? 'diag LOS BLOCKED' : visibility.pathBlocked ? 'diag PATH BLOCKED' : 'diag DIRECT';
  el.worldMeta.textContent = `seed ${viewWorld.seed} • ${losLabel}`;
  el.viewBrainBadge.textContent = selectedSourceLabel();
}

function policyOriginText() {
  const lineage = session.learnerLineage;
  const parent = lineage?.parent;
  if (!parent) return 'native learner';
  if (parent.type === 'hall-of-fame') return `${parent.hallId || 'Hall'} @ ${Number(parent.sourceSteps || 0).toLocaleString()}`;
  if (parent.type === 'champion') return `Champion ${title(parent.category || 'balanced')} @ ${Number(parent.sourceSteps || 0).toLocaleString()}`;
  if (parent.type === 'curiosity-audit-origin') return `A/B origin ${parent.lineageId || 'learner'} @ ${Number(parent.sourceSteps || 0).toLocaleString()}`;
  if (parent.legacySchema) return `schema-${parent.legacySchema} migration`;
  return lineage.reason || 'derived learner';
}

function updateUI() {
  const uiStarted = performance.now();
  // Decision HUD is lightweight and must stay live even when expensive LIVE
  // canvases are sleeping off-screen. The normal LEARN UI cadence is 5 Hz,
  // which is fast enough for readable action/value feedback without waking any
  // canvas or changing simulation/training behavior.
  updateDecision();
  const m = session.metrics.at(-1);
  el.steps.textContent = session.totalSteps.toLocaleString();
  el.episodes.textContent = session.totalEpisodes.toLocaleString();
  el.ret.textContent = (m?.meanReturn ?? 0).toFixed(2);
  el.food.textContent = (m?.meanFood ?? 0).toFixed(2);
  el.entropy.textContent = m?.entropy?.toFixed(3) ?? '—';
  const train5 = trainRate.stats(5000);
  const train30 = trainRate.stats(30000);
  el.speed.textContent = mode === 'LEARN' && !paused ? formatRollingRate(train5) : '—';
  el.speed30.textContent = mode === 'LEARN' && !paused ? formatRollingRate(train30) : '—';
  el.lr.textContent = Number.isFinite(m?.learningRate) ? m.learningRate.toExponential(2) : session.trainer.learningRate.toExponential(2);
  el.kl.textContent = Number.isFinite(m?.maxEpochKL) ? m.maxEpochKL.toFixed(4) : '—';
  el.epochs.textContent = Number.isFinite(m?.epochsRun) ? `${m.epochsRun}${m.earlyStopped ? ' stop' : ''}` : '—';
  el.paramCount.textContent = `policy ${session.model.paramCount().toLocaleString()} + curiosity ${session.curiosity.paramCount().toLocaleString()} params`;
  el.curriculum.textContent = session.curriculum.current().name;
  const lv = session.validationHistory.at(-1);
  el.latestValidation.textContent = lv ? `${(lv.validation.score * 100).toFixed(1)}% @ ${lv.steps.toLocaleString()}${lv.confidenceAudit?.label === 'confirmed-regression' ? ' CONFIRMED' : lv.confidenceAudit?.label === 'likely-noise' ? ' NOISE?' : lv.balancedEvidence ? ' WATCH' : ''}` : '—';
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
  el.simSpeed.textContent = Number.isFinite(profile.simulationStepsPerSec) ? `${Math.round(profile.simulationStepsPerSec).toLocaleString()}/s raw` : '—';
  el.pageState.textContent = document.hidden ? 'BACKGROUND' : 'FOREGROUND';
  el.visualState.textContent = currentVisualState();
  const meterReady = train5.ready && train30.ready;
  el.meterState.textContent = meterReady ? 'ROLLING READY' : 'WARMING';
  el.ppoMs.textContent = Number.isFinite(profile.ppoMs) ? `${profile.ppoMs.toFixed(1)} ms` : '—';
  el.fps.textContent = `${browserFps.toFixed(0)}`;
  el.validationMs.textContent = Number.isFinite(profile.validationMs) && profile.validationMs > 0 ? `${profile.validationMs.toFixed(0)} ms` : '—';
  el.storageMs.textContent = lastStorageDurationMs > 0 ? `${lastStorageDurationMs.toFixed(0)} ms` : '—';

  const curiosityMetric = m?.curiosity || {};
  const curiosityLive = session.lastCuriosity;
  const episodeDiagnostics = curiosityMetric.episodeDiagnostics || session.curiosityDiagnosticsSummary();
  const audit = session.curiosityAuditSummary();
  const potentialBonus = Number(curiosityLive?.potentialBonus ?? curiosityLive?.bonus);
  const appliedBonus = Number(curiosityLive?.appliedBonus ?? (session.curiosityRewardMode === 'reward' ? potentialBonus : 0));
  el.curiosityError.textContent = Number.isFinite(curiosityLive?.error) ? curiosityLive.error.toFixed(4) : Number.isFinite(curiosityMetric.meanPredictionError) ? curiosityMetric.meanPredictionError.toFixed(4) : '—';
  el.curiosityNovelty.textContent = Number.isFinite(curiosityLive?.novelty) ? `${(curiosityLive.novelty * 100).toFixed(0)}%` : Number.isFinite(curiosityMetric.meanNovelty) ? `${(curiosityMetric.meanNovelty * 100).toFixed(0)}%` : '—';
  el.curiosityBonus.textContent = Number.isFinite(potentialBonus) ? `+${potentialBonus.toFixed(5)}` : '—';
  el.curiosityApplied.textContent = Number.isFinite(appliedBonus) ? `+${appliedBonus.toFixed(5)}` : '—';
  el.curiosityBudget.textContent = Number.isFinite(curiosityLive?.budgetRemaining) ? `${curiosityLive.budgetRemaining.toFixed(3)} / ${CONFIG.curiosity.maxEpisodeBonus.toFixed(2)}` : `max ${CONFIG.curiosity.maxEpisodeBonus.toFixed(2)}`;
  el.curiosityLoss.textContent = Number.isFinite(curiosityMetric.predictorLoss) ? curiosityMetric.predictorLoss.toFixed(5) : '—';
  el.curiosityParams.textContent = session.curiosity.paramCount().toLocaleString();
  el.curiosityInfluence.textContent = session.curiosityRewardMode === 'reward' ? 'REWARD ON' : 'OBSERVE ONLY';
  el.curiosityShare.textContent = Number.isFinite(curiosityMetric.rewardMagnitudeShare) ? `${(curiosityMetric.rewardMagnitudeShare * 100).toFixed(3)}%` : '—';
  el.curiosityResets.textContent = Number(episodeDiagnostics.budgetResets || 0).toLocaleString();
  el.curiosityBudgetUse.textContent = episodeDiagnostics.episodes
    ? `${(Number(episodeDiagnostics.meanBudgetUseFraction || 0) * 100).toFixed(1)}% • ${Number(episodeDiagnostics.meanBudgetUsed || 0).toFixed(4)}`
    : '—';
  el.curiosityExhaust.textContent = episodeDiagnostics.episodes
    ? `${(Number(episodeDiagnostics.budgetExhaustionRate || 0) * 100).toFixed(0)}%${Number.isFinite(episodeDiagnostics.meanExhaustionFraction) ? ` • @ ${(episodeDiagnostics.meanExhaustionFraction * 100).toFixed(0)}%` : ''}`
    : '—';
  if (curiosityLive?.mostSurprisingIndex != null) {
    const names = ['food x','food y','food dist','danger L','danger F','danger R','speed','turn rate','energy'];
    const name = names[curiosityLive.mostSurprisingIndex] || `sensor ${curiosityLive.mostSurprisingIndex}`;
    const influence = session.curiosityRewardMode === 'reward' ? `applied +${Number(appliedBonus || 0).toFixed(5)}` : `potential +${Number(potentialBonus || 0).toFixed(5)} • applied +0.00000`;
    el.curiosityInspect.textContent = `Most surprising now: ${name} • |prediction error| ${Number(curiosityLive.mostSurprisingError || 0).toFixed(4)} • ${influence}. Predictor learning stays on; evaluation curiosity stays OFF.`;
  }

  el.curiosityMode.value = session.curiosityRewardMode;
  el.curiosityMode.disabled = audit.active;
  el.startCuriosityAudit.disabled = audit.active;
  el.switchCuriosityAudit.disabled = !audit.active;
  el.endCuriosityAudit.disabled = !audit.active;
  el.curiosityAuditRole.textContent = audit.active ? `${String(audit.role || '—').toUpperCase()} • ${session.curiosityRewardMode === 'reward' ? 'reward on' : 'reward 0'}` : 'none';
  const target = Number(audit.targetStepsPerBranch || CONFIG.curiosityAudit.targetStepsPerBranch);
  el.curiosityControlProgress.textContent = audit.id ? `${Number(audit.controlProgress || 0).toLocaleString()} / ${target.toLocaleString()}` : '—';
  el.curiosityRewardProgress.textContent = audit.id ? `${Number(audit.curiosityProgress || 0).toLocaleString()} / ${target.toLocaleString()}` : '—';
  el.curiosityPairCheckpoint.textContent = Number.isFinite(audit.comparison?.checkpointSteps) ? Number(audit.comparison.checkpointSteps).toLocaleString() : '—';
  el.curiosityControlScore.textContent = Number.isFinite(audit.comparison?.controlScore) ? pct(audit.comparison.controlScore) : '—';
  el.curiosityRewardScore.textContent = Number.isFinite(audit.comparison?.curiosityScore) ? pct(audit.comparison.curiosityScore) : '—';
  el.switchCuriosityAudit.textContent = audit.active ? `Switch to ${audit.role === 'control' ? 'CURIOSITY' : 'CONTROL'}` : 'Switch A/B Branch';
  el.curiosityAuditResults.textContent = formatCuriosityAuditResults(audit);

  const stability = session.stabilitySummary();
  el.stabilityStatus.textContent = stability.status;
  el.stabilityPolicyLoss.textContent = fmt(stability.policyLoss, 4);
  el.stabilityValueLoss.textContent = fmt(stability.valueLoss, 4);
  el.stabilityExplained.textContent = fmt(stability.explainedVariance, 3);
  el.stabilityKl.textContent = fmt(stability.maxEpochKL, 5);
  el.stabilityClip.textContent = Number.isFinite(stability.clipFraction) ? `${(stability.clipFraction * 100).toFixed(1)}%` : '—';
  el.stabilityGrad.textContent = fmt(stability.gradientNormMax, 4);
  el.stabilityGradClip.textContent = Number.isFinite(stability.gradientClipFraction) ? `${(stability.gradientClipFraction * 100).toFixed(0)}%` : '—';
  el.stabilityParamDelta.textContent = Number.isFinite(stability.parameterRelativeDelta) ? stability.parameterRelativeDelta.toExponential(2) : '—';
  el.stabilityParamMax.textContent = Number.isFinite(stability.parameterMaxAbsDelta) ? stability.parameterMaxAbsDelta.toExponential(2) : '—';
  el.stabilityAdvantage.textContent = Number.isFinite(stability.advantageMean) && Number.isFinite(stability.advantageStd)
    ? `${stability.advantageMean.toFixed(3)} / ${stability.advantageStd.toFixed(3)}` : '—';
  el.stabilityRejected.textContent = Number(stability.rejectedUpdates || 0).toLocaleString();
  el.stabilityCompactSummary.textContent = formatStabilityCompact(stability);
  el.stabilityValidation.textContent = formatStabilityValidation(stability.lastValidationDeltas);
  el.stabilityEvents.textContent = formatStabilityEvents(session.stabilityEvents);

  const confidence = session.validationConfidenceHistory?.at(-1) || session.retentionStatus?.confidenceAudit || null;
  el.confidenceVerdict.textContent = confidence ? confidenceLabel(confidence.label) : '—';
  el.confidenceReference.textContent = Number.isFinite(Number(confidence?.referenceSteps)) ? Number(confidence.referenceSteps).toLocaleString() : 'awaiting baseline';
  el.confidenceSample.textContent = confidence?.paired ? `${confidence.episodesPerStage}/skill paired` : confidence?.label === 'baseline-established' ? 'baseline only' : 'not triggered';
  el.confidenceBalanced.textContent = formatConfidenceBalanced(confidence?.balanced, confidence?.rawBalancedDelta);
  el.confidenceCompactSummary.textContent = confidence ? `${confidenceLabel(confidence.label)} • paired fixed seeds • ${confidence.paired ? `${confidence.episodesPerStage}/skill` : 'cheap routine check'}` : 'waiting for validation baseline';
  el.confidenceEvidence.textContent = formatConfidenceEvidence(confidence);
  el.confidenceHistory.textContent = formatConfidenceHistory(session.validationConfidenceHistory);

  if (lastOrientationAudit) {
    const learner = lastOrientationAudit.learner?.result;
    const champion = lastOrientationAudit.champion?.result;
    const rear = learner?.bands?.rear;
    el.orientationSource.textContent = champion ? `Learner + Champion @ ${Number(lastOrientationAudit.champion.steps || 0).toLocaleString()}` : 'Learner only';
    el.orientationProtocol.textContent = learner?.protocol || '—';
    el.orientationRear.textContent = rear ? `${pct(rear.facingRate)} face • ${pct(rear.reachRate)} reach` : '—';
    el.orientationSpin.textContent = rear ? `${pct(rear.spinRate)} rear spin` : '—';
    el.orientationCompactSummary.textContent = rear ? `Learner rear face ${pct(rear.facingRate)} • spin ${pct(rear.spinRate)}${champion ? ` • Champion spin ${pct(champion.bands.rear.spinRate)}` : ''}` : 'rear-target turn control • observational';
  } else {
    el.orientationSource.textContent = '—';
    el.orientationProtocol.textContent = '—';
    el.orientationRear.textContent = '—';
    el.orientationSpin.textContent = '—';
    el.orientationCompactSummary.textContent = 'rear-target turn control • observational';
  }
  updateSpinTelemetryUI();

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
  el.researchCompactSummary.textContent = `${session.learnerLineage?.id || 'learner'} • Champion ${balancedBrain ? balancedBrain.savedAtSteps.toLocaleString() : 'none'} • Hall ${hall.length}`;
  el.curiosityCompactSummary.textContent = `${session.curiosityRewardMode === 'reward' ? 'reward ON' : 'observe-only'} • error ${Number.isFinite(curiosityLive?.error) ? curiosityLive.error.toFixed(4) : '—'} • share ${Number.isFinite(curiosityMetric.rewardMagnitudeShare) ? `${(curiosityMetric.rewardMagnitudeShare * 100).toFixed(2)}%` : '—'}${audit.active ? ` • A/B ${String(audit.role || '').toUpperCase()}` : ''}`;
  el.resultsCompactSummary.textContent = `learner ${lv ? pct(lv.validation.categoryScores?.balanced ?? lv.validation.score) : '—'} • Champion ${balancedBrain?.validation?.categoryScores ? pct(balancedBrain.validation.categoryScores.balanced) : '—'} • final holdout diagnostic only`;

  renderSkillRetention();
  if (activeObservatoryView === 'live' && canvasVisible(el.chart)) chartRenderer.draw(session.metrics);
  syncArchiveControls();
  lastUiDurationMs = performance.now() - uiStarted;
  el.uiMs.textContent = `${lastUiDurationMs.toFixed(1)} ms`;
}

function renderSkillRetention() {
  const validation = session.lastSkillValidation;
  if (!validation?.stageResults?.length) { el.skillRetention.textContent = 'Waiting for autonomous continual-learning validation…'; el.skillCompactSummary.textContent = 'waiting for validation'; return; }
  el.skillRetention.innerHTML = '';
  const alerts = new Map((session.retentionStatus?.alerts || session.retentionStatus?.forgetting || []).map(x => [x.stage, x]));
  for (const stage of validation.stageResults) {
    const cell = document.createElement('div');
    const alert = alerts.get(stage.stage);
    const best = session.skillBestRecords?.[stage.stage];
    const likelyNoise = alert?.confidenceLabel === 'likely-noise';
    const cls = alert ? (alert.confirmed ? ' forgetting confirmed' : ' forgetting watch') : '';
    cell.className = `skillCell${cls}`;
    const nowRange = `${pct(stage.skillCiLow)}–${pct(stage.skillCiHigh)}`;
    const bestScore = best?.score ?? stage.skillScore;
    const bestRange = best ? `${pct(best.ciLow)}–${pct(best.ciHigh)}` : nowRange;
    const tag = alert ? ` • ${alert.confirmed ? 'CONFIRMED' : likelyNoise ? 'LIKELY NOISE' : 'MEASURED DROP'} ${alert.severity.toUpperCase()} x${alert.streak}` : '';
    cell.innerHTML = `<b>${stage.name}</b><span>now ${pct(stage.skillScore)} [${nowRange}]</span><span>best ${pct(bestScore)} [${bestRange}]${tag}</span>`;
    el.skillRetention.append(cell);
  }
  el.skillCompactSummary.textContent = validation.stageResults.map(x => `${x.name.replace('Obstacle Avoidance','Obstacle').replace('Motor Nursery','Motor')} ${pct(x.skillScore)}`).join(' • ');
}
function formatCuriosityAuditResults(audit) {
  if (!audit?.id) return 'Audit idle. Start it only after loading/saving the learner you want to preserve as the experiment origin.';
  const lines = [];
  const origin = audit.results?.find(x => x.role === 'origin');
  lines.push(`${audit.id} • origin @ ${Number(audit.originSteps || 0).toLocaleString()} global steps${Number.isFinite(origin?.balancedScore) ? ` • baseline ${pct(origin.balancedScore)}` : ''}`);
  const checkpoints = [...new Set((audit.results || []).filter(x => x.role !== 'origin').map(x => Number(x.checkpointSteps) || 0))].sort((a, b) => a - b);
  for (const cp of checkpoints) {
    const c = audit.results.find(x => x.role === 'control' && Number(x.checkpointSteps) === cp);
    const q = audit.results.find(x => x.role === 'curiosity' && Number(x.checkpointSteps) === cp);
    lines.push(`${cp.toLocaleString()} branch steps • CONTROL ${Number.isFinite(c?.balancedScore) ? pct(c.balancedScore) : '—'} • CURIOSITY ${Number.isFinite(q?.balancedScore) ? pct(q.balancedScore) : '—'}`);
  }
  const cmp = audit.comparison;
  if (Number.isFinite(cmp?.delta)) {
    const pp = cmp.delta * 100;
    lines.push(`latest paired delta (curiosity − control): ${pp >= 0 ? '+' : ''}${pp.toFixed(1)} points • ${String(cmp.interpretation || '').replaceAll('-', ' ')}`);
  } else {
    lines.push('Paired verdict: waiting until both branches reach the same audit checkpoint.');
  }
  if (audit.completed) lines.push('Both branches reached the target. No winner was promoted automatically; review the paired results, then End Audit when ready.');
  else if (audit.active) lines.push(`ACTIVE ${String(audit.role || '').toUpperCase()}: ordinary Champion promotion is suspended until this audit ends.`);
  else lines.push('Audit ended. Results are preserved in the checkpoint; no branch was automatically promoted.');
  return lines.join('\n');
}

function pct(x) { return `${(Math.max(0, Math.min(1, Number(x) || 0)) * 100).toFixed(0)}%`; }
function fmt(x, digits = 3) { return Number.isFinite(Number(x)) ? Number(x).toFixed(digits) : '—'; }
function signedPoints(x) {
  if (!Number.isFinite(Number(x))) return '—';
  const points = Number(x) * 100;
  return `${points >= 0 ? '+' : ''}${points.toFixed(1)} pts`;
}
function formatStabilityCompact(stability) {
  const pieces = [stability.status];
  if (Number.isFinite(stability.maxEpochKL)) pieces.push(`KL ${stability.maxEpochKL.toFixed(4)}`);
  if (Number.isFinite(stability.clipFraction)) pieces.push(`clip ${(stability.clipFraction * 100).toFixed(0)}%`);
  if (Number.isFinite(stability.parameterRelativeDelta)) pieces.push(`Δw ${stability.parameterRelativeDelta.toExponential(1)}`);
  if (stability.events) pieces.push(`${stability.events} regression event${stability.events === 1 ? '' : 's'}`);
  return pieces.join(' • ');
}
function formatStabilityValidation(deltas) {
  if (!deltas) return 'Waiting for two comparable validation checkpoints to calculate per-skill deltas.';
  const skills = (deltas.skillDeltas || []).filter(x => Number.isFinite(x.delta)).map(x => `${x.name.replace('Obstacle Avoidance','Obstacle').replace('Motor Nursery','Motor')} ${signedPoints(x.delta)}`);
  return `latest validation change • balanced ${signedPoints(deltas.balancedDelta)}${skills.length ? ` • ${skills.join(' • ')}` : ''}`;
}
function formatStabilityEvents(events) {
  const rows = Array.isArray(events) ? events.slice(-4).reverse() : [];
  if (!rows.length) return 'No large validation regression event captured yet.';
  return rows.map(event => {
    const skill = event.largestSkillDrop && Number.isFinite(event.largestSkillDrop.delta)
      ? ` • ${event.largestSkillDrop.name.replace('Obstacle Avoidance','Obstacle')} ${signedPoints(event.largestSkillDrop.delta)}` : '';
    const ppo = event.ppo || {};
    const kl = Number.isFinite(ppo.maxEpochKL) ? ` • KL ${ppo.maxEpochKL.toFixed(4)}` : '';
    const clip = Number.isFinite(ppo.clipFraction) ? ` • clip ${(ppo.clipFraction * 100).toFixed(0)}%` : '';
    const delta = Number.isFinite(ppo.parameterRelativeDelta) ? ` • Δw ${ppo.parameterRelativeDelta.toExponential(1)}` : '';
    const confidence = event.validationConfidence?.label ? ` • ${confidenceLabel(event.validationConfidence.label)}` : '';
    return `${Number(event.atSteps || 0).toLocaleString()} • ${String(event.trigger || 'regression').replaceAll('-', ' ')} • balanced ${signedPoints(event.balancedDelta)}${skill}${kl}${clip}${delta}${confidence}`;
  }).join('\n');
}
function confidenceLabel(label) {
  const value = String(label || 'unavailable');
  if (value === 'confirmed-regression') return 'CONFIRMED REGRESSION';
  if (value === 'likely-noise') return 'LIKELY NOISE';
  if (value === 'measured-drop') return 'MEASURED DROP';
  if (value === 'baseline-established') return 'BASELINE ESTABLISHED';
  if (value === 'stable') return 'STABLE';
  return value.replaceAll('-', ' ').toUpperCase();
}
function formatConfidenceBalanced(balanced, rawDelta) {
  if (balanced?.classification) {
    const paired = Number.isFinite(Number(balanced.pairedDelta)) ? signedPoints(balanced.pairedDelta) : '—';
    const range = Number.isFinite(Number(balanced.ciLow)) && Number.isFinite(Number(balanced.ciHigh))
      ? `[${signedPoints(balanced.ciLow)} to ${signedPoints(balanced.ciHigh)}]` : '';
    return `${confidenceLabel(balanced.classification)} • ${paired} ${range}`.trim();
  }
  return Number.isFinite(Number(rawDelta)) ? `raw ${signedPoints(rawDelta)}` : '—';
}
function formatConfidenceEvidence(confidence) {
  if (!confidence) return 'Waiting for a same-lineage validation reference.';
  const lines = [];
  if (confidence.note) lines.push(confidence.note);
  if (confidence.balanced) {
    const x = confidence.balanced;
    lines.push(`Balanced • raw ${signedPoints(x.rawDelta)}${Number.isFinite(Number(x.pairedDelta)) ? ` • paired ${signedPoints(x.pairedDelta)} [${signedPoints(x.ciLow)} to ${signedPoints(x.ciHigh)}]` : ''} • ${confidenceLabel(x.classification)}`);
  }
  for (const x of confidence.skills || []) {
    lines.push(`${x.name} • raw ${signedPoints(x.rawDelta)}${Number.isFinite(Number(x.pairedDelta)) ? ` • paired ${signedPoints(x.pairedDelta)} [${signedPoints(x.ciLow)} to ${signedPoints(x.ciHigh)}]` : ''} • ${confidenceLabel(x.classification)}`);
  }
  if (!confidence.measured && confidence.label === 'stable') lines.push('No raw drop crossed the confirmation trigger; larger replay was skipped to protect phone throughput.');
  return lines.join('\n');
}
function formatConfidenceHistory(history) {
  const rows = Array.isArray(history) ? history.filter(x => x?.label && x.label !== 'stable').slice(-5).reverse() : [];
  if (!rows.length) return 'No measured-drop confidence events yet.';
  return rows.map(x => `${Number(x.atSteps || 0).toLocaleString()} • ${confidenceLabel(x.label)}${Number.isFinite(Number(x.referenceSteps)) ? ` • vs ${Number(x.referenceSteps).toLocaleString()}` : ''}`).join('\n');
}
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
    branches: branches.map(x => [x.id, x.frozenAtSteps, x.auditRole, x.curiosityRewardMode]),
    active: session.learnerLineage?.id,
    audit: [session.curiosityAudit?.active, session.auditRole, session.curiosityRewardMode],
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
      const auditTag = branch.auditRole ? ` • A/B ${branch.auditRole.toUpperCase()}` : '';
      option.textContent = `${branch.id} • frozen @ ${Number(branch.frozenAtSteps || 0).toLocaleString()} • ${Number(branch.learnerExperienceSteps || 0).toLocaleString()} branch steps${auditTag}`;
      el.branchSelect.append(option);
    }
    if ([...el.branchSelect.options].some(o => o.value === priorBranchValue)) el.branchSelect.value = priorBranchValue;
    else el.branchSelect.value = 'active';
  }

  const ref = selectedBrainRef();
  if (ref.type !== 'learner' && !ref.brain?.model) el.brainSource.value = 'latest';
  const currentRef = selectedBrainRef();
  const forkTarget = currentRef.type === 'learner' ? session.getArchiveBrain('balanced') : currentRef.brain;
  const auditActive = Boolean(session.curiosityAudit?.active);
  el.restoreBest.disabled = auditActive || session.archiveNeedsRebaseline || !forkTarget?.model;
  el.pinChampion.disabled = session.archiveNeedsRebaseline || currentRef.type !== 'champion' || !currentRef.brain?.model;
  el.switchBranch.disabled = auditActive || el.branchSelect.value === 'active' || !session.frozenLearners.some(x => x.id === el.branchSelect.value);
}

function suiteText(name, r) {
  const skillLines = r.stageResults.map(x => `  ${x.name}: ${pct(x.skillScore)} [${pct(x.skillCiLow)}–${pct(x.skillCiHigh)}]`).join('\n');
  return `${name}\nprotocol: ${r.protocol}\nepisodes: ${r.episodes} (${r.episodesPerStage}/skill)\ngeneralization score: ${pct(r.balancedScore)} [${pct(r.balancedCiLow)}–${pct(r.balancedCiHigh)}]\nmean return: ${r.meanReturn.toFixed(3)}\nmean food: ${r.meanFood.toFixed(3)}\nsurvival: ${(r.survivalRate * 100).toFixed(1)}%\nmean energy: ${r.meanEnergy.toFixed(3)}\nmean steps: ${r.meanSteps.toFixed(1)}\nskills:\n${skillLines}`;
}
function spinEventTableHtml(events) {
  if (!events.length) return '<div class="spinEmpty">No sustained spin event captured yet. Put MicroMind in OBSERVE and let it run normally; the recorder only watches and never feeds line-of-sight information back to the policy.</div>';
  const rows = [...events].reverse().map(e => `<tr><td>#${e.id}</td><td>${e.seed}</td><td>${e.stageName}</td><td>${e.triggerStep}</td><td>${e.cause}</td><td>${pct(e.blockedRate)}</td><td>${pct(e.forwardConeRate)}</td><td>${e.onsetBearingDeg.toFixed(0)}°</td><td>${e.onsetOmegaFraction.toFixed(2)}</td><td>${e.triggerRotations.toFixed(2)}</td><td>${e.triggerProgress.toFixed(3)}</td><td>${e.recovery}</td></tr>`).join('');
  return `<div class="spinTableWrap"><table class="resultsTable spinTable"><thead><tr><th>event</th><th>seed</th><th>stage</th><th>step</th><th>classified context</th><th>LOS blocked</th><th>diag front</th><th>bearing</th><th>ω/max</th><th>rotations</th><th>food Δ</th><th>recovery</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function occlusionAuditResultHtml(label, audit, meta = '') {
  const rows = audit.cases.map(row => {
    const geometry = row.initialLosBlocked ? 'LOS BLOCKED' : row.initialPathBlocked ? 'PATH BLOCKED / LOS CLEAR' : 'OPEN';
    const reachSteps = row.meanReachSteps > 0 ? row.meanReachSteps.toFixed(1) : '—';
    return `<tr><td>${row.caseName}</td><td>${geometry}</td><td>${pct(row.reachRate)}</td><td>${pct(row.spinRate)}</td><td>${reachSteps}</td><td>${row.meanMaxWindowRotations.toFixed(2)}</td><td>${row.meanTotalRotations.toFixed(2)}</td><td>${row.meanDistanceProgress.toFixed(3)}</td><td>${row.meanWallHits.toFixed(1)}</td></tr>`;
  }).join('');
  return `<div class="orientationResultBlock"><div class="orientationResultHead"><b>${label}</b><span>${meta}</span></div><div class="orientationBandLine">Same start pose, food position, zero recurrent state, and paired stochastic action RNG seed per trial. Only wall geometry differs.</div><div class="spinTableWrap"><table class="resultsTable spinTable"><thead><tr><th>case</th><th>geometry</th><th>reach</th><th>spin</th><th>steps→food</th><th>max rolling turns</th><th>total turns</th><th>food Δ</th><th>wall hits</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function updateSpinTelemetryUI() {
  if (!el.spinCompactSummary) return;
  const summary = liveSpinRecorder.summary();
  const recorderState = mode === 'OBSERVE' ? (paused ? 'PAUSED' : summary.active ? 'CAPTURING EVENT' : 'ARMED') : 'STANDBY';
  el.spinRecorderStatus.textContent = `${recorderState} • ${summary.samples.toLocaleString()} samples`;
  el.spinEventCount.textContent = `${summary.events}${summary.active ? ' + active' : ''}`;
  el.spinBlockedRate.textContent = summary.events ? `${pct(summary.blockedAtOnsetRate)} event onsets` : '—';
  el.spinLatestCause.textContent = summary.latestCause;
  el.spinCompactSummary.textContent = `${recorderState} • ${summary.events} captured${summary.events ? ` • blocked ${pct(summary.blockedAtOnsetRate)}` : ''}`;
  el.spinResults.innerHTML = spinEventTableHtml(liveSpinRecorder.events);

  if (lastOcclusionAudit) {
    let html = `<div class="orientationAuditIntro">Controlled paired geometry audit. OPEN has no wall. CLEARANCE keeps the food centerline visible but places a wall inside the agent's direct travel corridor. OCCLUDED crosses the food centerline itself. The LOS/path labels are diagnostics only and are never policy inputs.</div>`;
    html += occlusionAuditResultHtml(`LEARNER • ${lastOcclusionAudit.lineageId}`, lastOcclusionAudit.learner.result, `@ ${Number(lastOcclusionAudit.learner.steps).toLocaleString()} steps`);
    if (lastOcclusionAudit.champion) html += occlusionAuditResultHtml('CHAMPION BALANCED', lastOcclusionAudit.champion.result, `@ ${Number(lastOcclusionAudit.champion.steps).toLocaleString()} steps`);
    html += `<div class="orientationAuditNote">A telemetry “spin” requires at least ${SPIN_TELEMETRY.spinRotationThreshold.toFixed(2)} rolling rotations while food-distance progress stays ≤ ${SPIN_TELEMETRY.poorProgressDistance.toFixed(3)}. It is a diagnostic event definition, not a policy failure declaration.</div>`;
    el.spinOcclusionResults.innerHTML = html;
  }
}

async function runOcclusionConflictAudit() {
  paused = true;
  el.pause.textContent = 'Resume';
  el.spinOcclusionRun.disabled = true;
  setStatus(`Running read-only occlusion conflict audit: ${OCCLUSION_AUDIT.cases.length} geometries × ${OCCLUSION_AUDIT.trialsPerCase} paired seeded trials. Training is paused.`);
  await yieldUI();
  try {
    const learnerBefore = JSON.stringify(session.model.serialize());
    const learnerResult = runOcclusionAudit(session.model);
    if (learnerBefore !== JSON.stringify(session.model.serialize())) throw new Error('occlusion audit safety check failed: Learner weights changed');
    await yieldUI();
    const result = {
      ranAtSteps: session.totalSteps,
      lineageId: session.learnerLineage?.id || 'learner',
      learner: { steps: session.totalSteps, result: learnerResult },
      champion: null,
    };
    const champion = session.getArchiveBrain('balanced');
    if (champion?.model) {
      const model = new RecurrentActorCritic(1);
      model.restore(champion.model);
      const before = JSON.stringify(model.serialize());
      const championResult = runOcclusionAudit(model);
      if (before !== JSON.stringify(model.serialize())) throw new Error('occlusion audit safety check failed: Champion weights changed');
      result.champion = { steps: champion.savedAtSteps, result: championResult };
    }
    lastOcclusionAudit = result;
    updateSpinTelemetryUI();
    const open = learnerResult.cases.find(x => x.caseName === 'open');
    const blocked = learnerResult.cases.find(x => x.caseName === 'occluded');
    setStatus(`Occlusion audit complete. Learner OPEN reach ${pct(open?.reachRate || 0)} / spin ${pct(open?.spinRate || 0)}; OCCLUDED reach ${pct(blocked?.reachRate || 0)} / spin ${pct(blocked?.spinRate || 0)}. No learning, physics, sensor, reward, checkpoint, or Champion state changed.`);
  } catch (e) {
    setStatus(`Occlusion audit failed safely: ${e.message}`);
  } finally {
    el.spinOcclusionRun.disabled = false;
  }
}

function orientationResultHtml(label, audit, meta = '') {
  const rows = audit.bearings.map(row => {
    const face = pct(row.facingRate);
    const spin = pct(row.spinRate);
    const reach = pct(row.reachRate);
    const faceSteps = Number.isFinite(row.meanFacingSteps) ? row.meanFacingSteps.toFixed(1) : '—';
    const rotations = row.meanRotations.toFixed(2);
    const brakeTurn = pct(row.brakeWhileTurningRate);
    const first = `${row.initialDominantAction} ${(row.initialDominantProbability * 100).toFixed(0)}%`;
    return `<tr><td>${row.bearingDeg > 0 ? '+' : ''}${row.bearingDeg}°</td><td>${face}</td><td>${spin}</td><td>${reach}</td><td>${faceSteps}</td><td>${rotations}</td><td>${brakeTurn}</td><td>${first}</td></tr>`;
  }).join('');
  const rear = audit.bands.rear;
  return `<div class="orientationResultBlock"><div class="orientationResultHead"><b>${label}</b><span>${meta}</span></div><div class="orientationBandLine">rear ≥135° • face ${pct(rear.facingRate)} • spin ${pct(rear.spinRate)} • reach ${pct(rear.reachRate)} • mean turn ${rear.meanRotations.toFixed(2)} rotations • BRAKE while already turning ${pct(rear.brakeWhileTurningRate)}</div><div class="orientationTableWrap"><table class="resultsTable orientationTable"><thead><tr><th>bearing</th><th>face</th><th>spin</th><th>reach</th><th>steps→face</th><th>rotations</th><th>brake@ω</th><th>initial argmax</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

async function runRearTargetAudit() {
  paused = true;
  el.pause.textContent = 'Resume';
  el.orientationRun.disabled = true;
  setStatus(`Running read-only rear-target rotation audit: ${ORIENTATION_AUDIT.bearingsDeg.length} bearings × ${ORIENTATION_AUDIT.trialsPerBearing} seeded trials. Training is paused so the tested weights cannot move.`);
  await yieldUI();
  try {
    const learnerBefore = JSON.stringify(session.model.serialize());
    const learnerResult = runOrientationAudit(session.model);
    const learnerAfter = JSON.stringify(session.model.serialize());
    if (learnerBefore !== learnerAfter) throw new Error('audit safety check failed: Learner weights changed');
    await yieldUI();

    const result = {
      ranAtSteps: session.totalSteps,
      lineageId: session.learnerLineage?.id || 'learner',
      learner: { steps: session.totalSteps, result: learnerResult },
      champion: null,
    };
    const champion = session.getArchiveBrain('balanced');
    if (champion?.model) {
      const model = new RecurrentActorCritic(1);
      model.restore(champion.model);
      const before = JSON.stringify(model.serialize());
      const championResult = runOrientationAudit(model);
      if (before !== JSON.stringify(model.serialize())) throw new Error('audit safety check failed: Champion weights changed');
      result.champion = { steps: champion.savedAtSteps, result: championResult };
    }
    lastOrientationAudit = result;
    let html = `<div class="orientationAuditIntro">Controlled empty-arena diagnostic. Food direction remains present in the existing policy observation even when the target is behind the agent; this audit changes no sensors, rewards, physics, recurrent state, PPO, curriculum, checkpoints, or Champion state. Each trial starts from rest with zero recurrent state and uses the real stochastic policy plus real angular dynamics.</div>`;
    html += orientationResultHtml(`LEARNER • ${result.lineageId}`, learnerResult, `@ ${Number(session.totalSteps).toLocaleString()} steps`);
    if (result.champion) html += orientationResultHtml('CHAMPION BALANCED', result.champion.result, `@ ${Number(result.champion.steps).toLocaleString()} steps`);
    html += `<div class="orientationAuditNote">Interpretation aid: “spin” means cumulative angular travel reached the audit threshold (${ORIENTATION_AUDIT.spinRotationThreshold.toFixed(2)} rotations), not that the code declared the policy broken. “brake@ω” measures how often BRAKE was selected while angular speed was already substantial; current world physics applies no special angular braking on BRAKE beyond the same angular drag every action receives.</div>`;
    el.orientationResults.innerHTML = html;
    updateUI();
    const rear = learnerResult.bands.rear;
    setStatus(`Rear-target audit complete. Learner rear-facing success ${pct(rear.facingRate)}, spin incidence ${pct(rear.spinRate)}, reach ${pct(rear.reachRate)}. Training remains paused; no learning or save state changed.`);
  } catch (e) {
    setStatus(`Rear-target audit failed safely: ${e.message}`);
  } finally {
    el.orientationRun.disabled = false;
  }
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
  text += `\n\n${diagnostic.text}\nFINAL HOLDOUT IS DIAGNOSTIC ONLY — curiosity reward is OFF; no weights, optimizer, Champion archive, curriculum, validation history, or promotion state were changed. Repeatedly consulting this set can still bias human decisions, so use it sparingly.`;
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

document.addEventListener('visibilitychange', () => {
  // Safari may suspend timers/rAF while backgrounded. Reset wall-clock rate windows
  // on either transition so foreground throughput is never contaminated by a
  // background pause or by the first burst after the page resumes.
  resetTrainRateMeter();
  maybeUpdateUI(true);
});

el.newBrain.addEventListener('click', () => { if (confirm('Create a new untrained brain? Current unsaved progress will be replaced.')) resetBrain(); });
el.learn.addEventListener('click', () => setMode('LEARN'));
el.observe.addEventListener('click', () => setMode('OBSERVE'));
el.probe.addEventListener('click', () => setMode('PROBE'));
el.pause.addEventListener('click', () => { paused = !paused; resetTrainRateMeter(); el.pause.textContent = paused ? 'Resume' : 'Pause'; updateSpinTelemetryUI(); setStatus(paused ? 'Paused. Neural state remains inspectable.' : 'Resumed.'); });
el.brainView.addEventListener('change', () => { neuralRenderer.mode = el.brainView.value; });
el.worldFx?.addEventListener('change', () => {
  worldRenderer.overlayMode = el.worldFx.value;
  if (el.worldFxBadge) el.worldFxBadge.textContent = el.worldFx.options[el.worldFx.selectedIndex]?.textContent?.toUpperCase() || 'VISUAL';
});
el.brainSource.addEventListener('change', () => {
  const ref = selectedBrainRef();
  if (ref.type !== 'learner' && !ref.brain?.model) { el.brainSource.value = 'latest'; return; }
  if (mode !== 'LEARN') { liveSpinRecorder.reset(); resetViewState(); }
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
el.curiosityMode.addEventListener('change', () => {
  try {
    session.setCuriosityRewardMode(el.curiosityMode.value);
    updateUI();
    setStatus(session.curiosityRewardMode === 'reward'
      ? 'Curiosity reward influence ON. The predictor learns and its bounded intrinsic bonus is added to PPO training reward.'
      : 'Curiosity OBSERVE-ONLY. The predictor still learns and all curiosity diagnostics still run, but PPO receives exactly zero intrinsic reward.');
  } catch (e) {
    el.curiosityMode.value = session.curiosityRewardMode;
    setStatus(`Curiosity mode change blocked: ${e.message}`);
  }
});
el.startCuriosityAudit.addEventListener('click', async () => {
  if (session.curiosityAudit?.active) return;
  if (!confirm(`Start the controlled Curiosity A/B audit from the CURRENT Learner @ ${session.totalSteps.toLocaleString()} steps? The current lineage will be frozen as the recoverable origin. Two matched descendants will be created: CONTROL (predictor learns, curiosity reward = 0) and CURIOSITY (current v0.1.3 reward). Each branch runs ${CONFIG.curiosityAudit.targetStepsPerBranch.toLocaleString()} steps with fixed read-only audit evaluations.`)) return;
  paused = true;
  el.pause.textContent = 'Resume';
  try {
    const existing = await loadCheckpointRecord('latest');
    const storedSteps = Number(existing?.snapshot?.totalSteps || 0);
    if (existing && storedSteps > session.totalSteps) {
      setStatus(`A/B audit NOT started: your protected Manual Save has ${storedSteps.toLocaleString()} steps, newer than the current ${session.totalSteps.toLocaleString()}-step learner. Load Manual first so we do not audit the wrong brain.`);
      await refreshSaveSlots();
      return;
    }
    const saveStarted = performance.now();
    await saveCheckpoint(session.snapshot(), 'latest');
    await persistHallOfFame();
    lastStorageDurationMs = performance.now() - saveStarted;
    setStatus('Preserved the exact pre-audit learner. Building matched CONTROL and CURIOSITY descendants and running the read-only audit baseline…');
    await yieldUI();
    const audit = session.startCuriosityAudit();
    await saveCheckpoint(session.snapshot(), 'autosave');
    await refreshSaveSlots();
    controlSignature = '';
    el.brainSource.value = 'latest';
    resetViewState();
    updateUI();
    setStatus(`Curiosity A/B audit ${audit.id} ready. CONTROL is active first with intrinsic reward exactly 0; predictor learning remains ON. Training is paused—press Resume or Learn when ready.`);
  } catch (e) {
    setStatus(`Curiosity A/B audit start failed: ${e.message}`);
  }
});
el.switchCuriosityAudit.addEventListener('click', async () => {
  const audit = session.curiosityAuditSummary();
  if (!audit.active) return;
  const nextRole = audit.role === 'control' ? 'curiosity' : 'control';
  const currentProgress = audit.role === 'control' ? audit.controlProgress : audit.curiosityProgress;
  if (currentProgress < audit.targetStepsPerBranch && !confirm(`${String(audit.role || '').toUpperCase()} has only ${Number(currentProgress).toLocaleString()} / ${Number(audit.targetStepsPerBranch).toLocaleString()} audit steps. Switch early to ${nextRole.toUpperCase()}? You can switch back later; the current branch will be frozen exactly.`)) return;
  paused = true;
  el.pause.textContent = 'Resume';
  try {
    const event = session.switchCuriosityAuditBranch(nextRole);
    resetTrainRateMeter();
    await saveCheckpoint(session.snapshot(), 'autosave');
    controlSignature = '';
    el.brainSource.value = 'latest';
    resetViewState();
    updateUI();
    setStatus(`A/B switched to ${nextRole.toUpperCase()} (${event.lineageId}). Branch state, optimizer, predictor and RNG continuation restored. Training is paused—press Resume or Learn when ready.`);
  } catch (e) { setStatus(`A/B branch switch failed: ${e.message}`); }
});
el.endCuriosityAudit.addEventListener('click', async () => {
  const audit = session.curiosityAuditSummary();
  if (!audit.active) return;
  if (!confirm('End the curiosity A/B audit? No branch will be promoted automatically. The currently active policy remains the Learner and ordinary validation will be scheduled before continued training.')) return;
  paused = true;
  el.pause.textContent = 'Resume';
  try {
    session.endCuriosityAudit();
    await saveCheckpoint(session.snapshot(), 'autosave');
    controlSignature = '';
    updateUI();
    setStatus('Curiosity A/B audit ended. Results are preserved; no automatic winner was selected. Ordinary validation is scheduled for the active Learner before long training resumes.');
  } catch (e) { setStatus(`Could not end curiosity audit: ${e.message}`); }
});

el.branchSelect.addEventListener('change', syncArchiveControls);
el.switchBranch.addEventListener('click', () => {
  const id = el.branchSelect.value;
  if (!id || id === 'active') return;
  if (!confirm(`Switch active training to frozen Learner ${id}? The current Learner will be frozen first. Champions and Hall of Fame are shared; global experience age remains monotonic.`)) return;
  try {
    const event = session.switchToFrozenLearner(id);
    resetTrainRateMeter();
    archiveModelCache = null;
    archiveModelCacheKey = null;
    controlSignature = '';
    el.brainSource.value = 'latest';
    resetViewState();
    updateUI();
    setStatus(`Active Learner switched to ${event.lineageId}. Previous Learner ${event.preservedLineageId} is frozen. Immediate validation is scheduled before long training continues.`);
  } catch (e) { setStatus(`Branch switch failed: ${e.message}`); }
});
el.orientationRun?.addEventListener('click', runRearTargetAudit);
el.spinOcclusionRun?.addEventListener('click', runOcclusionConflictAudit);
el.spinClear?.addEventListener('click', () => {
  liveSpinRecorder.reset();
  updateSpinTelemetryUI();
  setStatus('Live spin telemetry cleared. Recorder remains read-only and will arm automatically in OBSERVE mode.');
});
el.test.addEventListener('click', runUnseen);
el.compare.addEventListener('click', compareBrains);
el.brain.addEventListener('pointerdown', e => { const text = neuralRenderer.inspectAt(e.clientX, e.clientY); if (text) el.inspect.textContent = text; });
el.curiosityCanvas.addEventListener('pointerdown', e => { const text = curiosityRenderer.inspectAt(e.clientX, e.clientY); if (text) el.curiosityInspect.textContent = text; });
el.observatoryNav?.addEventListener('click', e => {
  const button = e.target.closest('[data-view]');
  if (!button) return;
  setObservatoryView(button.dataset.view);
});
el.memoryCanvas?.addEventListener('pointerdown', e => {
  const text = memoryRenderer.inspectAt(e.clientX, e.clientY, memoryPoints);
  if (text) el.memoryInspect.textContent = text;
});
el.historyCanvas?.addEventListener('pointerdown', e => {
  const text = historyRenderer.inspectAt(e.clientX, e.clientY);
  if (text) el.historyInspect.textContent = text;
});
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

setObservatoryView('live');
resetMemoryConstellation();
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
