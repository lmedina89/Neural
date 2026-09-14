import { BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { evaluateFullRetentionSuite } from '../evaluation/evaluator.js';
import { RecurrentActorCritic } from './model.js';
import { PPOTrainer } from './ppo.js';
import { computeGAE } from './rollout.js';

const ARCHIVE_CATEGORIES = ['balanced', 'overall', 'forager', 'survivor', 'efficiency'];

export class TrainingSession {
  constructor({ seed = 1337, envCount = CONFIG.runtime.trainEnvs, autoCurriculum = true } = {}) {
    this.seed = seed;
    this.autoCurriculum = autoCurriculum;
    this.model = new RecurrentActorCritic(seed);
    this.trainer = new PPOTrainer(this.model, seed ^ 0x9e3779b9);
    this.curriculum = new CurriculumManager(0);
    this.actionRng = new PRNG(seed ^ 0xa5a5a5a5);
    this.totalSteps = 0;
    this.totalEpisodes = 0;
    this.envSeedCursor = 0;
    this.metrics = [];
    this.episodeHistory = [];
    this.milestones = new Map();
    this.nextMilestoneStep = nextHistoricalMilestoneAfter(0);
    this.bestBrain = null;
    this.bestArchive = {};
    this.validationHistory = [];
    this.legacyValidationHistory = [];
    this.lastValidationStep = -1;
    this.nextValidationStep = nextValidationAfter(0);
    this.lastSkillValidation = null;
    this.skillBestScores = Array(CURRICULUM.length).fill(0);
    this.skillBestRecords = emptySkillBestRecords();
    this.skillRegressionStreaks = Array(CURRICULUM.length).fill(0);
    this.balancedRegressionStreak = 0;
    this.retentionStatus = defaultRetentionStatus();
    this.pendingLegacyBest = null;
    this.pendingArchiveMigration = [];
    this.archiveNeedsRebaseline = false;
    this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
    this.promotionCandidates = {};
    this.lineageCounter = 0;
    this.learnerLineage = makeRootLineage(seed, 0);
    this.lineageHistory = [this.learnerLineage];
    this.rehearsalEpisodeHistory = [];
    this.rollbackHistory = [];
    this.hallOfFame = [];
    this.nextHallOfFameId = 1;
    this.frozenLearners = [];
    this.learnerExperienceSteps = 0;
    this.lastValidationDurationMs = 0;
    this.envs = [];
    this.obs = [];
    this.hidden = [];
    for (let i = 0; i < envCount; i++) this.addEnv(i);
    this.saveMilestone(0);
  }

  trainingMix(stage = this.curriculum.stage) {
    const row = CONFIG.continual.rehearsalMix[Math.max(0, Math.min(CURRICULUM.length - 1, stage))] || [];
    const values = CURRICULUM.map((_, i) => Math.max(0, Number(row[i]) || 0));
    const sum = values.reduce((a, b) => a + b, 0);
    if (!(sum > 0)) return CURRICULUM.map((_, i) => i === stage ? 1 : 0);
    return values.map(x => x / sum);
  }

  chooseTrainingStage(cursor = this.envSeedCursor) {
    const mix = this.trainingMix();
    const rng = new PRNG(domainSeed(`rehearsal:${this.seed}:curriculum:${this.curriculum.stage}`, cursor));
    let r = rng.next();
    for (let i = 0; i < mix.length; i++) {
      r -= mix[i];
      if (r <= 0) return i;
    }
    return this.curriculum.stage;
  }

  addEnv(id) {
    const cursor = this.envSeedCursor++;
    const stageIndex = this.chooseTrainingStage(cursor);
    const seed = domainSeed(`train:${this.seed}:stage:${stageIndex}`, cursor);
    const env = new World(seed, CURRICULUM[stageIndex]);
    this.envs[id] = env;
    this.obs[id] = env.observe();
    this.hidden[id] = this.model.zeroHidden();
  }

  resetEnv(id) {
    const cursor = this.envSeedCursor++;
    const stageIndex = this.chooseTrainingStage(cursor);
    const seed = domainSeed(`train:${this.seed}:stage:${stageIndex}`, cursor);
    const env = new World(seed, CURRICULUM[stageIndex]);
    this.envs[id] = env;
    this.obs[id] = env.observe();
    this.hidden[id] = this.model.zeroHidden();
  }

  recentRehearsalMix() {
    const counts = Array(CURRICULUM.length).fill(0);
    for (const stage of this.rehearsalEpisodeHistory) if (Number.isInteger(stage) && counts[stage] != null) counts[stage]++;
    const total = counts.reduce((a, b) => a + b, 0);
    return { counts, fractions: counts.map(x => total ? x / total : 0), total, target: this.trainingMix() };
  }

  trainRollout(steps = CONFIG.runtime.rolloutSteps) {
    const transitions = [];
    const completed = [];
    let curriculumEvent = null;
    let validationMs = 0;

    const preValidationStarted = performanceNow();
    const preValidation = this.totalSteps >= this.nextValidationStep && this.lastValidationStep !== this.totalSteps ? this.runValidation() : null;
    if (preValidation) validationMs += performanceNow() - preValidationStarted;

    const simulationStarted = performanceNow();
    for (let s = 0; s < steps; s++) {
      for (let i = 0; i < this.envs.length; i++) {
        const obs = this.obs[i];
        const hPrev = this.hidden[i];
        const act = this.model.act(obs, hPrev, this.actionRng, false, false);
        const result = this.envs[i].step(act.action);
        transitions.push({
          env: i,
          obs: new Float64Array(obs),
          hPrev: new Float64Array(hPrev),
          action: act.action,
          logProb: act.logProb,
          value: act.value,
          reward: result.reward,
          done: result.done,
        });
        this.totalSteps++;
        this.obs[i] = result.obs;
        this.hidden[i] = act.hidden;
        if (result.done) {
          const info = result.info;
          completed.push(info);
          this.totalEpisodes++;
          this.episodeHistory.push(info);
          if (this.episodeHistory.length > 200) this.episodeHistory.shift();
          this.rehearsalEpisodeHistory.push(info.stageId);
          if (this.rehearsalEpisodeHistory.length > CONFIG.continual.recentMixWindow) this.rehearsalEpisodeHistory.shift();
          if (this.autoCurriculum && info.stageId === this.curriculum.stage) {
            const event = this.curriculum.noteEpisode(info, this.totalEpisodes, {
              promotionAllowed: true,
              gateReason: 'autonomous current-stage performance',
            });
            if (event) curriculumEvent = event;
          }
          this.resetEnv(i);
        }
      }
    }
    const simulationMs = performanceNow() - simulationStarted;
    this.learnerExperienceSteps += transitions.length;

    const advantageStarted = performanceNow();
    const lastValues = new Map();
    for (let i = 0; i < this.envs.length; i++) lastValues.set(i, this.model.forward(this.obs[i], this.hidden[i], false).value);
    const { advantages, returns } = computeGAE(transitions, lastValues);
    const advantageMs = performanceNow() - advantageStarted;

    const ppoStarted = performanceNow();
    const ppo = this.trainer.update(transitions, advantages, returns, { trainingStep: this.totalSteps });
    const ppoMs = performanceNow() - ppoStarted;
    const coreMs = Math.max(0.001, simulationMs + advantageMs + ppoMs);
    const recent = this.episodeHistory.slice(-40);
    const mean = key => recent.length ? recent.reduce((sum, x) => sum + (x[key] || 0), 0) / recent.length : 0;

    const bookkeepingStarted = performanceNow();
    this.maybeSaveMilestones();
    const forceValidation = Boolean(curriculumEvent && curriculumEvent.reason !== 'promotion-blocked');
    const bookkeepingBeforeValidationMs = performanceNow() - bookkeepingStarted;
    const postValidationStarted = performanceNow();
    const postValidation = this.maybeAutoValidate(forceValidation);
    if (postValidation) validationMs += performanceNow() - postValidationStarted;
    const validation = postValidation || preValidation;
    const trainingStepsPerSec = transitions.length / (coreMs / 1000);
    const simulationStepsPerSec = transitions.length / Math.max(0.001, simulationMs / 1000);
    const metric = {
      steps: this.totalSteps,
      episodes: this.totalEpisodes,
      curriculum: this.curriculum.stage,
      curriculumName: CURRICULUM[this.curriculum.stage].name,
      curriculumEvent,
      meanReturn: mean('totalReward'),
      meanFood: mean('food'),
      meanEnergy: mean('energy'),
      hazardHits: mean('hazardHits'),
      wallHits: mean('wallHits'),
      throughput: trainingStepsPerSec,
      simulationThroughput: simulationStepsPerSec,
      profile: {
        simulationMs,
        advantageMs,
        ppoMs,
        bookkeepingMs: bookkeepingBeforeValidationMs,
        validationMs,
        coreMs,
        trainingStepsPerSec,
        simulationStepsPerSec,
        ppoUpdatesPerSec: 1000 / Math.max(0.001, ppoMs),
      },
      validation: validation ? compactValidation(validation) : null,
      retentionAlertCount: this.retentionStatus.forgetting.length,
      skillScores: this.lastSkillValidation?.stageResults?.map(x => x.skillScore) || [],
      rehearsalMix: this.recentRehearsalMix(),
      learnerLineage: { ...this.learnerLineage },
      learnerExperienceSteps: this.learnerExperienceSteps,
      ...ppo,
    };
    this.metrics.push(metric);
    if (this.metrics.length > CONFIG.runtime.chartPoints) this.metrics.shift();
    return { metric, completed, transitions: transitions.length, validation, curriculumEvent };
  }

  promotionGate() {
    return { allowed: true, reason: 'autonomous curriculum; retention metrics are observational' };
  }

  maybeSaveMilestones() {
    if (this.totalSteps < this.nextMilestoneStep) return;
    const label = this.nextMilestoneStep;
    this.saveMilestone(label);
    this.nextMilestoneStep = nextHistoricalMilestoneAfter(this.totalSteps);
  }

  saveMilestone(label = this.totalSteps) {
    this.milestones.set(label, {
      label,
      savedAtSteps: this.totalSteps,
      model: this.model.serialize(),
      curriculum: this.curriculum.serialize(),
      lineage: structuredLineage(this.learnerLineage),
      metric: this.metrics.at(-1) || null,
    });
  }

  maybeAutoValidate(force = false) {
    if (!force && this.totalSteps < this.nextValidationStep) return null;
    if (this.lastValidationStep === this.totalSteps) return null;
    return this.runValidation();
  }

  runValidation() {
    const validation = evaluateFullRetentionSuite(this.model, {
      episodesPerStage: CONFIG.validation.episodesPerStage,
      seedBase: CONFIG.validation.seedBase,
      deterministic: false,
    });

    // Re-evaluate legacy protected brains before comparing them with the learner.
    // Migration is archive preservation, not a learner intervention.
    let migratedArchive = [];
    if (this.archiveNeedsRebaseline || this.pendingArchiveMigration.length || this.pendingLegacyBest?.model) {
      const migrationSources = uniqueCandidates([
        ...this.pendingArchiveMigration,
        ...(this.pendingLegacyBest?.model ? [this.pendingLegacyBest] : []),
      ]);
      const reevaluated = [];
      for (const base of migrationSources) {
        if (!base?.model) continue;
        const legacyModel = new RecurrentActorCritic(1);
        legacyModel.restore(base.model);
        const legacyValidation = evaluateFullRetentionSuite(legacyModel, {
          episodesPerStage: CONFIG.validation.episodesPerStage,
          seedBase: CONFIG.validation.seedBase,
          deterministic: false,
        });
        reevaluated.push(this.makeCandidate(legacyValidation, 'recalibrated-legacy-champion', base));
      }
      this.bestArchive = {};
      this.bestBrain = null;
      this.skillBestRecords = emptySkillBestRecords();
      this.skillBestScores = Array(CURRICULUM.length).fill(0);
      for (const candidate of reevaluated) {
        this.seedArchiveCandidate(candidate);
        updateSkillBestRecords(this.skillBestRecords, candidate.validation.stageResults, candidate.savedAtSteps);
      }
      syncSkillBestScores(this);
      migratedArchive = reevaluated.map(x => ({ savedAtSteps: x.savedAtSteps, categoryScores: x.validation.categoryScores }));
      this.pendingArchiveMigration = [];
      this.pendingLegacyBest = null;
      this.archiveNeedsRebaseline = false;
      this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
      this.promotionCandidates = {};
    }

    const priorBalanced = this.getArchiveBrain('balanced');
    const priorBalancedScore = priorBalanced?.validation?.categoryScores?.balanced ?? null;
    const skillAssessment = assessSkillRetention(
      validation.stageResults,
      this.skillBestRecords,
      this.skillRegressionStreaks,
    );
    this.skillRegressionStreaks = skillAssessment.streaks;

    const currentBalanced = validation.categoryScores.balanced;
    const deltaFromBest = priorBalancedScore == null ? null : currentBalanced - priorBalancedScore;
    const balancedEvidence = priorBalancedScore != null && balancedRegressionEvidence(validation, priorBalanced.validation);
    this.balancedRegressionStreak = balancedEvidence ? this.balancedRegressionStreak + 1 : 0;
    const balancedConfirmed = balancedEvidence && this.balancedRegressionStreak >= CONFIG.validation.balancedConfirmationCount;

    const skillImprovements = detectSkillImprovements(validation.stageResults, this.skillBestRecords);
    const confirmedCatastrophic = skillAssessment.alerts.filter(x => x.confirmed && x.severity === 'catastrophic');
    const confirmedAlerts = skillAssessment.alerts.filter(x => x.confirmed);
    let interpretation = 'healthy';
    if (balancedConfirmed || confirmedCatastrophic.length >= 2) interpretation = 'confirmed-regression-observed';
    else if (confirmedAlerts.length && !balancedEvidence && skillImprovements.length) interpretation = 'confirmed-specialization';
    else if (confirmedAlerts.length) interpretation = 'confirmed-skill-regression-observed';
    else if (skillAssessment.alerts.length && !balancedEvidence && skillImprovements.length) interpretation = 'specialization-watch';
    else if (skillAssessment.alerts.length) interpretation = 'skill-regression-watch';
    else if (balancedEvidence) interpretation = 'balanced-regression-watch';

    const currentCandidate = this.makeCandidate(validation, 'learner');
    const promotion = this.considerCandidate(currentCandidate);
    const archiveUpdates = promotion.updates;
    updateSkillBestRecords(this.skillBestRecords, validation.stageResults, this.totalSteps);
    syncSkillBestScores(this);
    this.lastSkillValidation = validation;
    this.bestBrain = this.getArchiveBrain('balanced');

    const improved = archiveUpdates.includes('balanced');
    const regression = balancedEvidence;
    const autoRollback = null; // v0.1.2: behavioral regression is observed, never auto-restored.

    this.retentionStatus = {
      alerts: skillAssessment.alerts,
      forgetting: skillAssessment.alerts,
      healthy: skillAssessment.alerts.length === 0 && !balancedEvidence,
      interpretation,
      confirmed: balancedConfirmed || confirmedAlerts.length > 0,
      balancedEvidence,
      balancedConfirmed,
      balancedRegressionStreak: this.balancedRegressionStreak,
      skillImprovements,
      checkedAtSteps: this.totalSteps,
      intervention: 'observe-only',
    };

    const record = {
      steps: this.totalSteps,
      episodes: this.totalEpisodes,
      curriculum: this.curriculum.stage,
      validation,
      improved,
      regression,
      balancedEvidence,
      balancedConfirmed,
      balancedRegressionStreak: this.balancedRegressionStreak,
      deltaFromBest,
      forgetting: skillAssessment.alerts,
      skillImprovements,
      interpretation,
      archiveUpdates,
      promotionPending: promotion.pending,
      autoRollback,
      migratedArchive,
      learnerLineage: { ...this.learnerLineage },
      bestSteps: this.bestBrain?.savedAtSteps ?? null,
      bestScore: this.bestBrain?.validation?.categoryScores?.balanced ?? null,
    };
    this.validationHistory.push(record);
    if (this.validationHistory.length > 64) this.validationHistory.shift();
    this.lastValidationStep = this.totalSteps;
    const regularNextValidation = nextValidationAfter(this.totalSteps);
    const needsFollowup = Boolean(skillAssessment.alerts.length || balancedEvidence || promotion.pending.length);
    this.nextValidationStep = needsFollowup
      ? Math.min(regularNextValidation, this.totalSteps + CONFIG.validation.watchValidationInterval)
      : regularNextValidation;
    return record;
  }

  makeCandidate(validation, source = 'learner', base = null) {
    return {
      savedAtSteps: base?.savedAtSteps ?? this.totalSteps,
      savedAtEpisodes: base?.savedAtEpisodes ?? this.totalEpisodes,
      model: base?.model ?? this.model.serialize(),
      optimizer: base?.optimizer ?? this.trainer.serialize(),
      curriculum: base?.curriculum ?? this.curriculum.serialize(),
      validation,
      source,
      lineage: base?.lineage ? structuredLineage(base.lineage) : structuredLineage(this.learnerLineage),
      observedAtSteps: this.totalSteps,
    };
  }

  seedArchiveCandidate(candidate) {
    const updates = [];
    for (const category of ARCHIVE_CATEGORIES) {
      const score = candidate.validation?.categoryScores?.[category];
      if (!Number.isFinite(score)) continue;
      const prior = this.bestArchive[category];
      const priorScore = prior?.validation?.categoryScores?.[category];
      if (!Number.isFinite(priorScore) || score > priorScore + CONFIG.validation.improvementEpsilon) {
        this.bestArchive[category] = {
          ...candidate,
          category,
          categoryScore: score,
          promotedAtSteps: this.totalSteps,
          promotionEvidence: { type: 'archive-rebaseline', confirmations: 0 },
        };
        updates.push(category);
      }
    }
    this.bestBrain = this.getArchiveBrain('balanced');
    return updates;
  }

  considerCandidate(candidate) {
    const updates = [];
    const pending = [];
    for (const category of ARCHIVE_CATEGORIES) {
      const score = candidate.validation?.categoryScores?.[category];
      if (!Number.isFinite(score)) continue;
      const prior = this.bestArchive[category];
      const priorScore = prior?.validation?.categoryScores?.[category];
      if (!Number.isFinite(priorScore)) {
        this.bestArchive[category] = {
          ...candidate,
          category,
          categoryScore: score,
          promotedAtSteps: this.totalSteps,
          promotionEvidence: { type: 'first-champion', confirmations: 1 },
        };
        this.promotionStreaks[category] = 0;
        delete this.promotionCandidates[category];
        updates.push(category);
        continue;
      }

      const margin = Math.max(CONFIG.validation.improvementEpsilon, CONFIG.validation.championPromotionMargin);
      if (score > priorScore + margin) {
        const streak = (Number(this.promotionStreaks[category]) || 0) + 1;
        this.promotionStreaks[category] = streak;
        this.promotionCandidates[category] = {
          firstObservedAtSteps: this.promotionCandidates[category]?.firstObservedAtSteps ?? this.totalSteps,
          latestObservedAtSteps: this.totalSteps,
          streak,
          score,
          championScore: priorScore,
        };
        if (streak >= CONFIG.validation.championPromotionConfirmations) {
          this.bestArchive[category] = {
            ...candidate,
            category,
            categoryScore: score,
            promotedAtSteps: this.totalSteps,
            promotionEvidence: {
              type: 'repeat-confirmed-challenger',
              confirmations: streak,
              priorChampionSteps: prior.savedAtSteps,
              priorChampionScore: priorScore,
            },
          };
          this.promotionStreaks[category] = 0;
          delete this.promotionCandidates[category];
          updates.push(category);
        } else {
          pending.push({ category, streak, required: CONFIG.validation.championPromotionConfirmations, score, championScore: priorScore });
        }
      } else {
        this.promotionStreaks[category] = 0;
        delete this.promotionCandidates[category];
      }
    }
    this.bestBrain = this.getArchiveBrain('balanced');
    return { updates, pending };
  }

  getArchiveBrain(category = 'balanced') {
    return this.bestArchive?.[category] || (category === 'balanced' ? this.bestBrain : null);
  }

  archiveSummary() {
    return Object.fromEntries(ARCHIVE_CATEGORIES.map(category => {
      const brain = this.getArchiveBrain(category);
      return [category, brain ? {
        savedAtSteps: brain.savedAtSteps,
        score: brain.validation?.categoryScores?.[category] ?? null,
        lineage: brain.lineage || null,
        promotedAtSteps: brain.promotedAtSteps ?? brain.savedAtSteps,
      } : null];
    }));
  }

  getHallEntry(id) {
    return this.hallOfFame.find(entry => entry.id === id) || null;
  }

  hallOfFameSummary() {
    return this.hallOfFame.map(entry => ({
      id: entry.id,
      label: entry.label,
      category: entry.category,
      savedAtSteps: entry.savedAtSteps,
      pinnedAtSteps: entry.pinnedAtSteps,
      lineage: entry.lineage || null,
      sourceVersion: entry.sourceVersion || null,
      sourceBuild: entry.sourceBuild || null,
      fingerprint: entry.fingerprint,
      reason: entry.reason || null,
    }));
  }

  exportHallOfFame() {
    return {
      schema: 1,
      nextId: this.nextHallOfFameId,
      entries: cloneSerializable(this.hallOfFame),
    };
  }

  mergeHallOfFame(payload) {
    const entries = Array.isArray(payload?.entries) ? payload.entries : (Array.isArray(payload) ? payload : []);
    const existingFingerprints = new Set(this.hallOfFame.map(x => x.fingerprint || modelFingerprint(x.model)));
    const existingIds = new Set(this.hallOfFame.map(x => x.id));
    for (const raw of entries) {
      if (!raw?.model) continue;
      const fingerprint = raw.fingerprint || modelFingerprint(raw.model);
      if (existingFingerprints.has(fingerprint)) continue;
      let id = raw.id;
      if (!id || existingIds.has(id)) id = this.allocateHallId();
      const entry = { ...cloneSerializable(raw), id, fingerprint };
      this.hallOfFame.push(entry);
      existingFingerprints.add(fingerprint);
      existingIds.add(id);
    }
    const numericIds = this.hallOfFame.map(x => Number(String(x.id || '').match(/(\d+)$/)?.[1] || 0));
    this.nextHallOfFameId = Math.max(Number(payload?.nextId) || 1, this.nextHallOfFameId, ...numericIds.map(x => x + 1));
    return this.hallOfFameSummary();
  }

  allocateHallId() {
    const id = `HOF-${String(this.nextHallOfFameId).padStart(3, '0')}`;
    this.nextHallOfFameId++;
    return id;
  }

  pinChampion(category = 'balanced', { reason = 'manual-pin' } = {}) {
    const brain = this.getArchiveBrain(category);
    if (!brain?.model) throw new Error(`No validated ${category} Champion is available to pin`);
    const fingerprint = modelFingerprint(brain.model);
    const existing = this.hallOfFame.find(entry => entry.fingerprint === fingerprint);
    if (existing) return { entry: existing, created: false };
    const entry = {
      id: this.allocateHallId(),
      label: `Champion ${category}`,
      category,
      savedAtSteps: brain.savedAtSteps,
      savedAtEpisodes: brain.savedAtEpisodes ?? null,
      pinnedAtSteps: this.totalSteps,
      pinnedAtEpisodes: this.totalEpisodes,
      sourceVersion: VERSION,
      sourceBuild: BUILD_MARKER,
      fingerprint,
      reason,
      lineage: structuredLineage(brain.lineage),
      validation: cloneSerializable(brain.validation || null),
      curriculum: cloneSerializable(brain.curriculum || null),
      promotionEvidence: cloneSerializable(brain.promotionEvidence || null),
      model: cloneSerializable(brain.model),
      optimizer: cloneSerializable(brain.optimizer || null),
    };
    this.hallOfFame.push(entry);
    return { entry, created: true };
  }

  freezeCurrentLearner(reason = 'manual-branch-freeze') {
    const lineageId = this.learnerLineage?.id || `L-${(this.seed >>> 0).toString(16)}-unknown`;
    const entry = {
      id: lineageId,
      lineage: structuredLineage(this.learnerLineage),
      frozenAtSteps: this.totalSteps,
      frozenAtEpisodes: this.totalEpisodes,
      learnerExperienceSteps: this.learnerExperienceSteps,
      reason,
      curriculum: this.curriculum.serialize(),
      envSeedCursor: this.envSeedCursor,
      actionRngState: this.actionRng.state >>> 0,
      rehearsalEpisodeHistory: [...this.rehearsalEpisodeHistory],
      model: this.model.serialize(),
      optimizer: this.trainer.serialize(),
    };
    const idx = this.frozenLearners.findIndex(x => x.id === lineageId);
    if (idx >= 0) this.frozenLearners[idx] = entry;
    else this.frozenLearners.push(entry);
    return entry;
  }

  frozenLearnerSummary() {
    return this.frozenLearners.map(x => ({
      id: x.id,
      lineage: x.lineage || null,
      frozenAtSteps: x.frozenAtSteps,
      learnerExperienceSteps: x.learnerExperienceSteps || 0,
      reason: x.reason || null,
      curriculumStage: x.curriculum?.stage ?? null,
    }));
  }

  resetBranchValidationState() {
    this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
    this.promotionCandidates = {};
    this.skillRegressionStreaks = Array(CURRICULUM.length).fill(0);
    this.balancedRegressionStreak = 0;
    this.retentionStatus = defaultRetentionStatus();
    this.lastSkillValidation = null;
    this.lastValidationStep = -1;
    this.nextValidationStep = this.totalSteps;
  }

  activateForkSource(source, parent, reason) {
    if (!source?.model) throw new Error('Fork source has no model');
    const frozen = this.freezeCurrentLearner('preserved-before-fork');
    this.model.restore(source.model);
    if (source.optimizer) this.trainer.restore(source.optimizer);
    else this.trainer = new PPOTrainer(this.model, this.seed ^ 0x9e3779b9 ^ (this.lineageCounter + 1));
    this.lineageCounter++;
    const lineage = {
      id: `L-${(this.seed >>> 0).toString(16)}-${this.lineageCounter}`,
      startedAtSteps: this.totalSteps,
      parent,
      reason,
    };
    this.learnerLineage = lineage;
    this.learnerExperienceSteps = 0;
    this.lineageHistory.push(structuredLineage(lineage));
    if (this.lineageHistory.length > 64) this.lineageHistory.shift();
    this.resetBranchValidationState();
    this.rehearsalEpisodeHistory = [];
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = {
      atSteps: this.totalSteps,
      sourceSteps: source.savedAtSteps,
      automatic: false,
      type: reason,
      lineageId: lineage.id,
      preservedLineageId: frozen.id,
      category: parent?.category || null,
      hallId: parent?.hallId || null,
      parent: cloneSerializable(parent),
    };
    this.rollbackHistory.push(event);
    if (this.rollbackHistory.length > 32) this.rollbackHistory.shift();
    return event;
  }

  forkFromChampion(category = 'balanced') {
    const brain = this.getArchiveBrain(category);
    if (!brain?.model) throw new Error(`No validated ${category} Champion is available yet`);
    return this.activateForkSource(brain, {
      type: 'champion',
      category,
      sourceSteps: brain.savedAtSteps,
      lineageId: brain.lineage?.id || null,
    }, 'manual-champion-fork');
  }

  forkFromHallOfFame(id) {
    const entry = this.getHallEntry(id);
    if (!entry?.model) throw new Error(`Hall of Fame entry ${id} is unavailable`);
    return this.activateForkSource(entry, {
      type: 'hall-of-fame',
      hallId: entry.id,
      category: entry.category,
      sourceSteps: entry.savedAtSteps,
      lineageId: entry.lineage?.id || null,
    }, 'manual-hall-of-fame-fork');
  }

  switchToFrozenLearner(lineageId) {
    const idx = this.frozenLearners.findIndex(x => x.id === lineageId);
    if (idx < 0) throw new Error(`Frozen Learner ${lineageId} not found`);
    const target = this.frozenLearners[idx];
    this.frozenLearners.splice(idx, 1);
    const preserved = this.freezeCurrentLearner('preserved-before-lineage-switch');
    this.model.restore(target.model);
    this.trainer.restore(target.optimizer);
    this.curriculum.restore(target.curriculum || {});
    this.envSeedCursor = Number(target.envSeedCursor) || this.envSeedCursor;
    if (Number.isFinite(Number(target.actionRngState))) this.actionRng.state = Number(target.actionRngState) >>> 0;
    this.rehearsalEpisodeHistory = Array.isArray(target.rehearsalEpisodeHistory)
      ? target.rehearsalEpisodeHistory.slice(-CONFIG.continual.recentMixWindow)
      : [];
    this.learnerLineage = structuredLineage(target.lineage) || this.learnerLineage;
    this.learnerExperienceSteps = Math.max(0, Number(target.learnerExperienceSteps) || 0);
    this.resetBranchValidationState();
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = {
      atSteps: this.totalSteps,
      automatic: false,
      type: 'manual-lineage-switch',
      lineageId: this.learnerLineage?.id || lineageId,
      preservedLineageId: preserved.id,
    };
    this.rollbackHistory.push(event);
    if (this.rollbackHistory.length > 32) this.rollbackHistory.shift();
    return event;
  }

  restoreBest(category = 'balanced') {
    return this.forkFromChampion(category);
  }

  snapshot() {
    return {
      schema: 6,
      seed: this.seed,
      totalSteps: this.totalSteps,
      totalEpisodes: this.totalEpisodes,
      envSeedCursor: this.envSeedCursor,
      actionRngState: this.actionRng.state >>> 0,
      curriculum: this.curriculum.serialize(),
      model: this.model.serialize(),
      optimizer: this.trainer.serialize(),
      metrics: this.metrics,
      milestones: Array.from(this.milestones.entries()),
      nextMilestoneStep: this.nextMilestoneStep,
      bestBrain: this.bestBrain,
      bestArchive: this.bestArchive,
      validationHistory: this.validationHistory,
      legacyValidationHistory: this.legacyValidationHistory,
      lastValidationStep: this.lastValidationStep,
      nextValidationStep: this.nextValidationStep,
      lastSkillValidation: this.lastSkillValidation,
      skillBestScores: this.skillBestScores,
      skillBestRecords: this.skillBestRecords,
      skillRegressionStreaks: this.skillRegressionStreaks,
      balancedRegressionStreak: this.balancedRegressionStreak,
      retentionStatus: this.retentionStatus,
      pendingLegacyBest: this.pendingLegacyBest,
      pendingArchiveMigration: this.pendingArchiveMigration,
      archiveNeedsRebaseline: this.archiveNeedsRebaseline,
      promotionStreaks: this.promotionStreaks,
      promotionCandidates: this.promotionCandidates,
      learnerLineage: this.learnerLineage,
      lineageCounter: this.lineageCounter,
      lineageHistory: this.lineageHistory,
      rehearsalEpisodeHistory: this.rehearsalEpisodeHistory,
      learnerExperienceSteps: this.learnerExperienceSteps,
      hallOfFame: this.exportHallOfFame(),
      frozenLearners: this.frozenLearners,
      rollbackHistory: this.rollbackHistory,
    };
  }

  restore(data) {
    if (!data || ![1, 2, 3, 4, 5, 6].includes(data.schema)) throw new Error('Unsupported checkpoint schema');
    this.seed = data.seed;
    this.actionRng = new PRNG(this.seed ^ 0xa5a5a5a5);
    this.hallOfFame = [];
    this.nextHallOfFameId = 1;
    this.frozenLearners = [];
    this.learnerExperienceSteps = 0;
    this.totalSteps = data.totalSteps || 0;
    this.totalEpisodes = data.totalEpisodes || 0;
    this.envSeedCursor = data.envSeedCursor || 0;
    this.model.restore(data.model);
    this.trainer.restore(data.optimizer);
    this.curriculum.restore(data.curriculum || {});
    if (data.schema < 3) this.curriculum.cooldownRemaining = Math.max(this.curriculum.cooldownRemaining, CONFIG.curriculum.transitionCooldownEpisodes);
    this.metrics = Array.isArray(data.metrics) ? data.metrics.slice(-CONFIG.runtime.chartPoints) : [];
    this.milestones = new Map(Array.isArray(data.milestones) ? data.milestones : []);
    this.nextMilestoneStep = Number.isFinite(data.nextMilestoneStep) && data.nextMilestoneStep > this.totalSteps
      ? data.nextMilestoneStep
      : nextHistoricalMilestoneAfter(this.totalSteps);
    if (data.schema === 1 && this.totalSteps > 0 && !this.milestones.has(this.totalSteps)) this.saveMilestone(this.totalSteps);

    if (data.schema === 6 || data.schema === 5) {
      this.bestArchive = data.bestArchive && typeof data.bestArchive === 'object' ? data.bestArchive : {};
      this.bestBrain = this.bestArchive.balanced || data.bestBrain || null;
      this.validationHistory = Array.isArray(data.validationHistory) ? data.validationHistory.slice(-64) : [];
      this.legacyValidationHistory = Array.isArray(data.legacyValidationHistory) ? data.legacyValidationHistory.slice(-128) : [];
      this.lastValidationStep = Number(data.lastValidationStep ?? -1);
      this.nextValidationStep = Number.isFinite(data.nextValidationStep) ? data.nextValidationStep : nextValidationAfter(this.totalSteps);
      this.lastSkillValidation = data.lastSkillValidation || null;
      this.skillBestRecords = normalizeSkillBestRecords(data.skillBestRecords, data.skillBestScores);
      this.skillBestScores = this.skillBestRecords.map(x => x?.score || 0);
      this.skillRegressionStreaks = normalizeStreaks(data.skillRegressionStreaks);
      this.balancedRegressionStreak = Math.max(0, Number(data.balancedRegressionStreak) || 0);
      this.retentionStatus = normalizeRetentionStatus(data.retentionStatus);
      this.pendingLegacyBest = data.pendingLegacyBest || null;
      this.pendingArchiveMigration = Array.isArray(data.pendingArchiveMigration) ? data.pendingArchiveMigration : [];
      this.archiveNeedsRebaseline = Boolean(data.archiveNeedsRebaseline);
      this.promotionStreaks = normalizePromotionStreaks(data.promotionStreaks);
      this.promotionCandidates = data.promotionCandidates && typeof data.promotionCandidates === 'object' ? data.promotionCandidates : {};
      this.lineageCounter = Math.max(0, Number(data.lineageCounter) || 0);
      this.learnerLineage = data.learnerLineage ? structuredLineage(data.learnerLineage) : makeRootLineage(this.seed, this.totalSteps);
      this.lineageHistory = Array.isArray(data.lineageHistory) && data.lineageHistory.length
        ? data.lineageHistory.slice(-64).map(structuredLineage)
        : [structuredLineage(this.learnerLineage)];
      this.rehearsalEpisodeHistory = Array.isArray(data.rehearsalEpisodeHistory)
        ? data.rehearsalEpisodeHistory.filter(x => Number.isInteger(x) && x >= 0 && x < CURRICULUM.length).slice(-CONFIG.continual.recentMixWindow)
        : [];
      if (data.schema === 6) {
        if (Number.isFinite(Number(data.actionRngState))) this.actionRng.state = Number(data.actionRngState) >>> 0;
        this.mergeHallOfFame(data.hallOfFame || []);
        this.frozenLearners = Array.isArray(data.frozenLearners)
          ? data.frozenLearners.filter(x => x?.id && x?.model && x?.optimizer).map(cloneSerializable)
          : [];
        this.learnerExperienceSteps = Math.max(0, Number(data.learnerExperienceSteps) || 0);
      } else {
        this.learnerExperienceSteps = Math.max(0, this.totalSteps - (Number(this.learnerLineage?.startedAtSteps) || 0));
      }
    } else if (data.schema === 4) {
      // v0.1.1.1 already uses validation:v3. Preserve its champions exactly, but
      // start the active learner as a new continual-learning lineage and validate
      // immediately under the autonomous learner/champion protocol.
      this.bestArchive = data.bestArchive && typeof data.bestArchive === 'object' ? data.bestArchive : {};
      this.bestBrain = this.bestArchive.balanced || data.bestBrain || null;
      this.validationHistory = Array.isArray(data.validationHistory) ? data.validationHistory.slice(-64) : [];
      this.legacyValidationHistory = Array.isArray(data.legacyValidationHistory) ? data.legacyValidationHistory.slice(-128) : [];
      this.lastValidationStep = Number(data.lastValidationStep ?? -1);
      this.nextValidationStep = this.totalSteps;
      this.lastSkillValidation = data.lastSkillValidation || null;
      this.skillBestRecords = normalizeSkillBestRecords(data.skillBestRecords, data.skillBestScores);
      this.skillBestScores = this.skillBestRecords.map(x => x?.score || 0);
      this.skillRegressionStreaks = normalizeStreaks(data.skillRegressionStreaks);
      this.balancedRegressionStreak = Math.max(0, Number(data.balancedRegressionStreak) || 0);
      this.retentionStatus = normalizeRetentionStatus(data.retentionStatus);
      this.pendingLegacyBest = data.pendingLegacyBest || null;
      this.pendingArchiveMigration = Array.isArray(data.pendingArchiveMigration) ? data.pendingArchiveMigration : [];
      this.archiveNeedsRebaseline = Boolean(data.archiveNeedsRebaseline);
      this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
      this.promotionCandidates = {};
      this.lineageCounter = 0;
      this.learnerLineage = makeMigratedLineage(this.seed, this.totalSteps, 4);
      this.lineageHistory = [structuredLineage(this.learnerLineage)];
      this.rehearsalEpisodeHistory = [];
    } else if (data.schema === 3) {
      // v0.1.1 used validation:v2. Preserve every specialist model for inspection,
      // but schedule an immediate v3 rebaseline before autonomous training advances.
      this.bestArchive = data.bestArchive && typeof data.bestArchive === 'object' ? data.bestArchive : {};
      this.bestBrain = this.bestArchive.balanced || data.bestBrain || null;
      this.pendingArchiveMigration = uniqueCandidates([
        ...Object.values(this.bestArchive || {}),
        ...(data.bestBrain?.model ? [data.bestBrain] : []),
        ...(data.pendingLegacyBest?.model ? [data.pendingLegacyBest] : []),
      ]);
      this.archiveNeedsRebaseline = true;
      this.pendingLegacyBest = null;
      this.legacyValidationHistory = Array.isArray(data.validationHistory) ? data.validationHistory.slice(-128) : [];
      this.validationHistory = [];
      this.lastValidationStep = -1;
      this.nextValidationStep = this.totalSteps;
      this.lastSkillValidation = null;
      this.skillBestRecords = emptySkillBestRecords();
      this.skillBestScores = Array(CURRICULUM.length).fill(0);
      this.skillRegressionStreaks = Array(CURRICULUM.length).fill(0);
      this.balancedRegressionStreak = 0;
      this.retentionStatus = defaultRetentionStatus();
      this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
      this.promotionCandidates = {};
      this.lineageCounter = 0;
      this.learnerLineage = makeMigratedLineage(this.seed, this.totalSteps, 3);
      this.lineageHistory = [structuredLineage(this.learnerLineage)];
      this.rehearsalEpisodeHistory = [];
    } else {
      // Schema 1/2 had at most one trustworthy protected Best. Keep it as a model
      // candidate, but discard its incompatible numeric score until v3 recalibration.
      const legacyBest = data.bestBrain?.model ? data.bestBrain : strongestLegacyBest(data.bestBrains);
      this.bestArchive = {};
      this.bestBrain = legacyBest || null;
      this.pendingLegacyBest = legacyBest || null;
      this.pendingArchiveMigration = legacyBest ? [legacyBest] : [];
      this.archiveNeedsRebaseline = Boolean(legacyBest);
      this.legacyValidationHistory = Array.isArray(data.validationHistory) ? data.validationHistory.slice(-128) : [];
      this.validationHistory = [];
      this.lastValidationStep = -1;
      this.nextValidationStep = this.totalSteps;
      this.lastSkillValidation = null;
      this.skillBestRecords = emptySkillBestRecords();
      this.skillBestScores = Array(CURRICULUM.length).fill(0);
      this.skillRegressionStreaks = Array(CURRICULUM.length).fill(0);
      this.balancedRegressionStreak = 0;
      this.retentionStatus = defaultRetentionStatus();
      this.promotionStreaks = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
      this.promotionCandidates = {};
      this.lineageCounter = 0;
      this.learnerLineage = makeMigratedLineage(this.seed, this.totalSteps, data.schema);
      this.lineageHistory = [structuredLineage(this.learnerLineage)];
      this.rehearsalEpisodeHistory = [];
    }

    if (data.schema < 6) this.learnerExperienceSteps = Math.max(0, this.totalSteps - (Number(this.learnerLineage?.startedAtSteps) || 0));
    this.rollbackHistory = Array.isArray(data.rollbackHistory) ? data.rollbackHistory.slice(-32) : [];
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
  }

}

export function assessSkillRetention(stageResults, bestRecords, priorStreaks = []) {
  const alerts = [];
  const streaks = Array(CURRICULUM.length).fill(0);
  for (const stage of stageResults) {
    const best = bestRecords?.[stage.stage];
    const prior = Number(best?.score) || 0;
    const drop = prior - stage.skillScore;
    const confidenceSeparated = Number.isFinite(best?.ciLow)
      ? stage.skillCiHigh < best.ciLow
      : drop >= CONFIG.validation.skillCatastrophicDrop;
    const evidence = prior >= CONFIG.validation.forgettingFloor
      && drop >= CONFIG.validation.skillWarningDrop
      && (confidenceSeparated || drop >= CONFIG.validation.skillCatastrophicDrop);
    streaks[stage.stage] = evidence ? (Math.max(0, Number(priorStreaks?.[stage.stage]) || 0) + 1) : 0;
    if (!evidence) continue;
    const severity = drop >= CONFIG.validation.skillCatastrophicDrop ? 'catastrophic' : 'warning';
    alerts.push({
      stage: stage.stage,
      name: stage.name,
      prior,
      priorCiLow: Number(best?.ciLow) || prior,
      priorCiHigh: Number(best?.ciHigh) || prior,
      current: stage.skillScore,
      currentCiLow: stage.skillCiLow,
      currentCiHigh: stage.skillCiHigh,
      drop,
      severity,
      streak: streaks[stage.stage],
      confirmed: streaks[stage.stage] >= CONFIG.validation.confirmationCount,
      confidenceSeparated,
    });
  }
  return { alerts, streaks };
}

function balancedRegressionEvidence(current, prior) {
  const priorScore = Number(prior?.categoryScores?.balanced);
  if (!Number.isFinite(priorScore)) return false;
  const drop = priorScore - current.balancedScore;
  if (drop < CONFIG.validation.regressionTolerance) return false;
  const priorLow = Number(prior?.balancedCiLow);
  const currentHigh = Number(current?.balancedCiHigh);
  return (Number.isFinite(priorLow) && Number.isFinite(currentHigh) && currentHigh < priorLow)
    || drop >= CONFIG.validation.regressionTolerance * 1.5;
}

function detectSkillImprovements(stageResults, bestRecords) {
  const out = [];
  for (const stage of stageResults) {
    const prior = Number(bestRecords?.[stage.stage]?.score) || 0;
    if (stage.skillScore > prior + CONFIG.validation.improvementEpsilon) {
      out.push({ stage: stage.stage, name: stage.name, prior, current: stage.skillScore, gain: stage.skillScore - prior });
    }
  }
  return out;
}

function updateSkillBestRecords(records, stageResults, atSteps) {
  for (const stage of stageResults) {
    const current = records[stage.stage];
    if (!current || stage.skillScore > current.score + CONFIG.validation.improvementEpsilon) {
      records[stage.stage] = {
        stage: stage.stage,
        name: stage.name,
        score: stage.skillScore,
        ciLow: stage.skillCiLow,
        ciHigh: stage.skillCiHigh,
        atSteps,
      };
    }
  }
}

function emptySkillBestRecords() {
  return Array(CURRICULUM.length).fill(null);
}

function normalizeSkillBestRecords(records, legacyScores) {
  const out = emptySkillBestRecords();
  if (Array.isArray(records)) {
    for (let i = 0; i < out.length; i++) {
      const x = records[i];
      if (x && Number.isFinite(Number(x.score))) out[i] = { ...x, score: Math.max(0, Number(x.score)) };
    }
  } else if (Array.isArray(legacyScores)) {
    for (let i = 0; i < out.length; i++) {
      const score = Math.max(0, Number(legacyScores[i]) || 0);
      if (score) out[i] = { stage: i, name: CURRICULUM[i].name, score, ciLow: score, ciHigh: score, atSteps: null };
    }
  }
  return out;
}

function syncSkillBestScores(session) {
  session.skillBestScores = session.skillBestRecords.map(x => x?.score || 0);
}

function normalizeStreaks(values) {
  const out = Array(CURRICULUM.length).fill(0);
  if (Array.isArray(values)) for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.floor(Number(values[i]) || 0));
  return out;
}

function normalizePromotionStreaks(values) {
  const out = Object.fromEntries(ARCHIVE_CATEGORIES.map(x => [x, 0]));
  if (values && typeof values === 'object') {
    for (const category of ARCHIVE_CATEGORIES) out[category] = Math.max(0, Math.floor(Number(values[category]) || 0));
  }
  return out;
}

function makeRootLineage(seed, startedAtSteps = 0) {
  return {
    id: `L-${(seed >>> 0).toString(16)}-0`,
    startedAtSteps: Math.max(0, Number(startedAtSteps) || 0),
    parent: null,
    reason: 'new-brain',
  };
}

function makeMigratedLineage(seed, startedAtSteps, schema) {
  return {
    id: `L-${(seed >>> 0).toString(16)}-0`,
    startedAtSteps: Math.max(0, Number(startedAtSteps) || 0),
    parent: { legacySchema: schema },
    reason: `migrated-schema-${schema}-learner`,
  };
}

function structuredLineage(value) {
  if (!value || typeof value !== 'object') return null;
  return JSON.parse(JSON.stringify(value));
}

function defaultRetentionStatus() {
  return { alerts: [], forgetting: [], healthy: false, interpretation: 'unvalidated', confirmed: false };
}

function normalizeRetentionStatus(value) {
  if (!value || typeof value !== 'object') return defaultRetentionStatus();
  const alerts = Array.isArray(value.alerts) ? value.alerts : (Array.isArray(value.forgetting) ? value.forgetting : []);
  return { ...defaultRetentionStatus(), ...value, alerts, forgetting: alerts };
}

function cloneSerializable(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

function uniqueCandidates(values) {
  const out = [];
  const seen = new Set();
  for (const candidate of values || []) {
    if (!candidate?.model) continue;
    const key = `${candidate.savedAtSteps ?? 'x'}:${modelFingerprint(candidate.model)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(candidate);
  }
  return out;
}

function modelFingerprint(model) {
  const keys = Object.keys(model?.params || {}).sort();
  let h = 2166136261 >>> 0;
  for (const key of keys) {
    const arr = model.params[key];
    if (!arr || typeof arr.length !== 'number') continue;
    for (let i = 0; i < arr.length; i++) {
      const q = Math.round((Number(arr[i]) || 0) * 1e6);
      h ^= q & 0xffffffff;
      h = Math.imul(h, 16777619) >>> 0;
    }
  }
  return h.toString(16);
}

function strongestLegacyBest(bestBrains) {
  if (!bestBrains || typeof bestBrains !== 'object') return null;
  let best = null;
  for (const candidate of Object.values(bestBrains)) {
    if (!candidate?.model) continue;
    if (!best || (candidate.validation?.score ?? -Infinity) > (best.validation?.score ?? -Infinity)) best = candidate;
  }
  return best;
}

function compactValidation(record) {
  return {
    improved: record.improved,
    regression: record.regression,
    score: record.validation.score,
    bestScore: record.bestScore,
    bestSteps: record.bestSteps,
    forgetting: record.forgetting.map(x => x.stage),
    interpretation: record.interpretation,
    balancedConfirmed: record.balancedConfirmed,
    archiveUpdates: record.archiveUpdates,
    promotionPending: record.promotionPending || [],
    autoRollback: null,
  };
}

function nextValidationAfter(steps) {
  const cfg = CONFIG.validation;
  for (const step of cfg.schedule) if (step > steps) return step;
  const anchor = cfg.schedule.at(-1) || 0;
  const intervals = Math.floor(Math.max(0, steps - anchor) / cfg.intervalAfterSchedule) + 1;
  return anchor + intervals * cfg.intervalAfterSchedule;
}

export function nextHistoricalMilestoneAfter(steps) {
  const fixed = [1000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000];
  for (const step of fixed) if (step > steps) return step;
  if (steps < 5_000_000) return nextGridStep(steps, 1_000_000, 250_000);
  if (steps < 20_000_000) return nextGridStep(steps, 5_000_000, 500_000);
  if (steps < 100_000_000) return nextGridStep(steps, 20_000_000, 1_000_000);
  return nextGridStep(steps, 100_000_000, 5_000_000);
}

function nextGridStep(steps, anchor, interval) {
  const n = Math.floor(Math.max(0, steps - anchor) / interval) + 1;
  return anchor + n * interval;
}

function performanceNow() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}
