import { CONFIG } from '../config.js';
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
    this.rollbackHistory = [];
    this.envs = [];
    this.obs = [];
    this.hidden = [];
    for (let i = 0; i < envCount; i++) this.addEnv(i);
    this.saveMilestone(0);
  }

  addEnv(id) {
    const seed = domainSeed(`train:${this.seed}`, this.envSeedCursor++);
    const env = new World(seed, this.curriculum.current());
    this.envs[id] = env;
    this.obs[id] = env.observe();
    this.hidden[id] = this.model.zeroHidden();
  }

  resetEnv(id) {
    const seed = domainSeed(`train:${this.seed}`, this.envSeedCursor++);
    const env = new World(seed, this.curriculum.current());
    this.envs[id] = env;
    this.obs[id] = env.observe();
    this.hidden[id] = this.model.zeroHidden();
  }

  trainRollout(steps = CONFIG.runtime.rolloutSteps) {
    const transitions = [];
    const completed = [];
    let curriculumEvent = null;
    const started = performanceNow();
    const preValidation = this.totalSteps >= this.nextValidationStep && this.lastValidationStep !== this.totalSteps ? this.runValidation() : null;

    for (let s = 0; s < steps; s++) {
      for (let i = 0; i < this.envs.length; i++) {
        const obs = this.obs[i];
        const hPrev = this.hidden[i];
        const act = this.model.act(obs, hPrev, this.actionRng, false);
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
          if (this.autoCurriculum && info.stageId === this.curriculum.stage) {
            const gate = this.promotionGate();
            const event = this.curriculum.noteEpisode(info, this.totalEpisodes, {
              promotionAllowed: gate.allowed,
              gateReason: gate.reason,
            });
            if (event) curriculumEvent = event;
          }
          this.resetEnv(i);
        }
      }
    }

    const lastValues = new Map();
    for (let i = 0; i < this.envs.length; i++) lastValues.set(i, this.model.forward(this.obs[i], this.hidden[i]).value);
    const { advantages, returns } = computeGAE(transitions, lastValues);
    const ppo = this.trainer.update(transitions, advantages, returns, { trainingStep: this.totalSteps });
    const elapsed = Math.max(1, performanceNow() - started);
    const recent = this.episodeHistory.slice(-40);
    const mean = key => recent.length ? recent.reduce((sum, x) => sum + (x[key] || 0), 0) / recent.length : 0;

    this.maybeSaveMilestones();
    const forceValidation = Boolean(curriculumEvent && curriculumEvent.reason !== 'promotion-blocked');
    const postValidation = this.maybeAutoValidate(forceValidation);
    const validation = postValidation || preValidation;
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
      throughput: transitions.length / (elapsed / 1000),
      validation: validation ? compactValidation(validation) : null,
      retentionAlertCount: this.retentionStatus.forgetting.length,
      skillScores: this.lastSkillValidation?.stageResults?.map(x => x.skillScore) || [],
      ...ppo,
    };
    this.metrics.push(metric);
    if (this.metrics.length > CONFIG.runtime.chartPoints) this.metrics.shift();
    return { metric, completed, transitions: transitions.length, validation, curriculumEvent };
  }

  promotionGate() {
    const v = this.lastSkillValidation;
    if (!v) return { allowed: false, reason: 'waiting for skill-retention validation' };
    const required = v.stageResults.slice(0, this.curriculum.stage + 1);
    const weak = required.find(x => x.skillScore < CONFIG.curriculum.promotionSkillFloor);
    if (weak) return { allowed: false, reason: `${weak.name} retention ${(weak.skillScore * 100).toFixed(0)}% below ${(CONFIG.curriculum.promotionSkillFloor * 100).toFixed(0)}% floor` };
    const alert = (this.retentionStatus.alerts || this.retentionStatus.forgetting || []).find(x => x.stage <= this.curriculum.stage);
    if (alert) return { allowed: false, reason: `${alert.name} retention is under confirmation (${alert.severity || 'regression'}${alert.confirmed ? ', confirmed' : ''})` };
    return { allowed: true, reason: 'retained prior skills' };
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

    // Validation protocol v3 changes both sampling density and skill-score statistics.
    // Re-evaluate every unique protected legacy candidate before comparing scores so
    // a v0.1.1/v2 number is never treated as compatible with a v0.1.1.1/v3 number.
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
        reevaluated.push(this.makeCandidate(legacyValidation, 'recalibrated-v3-archive', base));
      }
      this.bestArchive = {};
      this.bestBrain = null;
      this.skillBestRecords = emptySkillBestRecords();
      this.skillBestScores = Array(CURRICULUM.length).fill(0);
      for (const candidate of reevaluated) {
        this.considerCandidate(candidate);
        updateSkillBestRecords(this.skillBestRecords, candidate.validation.stageResults, candidate.savedAtSteps);
      }
      syncSkillBestScores(this);
      migratedArchive = reevaluated.map(x => ({ savedAtSteps: x.savedAtSteps, categoryScores: x.validation.categoryScores }));
      this.pendingArchiveMigration = [];
      this.pendingLegacyBest = null;
      this.archiveNeedsRebaseline = false;
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
    if (balancedConfirmed || confirmedCatastrophic.length >= 2) interpretation = 'confirmed-regression';
    else if (confirmedAlerts.length && !balancedEvidence && skillImprovements.length) interpretation = 'confirmed-specialization';
    else if (confirmedAlerts.length) interpretation = 'confirmed-skill-regression';
    else if (skillAssessment.alerts.length && !balancedEvidence && skillImprovements.length) interpretation = 'specialization-watch';
    else if (skillAssessment.alerts.length) interpretation = 'skill-regression-watch';
    else if (balancedEvidence) interpretation = 'balanced-regression-watch';

    const currentCandidate = this.makeCandidate(validation, 'latest');
    const archiveUpdates = this.considerCandidate(currentCandidate);
    updateSkillBestRecords(this.skillBestRecords, validation.stageResults, this.totalSteps);
    syncSkillBestScores(this);
    this.lastSkillValidation = validation;
    this.bestBrain = this.getArchiveBrain('balanced');

    const improved = archiveUpdates.includes('balanced');
    const regression = balancedEvidence;
    const rollbackEligible = balancedConfirmed || confirmedCatastrophic.length >= 2;
    let autoRollback = null;
    if (rollbackEligible && CONFIG.validation.autoRollback && this.bestBrain?.model && this.bestBrain.savedAtSteps !== this.totalSteps) {
      autoRollback = this.autoRecoverBalanced();
    }

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
      autoRollback,
      migratedArchive,
      bestSteps: this.bestBrain?.savedAtSteps ?? null,
      bestScore: this.bestBrain?.validation?.categoryScores?.balanced ?? null,
    };
    this.validationHistory.push(record);
    if (this.validationHistory.length > 64) this.validationHistory.shift();
    this.lastValidationStep = this.totalSteps;
    const regularNextValidation = nextValidationAfter(this.totalSteps);
    const needsConfirmation = Boolean(skillAssessment.alerts.length || balancedEvidence);
    this.nextValidationStep = autoRollback || needsConfirmation
      ? Math.min(regularNextValidation, this.totalSteps + CONFIG.validation.recoveryValidationInterval)
      : regularNextValidation;
    return record;
  }

  makeCandidate(validation, source = 'latest', base = null) {
    return {
      savedAtSteps: base?.savedAtSteps ?? this.totalSteps,
      savedAtEpisodes: base?.savedAtEpisodes ?? this.totalEpisodes,
      model: base?.model ?? this.model.serialize(),
      optimizer: base?.optimizer ?? this.trainer.serialize(),
      curriculum: base?.curriculum ?? this.curriculum.serialize(),
      validation,
      source,
    };
  }

  considerCandidate(candidate) {
    const updates = [];
    for (const category of ARCHIVE_CATEGORIES) {
      const score = candidate.validation?.categoryScores?.[category];
      if (!Number.isFinite(score)) continue;
      const prior = this.bestArchive[category];
      const priorScore = prior?.validation?.categoryScores?.[category];
      if (!Number.isFinite(priorScore) || score > priorScore + CONFIG.validation.improvementEpsilon) {
        this.bestArchive[category] = { ...candidate, category, categoryScore: score };
        updates.push(category);
      }
    }
    this.bestBrain = this.getArchiveBrain('balanced');
    return updates;
  }

  getArchiveBrain(category = 'balanced') {
    return this.bestArchive?.[category] || (category === 'balanced' ? this.bestBrain : null);
  }

  archiveSummary() {
    return Object.fromEntries(ARCHIVE_CATEGORIES.map(category => {
      const brain = this.getArchiveBrain(category);
      return [category, brain ? { savedAtSteps: brain.savedAtSteps, score: brain.validation?.categoryScores?.[category] ?? null } : null];
    }));
  }

  autoRecoverBalanced() {
    const brain = this.getArchiveBrain('balanced');
    if (!brain?.model) return null;
    const sourceSteps = brain.savedAtSteps;
    this.model.restore(brain.model);
    if (brain.optimizer) this.trainer.restore(brain.optimizer);
    this.trainer.learningRate = Math.max(CONFIG.ppo.minLearningRate, this.trainer.learningRate * CONFIG.validation.rollbackLearningRateFactor);
    this.trainer.enterRecoveryCooldown(CONFIG.ppo.lrRecoveryCooldownUpdates);
    // Keep the learner's current curriculum stage. The recovery target is policy stability,
    // not pretending the experience/curriculum clock moved backward.
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = { atSteps: this.totalSteps, sourceSteps, category: 'balanced', automatic: true, learningRate: this.trainer.learningRate };
    this.rollbackHistory.push(event);
    if (this.rollbackHistory.length > 32) this.rollbackHistory.shift();
    return event;
  }

  restoreBest(category = 'balanced') {
    const brain = this.getArchiveBrain(category);
    if (!brain?.model) throw new Error(`No validated ${category} brain is available yet`);
    const sourceSteps = brain.savedAtSteps;
    this.model.restore(brain.model);
    if (brain.optimizer) this.trainer.restore(brain.optimizer);
    if (brain.curriculum) this.curriculum.restore(brain.curriculum);
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = { atSteps: this.totalSteps, sourceSteps, category };
    this.rollbackHistory.push(event);
    if (this.rollbackHistory.length > 32) this.rollbackHistory.shift();
    return event;
  }

  snapshot() {
    return {
      schema: 4,
      seed: this.seed,
      totalSteps: this.totalSteps,
      totalEpisodes: this.totalEpisodes,
      envSeedCursor: this.envSeedCursor,
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
      rollbackHistory: this.rollbackHistory,
    };
  }

  restore(data) {
    if (!data || ![1, 2, 3, 4].includes(data.schema)) throw new Error('Unsupported checkpoint schema');
    this.seed = data.seed;
    this.actionRng = new PRNG(this.seed ^ 0xa5a5a5a5);
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

    if (data.schema === 4) {
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
    } else if (data.schema === 3) {
      // v0.1.1 used validation:v2/retention-v2. Preserve every specialist model for
      // inspection, but schedule an immediate v3 rebaseline before training advances.
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
    }

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

function defaultRetentionStatus() {
  return { alerts: [], forgetting: [], healthy: false, interpretation: 'unvalidated', confirmed: false };
}

function normalizeRetentionStatus(value) {
  if (!value || typeof value !== 'object') return defaultRetentionStatus();
  const alerts = Array.isArray(value.alerts) ? value.alerts : (Array.isArray(value.forgetting) ? value.forgetting : []);
  return { ...defaultRetentionStatus(), ...value, alerts, forgetting: alerts };
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
    autoRollback: record.autoRollback,
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
