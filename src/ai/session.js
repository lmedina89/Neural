import { BUILD_MARKER, CONFIG, VERSION } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { evaluateFullRetentionSuite } from '../evaluation/evaluator.js';
import { RecurrentActorCritic } from './model.js';
import { CuriosityModule } from './curiosity.js';
import { PPOTrainer } from './ppo.js';
import { computeGAE } from './rollout.js';

const ARCHIVE_CATEGORIES = ['balanced', 'overall', 'forager', 'survivor', 'efficiency'];

export class TrainingSession {
  constructor({ seed = 1337, envCount = CONFIG.runtime.trainEnvs, autoCurriculum = true } = {}) {
    this.seed = seed;
    this.autoCurriculum = autoCurriculum;
    this.model = new RecurrentActorCritic(seed);
    this.trainer = new PPOTrainer(this.model, seed ^ 0x9e3779b9);
    this.curiosity = new CuriosityModule(seed ^ 0xc0decafe);
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
    this.curiosityEpisodeBudget = [];
    this.curiosityEpisodeStats = [];
    this.curiosityEpisodeHistory = [];
    this.curiosityBudgetResets = 0;
    this.curiosityRewardMode = 'reward';
    this.curiosityAudit = defaultCuriosityAudit();
    this.auditRole = null;
    this.lastCuriosity = null;
    this.curiosityTrail = [];
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
    this.curiosityEpisodeBudget[id] = CONFIG.curiosity.maxEpisodeBonus;
    this.curiosityEpisodeStats[id] = newCuriosityEpisodeStats();
  }

  resetEnv(id) {
    const cursor = this.envSeedCursor++;
    const stageIndex = this.chooseTrainingStage(cursor);
    const seed = domainSeed(`train:${this.seed}:stage:${stageIndex}`, cursor);
    const env = new World(seed, CURRICULUM[stageIndex]);
    this.envs[id] = env;
    this.obs[id] = env.observe();
    this.hidden[id] = this.model.zeroHidden();
    this.curiosityEpisodeBudget[id] = CONFIG.curiosity.maxEpisodeBonus;
    this.curiosityEpisodeStats[id] = newCuriosityEpisodeStats();
  }

  setCuriosityRewardMode(mode = 'reward') {
    if (!['reward', 'observe'].includes(mode)) throw new Error(`Unsupported curiosity reward mode: ${mode}`);
    if (this.curiosityAudit?.active) throw new Error('Curiosity reward mode is locked by the active A/B audit');
    this.curiosityRewardMode = mode;
    return this.curiosityRewardMode;
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
    let intrinsicRewardSum = 0;
    let potentialIntrinsicRewardSum = 0;
    let externalRewardMagnitudeSum = 0;
    let curiosityErrorSum = 0;
    let curiosityNoveltySum = 0;
    let curiositySampleCount = 0;

    // During a controlled curiosity A/B audit, ordinary validation/promotion is
    // deliberately suspended. Audit evaluations use their own fixed seed domain
    // and are read-only, so neither branch can replace a Champion mid-experiment.
    const auditActive = Boolean(this.curiosityAudit?.active);
    const preValidationStarted = performanceNow();
    const preValidation = !auditActive && this.totalSteps >= this.nextValidationStep && this.lastValidationStep !== this.totalSteps
      ? this.runValidation()
      : null;
    if (preValidation) validationMs += performanceNow() - preValidationStarted;

    const simulationStarted = performanceNow();
    for (let s = 0; s < steps; s++) {
      for (let i = 0; i < this.envs.length; i++) {
        const obs = this.obs[i];
        const hPrev = this.hidden[i];
        const act = this.model.act(obs, hPrev, this.actionRng, false, false);
        const result = this.envs[i].step(act.action);
        const episodeStats = this.curiosityEpisodeStats[i] || (this.curiosityEpisodeStats[i] = newCuriosityEpisodeStats());
        episodeStats.externalReward += Number(result.reward) || 0;
        externalRewardMagnitudeSum += Math.abs(Number(result.reward) || 0);

        const remainingBudget = Number.isFinite(this.curiosityEpisodeBudget[i])
          ? this.curiosityEpisodeBudget[i]
          : CONFIG.curiosity.maxEpisodeBonus;
        // Curiosity is intentionally sampled rather than evaluated on every world
        // transition. Each sample is still a genuine one-step prediction; sparse
        // sampling preserves the signal while protecting mobile training throughput.
        const curiositySampled = this.envs[i].stepCount === 1 || (this.envs[i].stepCount % CONFIG.curiosity.sampleStride === 0);
        let curiosity = null;
        let potentialIntrinsicReward = 0;
        let intrinsicReward = 0;
        if (curiositySampled) {
          curiosity = this.curiosity.scoreTransition(obs, act.action, result.obs, {
            remainingBudget,
            terminal: result.done,
            capture: i === 0,
          });
          potentialIntrinsicReward = curiosity.bonus;
          intrinsicReward = this.curiosityRewardMode === 'reward' ? potentialIntrinsicReward : 0;
          this.curiosityEpisodeBudget[i] = Math.max(0, remainingBudget - potentialIntrinsicReward);
          intrinsicRewardSum += intrinsicReward;
          potentialIntrinsicRewardSum += potentialIntrinsicReward;
          curiosityErrorSum += curiosity.error;
          curiosityNoveltySum += curiosity.novelty;
          curiositySampleCount++;

          episodeStats.potentialIntrinsicReward += potentialIntrinsicReward;
          episodeStats.appliedIntrinsicReward += intrinsicReward;
          episodeStats.samples++;
          episodeStats.noveltySum += curiosity.novelty;
          episodeStats.noveltyMax = Math.max(episodeStats.noveltyMax, curiosity.novelty);
          episodeStats.errorSum += curiosity.error;
          episodeStats.errorMax = Math.max(episodeStats.errorMax, curiosity.error);
          if (episodeStats.budgetExhaustedAtStep == null && remainingBudget > 1e-12 && this.curiosityEpisodeBudget[i] <= 1e-12) {
            episodeStats.budgetExhaustedAtStep = this.envs[i].stepCount;
          }

          if (i === 0 && curiosity.obs) {
            this.lastCuriosity = {
              ...curiosity,
              potentialBonus: potentialIntrinsicReward,
              appliedBonus: intrinsicReward,
              rewardMode: this.curiosityRewardMode,
              auditRole: this.auditRole,
              x: this.envs[i].agent.x,
              y: this.envs[i].agent.y,
              budgetRemaining: this.curiosityEpisodeBudget[i],
              budgetUsed: CONFIG.curiosity.maxEpisodeBonus - this.curiosityEpisodeBudget[i],
              atSteps: this.totalSteps + 1,
            };
            if (curiosity.novelty > 0.05) {
              this.curiosityTrail.push({ x: this.envs[i].agent.x, y: this.envs[i].agent.y, novelty: curiosity.novelty, atSteps: this.totalSteps + 1 });
              if (this.curiosityTrail.length > 72) this.curiosityTrail.shift();
            }
          }
        }

        transitions.push({
          env: i,
          obs: new Float64Array(obs),
          nextObs: curiositySampled ? new Float64Array(result.obs) : null,
          hPrev: new Float64Array(hPrev),
          action: act.action,
          logProb: act.logProb,
          value: act.value,
          extrinsicReward: result.reward,
          potentialIntrinsicReward,
          intrinsicReward,
          reward: result.reward + intrinsicReward,
          curiosityError: curiosity?.error ?? 0,
          curiosityNovelty: curiosity?.novelty ?? 0,
          curiositySampled,
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

          const curiosityEpisode = finalizeCuriosityEpisodeStats(episodeStats, info, this.curiosityEpisodeBudget[i], this.curiosityRewardMode);
          this.curiosityEpisodeHistory.push(curiosityEpisode);
          if (this.curiosityEpisodeHistory.length > 160) this.curiosityEpisodeHistory.shift();
          this.curiosityBudgetResets++;

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
    // A/B descendants must see the same PPO schedule age at the same branch-local
    // progress. Using monotonically increasing global steps here would make the
    // second-tested branch subtly different even when both began from identical
    // bytes, which would contaminate the ablation.
    const ppoTrainingStep = auditActive
      ? Math.max(0, Number(this.curiosityAudit?.originSteps) || 0) + this.learnerExperienceSteps
      : this.totalSteps;
    const ppo = this.trainer.update(transitions, advantages, returns, { trainingStep: ppoTrainingStep });
    const ppoMs = performanceNow() - ppoStarted;
    const curiosityStarted = performanceNow();
    const curiosityTrain = this.curiosity.trainBatch(transitions);
    const curiosityMs = performanceNow() - curiosityStarted;
    const coreMs = Math.max(0.001, simulationMs + advantageMs + ppoMs + curiosityMs);
    const recent = this.episodeHistory.slice(-40);
    const mean = key => recent.length ? recent.reduce((sum, x) => sum + (x[key] || 0), 0) / recent.length : 0;

    const bookkeepingStarted = performanceNow();
    // Ordinary historical milestones are intentionally suspended during an A/B
    // audit. Audit checkpoints live in their own read-only result stream so a
    // branch cannot masquerade as the single historical Learner trajectory.
    if (!auditActive) this.maybeSaveMilestones();
    const forceValidation = Boolean(curriculumEvent && curriculumEvent.reason !== 'promotion-blocked');
    const bookkeepingBeforeValidationMs = performanceNow() - bookkeepingStarted;
    const postValidationStarted = performanceNow();
    const postValidation = auditActive ? null : this.maybeAutoValidate(forceValidation);
    if (postValidation) validationMs += performanceNow() - postValidationStarted;
    const validation = postValidation || preValidation;

    const auditValidationStarted = performanceNow();
    const auditEvaluation = this.maybeRunCuriosityAuditEvaluation();
    if (auditEvaluation) validationMs += performanceNow() - auditValidationStarted;

    const trainingStepsPerSec = transitions.length / (coreMs / 1000);
    const simulationStepsPerSec = transitions.length / Math.max(0.001, simulationMs / 1000);
    const episodeDiagnostics = this.curiosityDiagnosticsSummary();
    const rewardMagnitudeShare = intrinsicRewardSum / Math.max(1e-12, externalRewardMagnitudeSum + intrinsicRewardSum);
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
        curiosityMs,
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
      curiosity: {
        rewardMode: this.curiosityRewardMode,
        auditRole: this.auditRole,
        meanIntrinsicReward: intrinsicRewardSum / Math.max(1, transitions.length),
        intrinsicRewardSum,
        meanPotentialIntrinsicReward: potentialIntrinsicRewardSum / Math.max(1, transitions.length),
        potentialIntrinsicRewardSum,
        rewardMagnitudeShare,
        meanPredictionError: curiosityErrorSum / Math.max(1, curiositySampleCount),
        meanNovelty: curiosityNoveltySum / Math.max(1, curiositySampleCount),
        samples: curiositySampleCount,
        sampleFraction: curiositySampleCount / Math.max(1, transitions.length),
        predictorLoss: curiosityTrain.loss,
        predictorParams: this.curiosity.paramCount(),
        errorBaseline: this.curiosity.errorMean,
        budgetResets: this.curiosityBudgetResets,
        episodeDiagnostics,
      },
      curiosityAudit: this.curiosityAuditSummary(),
      ...ppo,
    };
    this.metrics.push(metric);
    if (this.metrics.length > CONFIG.runtime.chartPoints) this.metrics.shift();
    const auditBranchComplete = Boolean(this.curiosityAudit?.active && this.auditRole && this.auditBranchProgress(this.auditRole) >= this.curiosityAudit.targetStepsPerBranch);
    return { metric, completed, transitions: transitions.length, validation, curriculumEvent, auditEvaluation, auditBranchComplete };
  }

  curiosityDiagnosticsSummary(limit = 80) {
    const rows = this.curiosityEpisodeHistory.slice(-Math.max(1, limit));
    if (!rows.length) {
      return {
        episodes: 0,
        budgetResets: this.curiosityBudgetResets,
        budgetExhaustionRate: 0,
        meanBudgetUsed: 0,
        meanBudgetUseFraction: 0,
        meanExhaustionFraction: null,
        meanPotentialIntrinsicPerEpisode: 0,
        meanAppliedIntrinsicPerEpisode: 0,
        meanNovelty: 0,
        maxNovelty: 0,
        meanPredictionError: 0,
        maxPredictionError: 0,
      };
    }
    const mean = key => rows.reduce((sum, row) => sum + (Number(row[key]) || 0), 0) / rows.length;
    const exhausted = rows.filter(row => Number.isFinite(row.budgetExhaustionFraction));
    const sampled = rows.filter(row => (Number(row.samples) || 0) > 0);
    const sampledWeight = sampled.reduce((sum, row) => sum + (Number(row.samples) || 0), 0);
    const noveltyWeighted = sampled.reduce((sum, row) => sum + (Number(row.meanNovelty) || 0) * (Number(row.samples) || 0), 0);
    const errorWeighted = sampled.reduce((sum, row) => sum + (Number(row.meanPredictionError) || 0) * (Number(row.samples) || 0), 0);
    return {
      episodes: rows.length,
      budgetResets: this.curiosityBudgetResets,
      budgetExhaustionRate: exhausted.length / rows.length,
      meanBudgetUsed: mean('budgetUsed'),
      meanBudgetUseFraction: mean('budgetUseFraction'),
      meanExhaustionFraction: exhausted.length ? exhausted.reduce((sum, row) => sum + row.budgetExhaustionFraction, 0) / exhausted.length : null,
      meanPotentialIntrinsicPerEpisode: mean('potentialIntrinsicReward'),
      meanAppliedIntrinsicPerEpisode: mean('appliedIntrinsicReward'),
      meanNovelty: sampledWeight ? noveltyWeighted / sampledWeight : 0,
      maxNovelty: rows.reduce((m, row) => Math.max(m, Number(row.maxNovelty) || 0), 0),
      meanPredictionError: sampledWeight ? errorWeighted / sampledWeight : 0,
      maxPredictionError: rows.reduce((m, row) => Math.max(m, Number(row.maxPredictionError) || 0), 0),
    };
  }

  auditBranchProgress(role) {
    if (!role) return 0;
    if (this.curiosityAudit?.active && this.auditRole === role) return Math.max(0, Number(this.learnerExperienceSteps) || 0);
    const branch = this.curiosityAudit?.branches?.[role];
    const branchId = branch?.lineageId;
    if (branchId) {
      const frozen = this.frozenLearners.find(x => x.id === branchId);
      if (frozen) return Math.max(0, Number(frozen.learnerExperienceSteps) || 0);
    }
    return Math.max(0, Number(branch?.progress) || 0);
  }

  curiosityAuditSummary() {
    const audit = this.curiosityAudit || defaultCuriosityAudit();
    const controlProgress = this.auditBranchProgress('control');
    const curiosityProgress = this.auditBranchProgress('curiosity');
    const rows = Array.isArray(audit.results) ? audit.results : [];
    const compactResults = rows.map(row => ({
      role: row.role,
      checkpointSteps: row.checkpointSteps,
      experienceSteps: row.experienceSteps,
      globalSteps: row.globalSteps,
      balancedScore: row.validation?.balancedScore ?? null,
      balancedCiLow: row.validation?.balancedCiLow ?? null,
      balancedCiHigh: row.validation?.balancedCiHigh ?? null,
      meanReturn: row.validation?.meanReturn ?? null,
      meanFood: row.validation?.meanFood ?? null,
      survivalRate: row.validation?.survivalRate ?? null,
      stageScores: row.validation?.stageResults?.map(x => x.skillScore) || [],
    }));
    const paired = pairedAuditComparison(compactResults);
    return {
      active: Boolean(audit.active),
      id: audit.id || null,
      role: this.auditRole,
      rewardMode: this.curiosityRewardMode,
      originSteps: audit.originSteps ?? null,
      targetStepsPerBranch: audit.targetStepsPerBranch || CONFIG.curiosityAudit.targetStepsPerBranch,
      checkpointInterval: audit.checkpointInterval || CONFIG.curiosityAudit.checkpointInterval,
      controlProgress,
      curiosityProgress,
      controlLineageId: audit.branches?.control?.lineageId || null,
      curiosityLineageId: audit.branches?.curiosity?.lineageId || null,
      results: compactResults,
      comparison: paired,
      completed: Boolean(audit.completed),
    };
  }

  startCuriosityAudit({
    targetStepsPerBranch = CONFIG.curiosityAudit.targetStepsPerBranch,
    checkpointInterval = CONFIG.curiosityAudit.checkpointInterval,
  } = {}) {
    if (this.curiosityAudit?.active) throw new Error('A curiosity A/B audit is already active');
    const target = Math.max(100_000, Number(targetStepsPerBranch) || CONFIG.curiosityAudit.targetStepsPerBranch);
    const interval = Math.max(50_000, Math.min(target, Number(checkpointInterval) || CONFIG.curiosityAudit.checkpointInterval));

    // Freeze the exact pre-audit learner as a recoverable origin before creating
    // either descendant. This is a preservation action, never a rollback.
    const origin = this.freezeCurrentLearner('preserved-before-curiosity-ab-audit');
    const auditId = `CURAUD-${this.totalSteps}-${this.lineageCounter + 1}`;
    const preAutoCurriculum = this.autoCurriculum;
    const preCuriosityRewardMode = this.curiosityRewardMode;

    this.lineageCounter++;
    const controlLineage = {
      id: `L-${(this.seed >>> 0).toString(16)}-${this.lineageCounter}`,
      startedAtSteps: this.totalSteps,
      parent: { type: 'curiosity-audit-origin', auditId, lineageId: origin.id, sourceSteps: this.totalSteps },
      reason: 'curiosity-ablation-control',
    };
    this.lineageCounter++;
    const curiosityLineage = {
      id: `L-${(this.seed >>> 0).toString(16)}-${this.lineageCounter}`,
      startedAtSteps: this.totalSteps,
      parent: { type: 'curiosity-audit-origin', auditId, lineageId: origin.id, sourceSteps: this.totalSteps },
      reason: 'curiosity-ablation-reward',
    };

    const makeAuditFrozen = (lineage, role, rewardMode) => ({
      id: lineage.id,
      lineage: structuredLineage(lineage),
      frozenAtSteps: this.totalSteps,
      frozenAtEpisodes: this.totalEpisodes,
      learnerExperienceSteps: 0,
      reason: 'curiosity-ab-audit-seed',
      curriculum: cloneSerializable(origin.curriculum),
      envSeedCursor: origin.envSeedCursor,
      actionRngState: origin.actionRngState,
      rehearsalEpisodeHistory: [...(origin.rehearsalEpisodeHistory || [])],
      episodeHistory: [],
      model: cloneSerializable(origin.model),
      optimizer: cloneSerializable(origin.optimizer),
      curiosity: cloneSerializable(origin.curiosity),
      curiosityRewardMode: rewardMode,
      curiosityEpisodeHistory: [],
      curiosityBudgetResets: 0,
      auditRole: role,
      auditId,
    });

    const curiosityBranch = makeAuditFrozen(curiosityLineage, 'curiosity', 'reward');
    const existing = this.frozenLearners.findIndex(x => x.id === curiosityBranch.id);
    if (existing >= 0) this.frozenLearners[existing] = curiosityBranch;
    else this.frozenLearners.push(curiosityBranch);

    // Activate the CONTROL descendant from the exact same frozen bytes. The
    // predictor still learns and its potential reward is measured, but PPO gets 0.
    this.model.restore(origin.model);
    this.trainer.restore(origin.optimizer);
    this.curiosity = new CuriosityModule(this.seed ^ 0xc0decafe ^ this.lineageCounter);
    this.curiosity.restore(origin.curiosity);
    this.curriculum.restore(origin.curriculum || {});
    this.envSeedCursor = Number(origin.envSeedCursor) || 0;
    if (Number.isFinite(Number(origin.actionRngState))) this.actionRng.state = Number(origin.actionRngState) >>> 0;
    this.rehearsalEpisodeHistory = Array.isArray(origin.rehearsalEpisodeHistory)
      ? origin.rehearsalEpisodeHistory.slice(-CONFIG.continual.recentMixWindow)
      : [];
    this.learnerLineage = controlLineage;
    this.learnerExperienceSteps = 0;
    this.episodeHistory = [];
    this.curiosityRewardMode = 'observe';
    this.auditRole = 'control';
    this.curiosityEpisodeHistory = [];
    this.curiosityBudgetResets = 0;
    this.lastCuriosity = null;
    this.curiosityTrail = [];
    this.autoCurriculum = false;
    this.lineageHistory.push(structuredLineage(controlLineage), structuredLineage(curiosityLineage));
    if (this.lineageHistory.length > 64) this.lineageHistory = this.lineageHistory.slice(-64);

    const baseline = evaluateFullRetentionSuite(this.model, {
      episodesPerStage: CONFIG.curiosityAudit.episodesPerStage,
      seedBase: CONFIG.curiosityAudit.seedBase,
      deterministic: false,
      protocolTag: 'curiosity-ablation-v1',
    });
    this.curiosityAudit = {
      active: true,
      completed: false,
      id: auditId,
      originLineageId: origin.id,
      originSteps: this.totalSteps,
      startedAtSteps: this.totalSteps,
      targetStepsPerBranch: target,
      checkpointInterval: interval,
      seedBase: CONFIG.curiosityAudit.seedBase,
      preAutoCurriculum,
      preCuriosityRewardMode,
      branches: {
        control: { lineageId: controlLineage.id, nextCheckpointSteps: interval, complete: false, progress: 0 },
        curiosity: { lineageId: curiosityLineage.id, nextCheckpointSteps: interval, complete: false, progress: 0 },
      },
      results: [{
        role: 'origin',
        checkpointSteps: 0,
        experienceSteps: 0,
        globalSteps: this.totalSteps,
        validation: baseline,
      }],
    };

    // Both descendants begin from the same environment cursor and RNG state.
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    return this.curiosityAuditSummary();
  }

  maybeRunCuriosityAuditEvaluation() {
    const audit = this.curiosityAudit;
    const role = this.auditRole;
    if (!audit?.active || !role || !audit.branches?.[role]) return null;
    const branch = audit.branches[role];
    const progress = Math.max(0, Number(this.learnerExperienceSteps) || 0);
    const target = Math.max(1, Number(audit.targetStepsPerBranch) || CONFIG.curiosityAudit.targetStepsPerBranch);
    const interval = Math.max(1, Number(audit.checkpointInterval) || CONFIG.curiosityAudit.checkpointInterval);
    const next = Math.min(target, Math.max(interval, Number(branch.nextCheckpointSteps) || interval));
    if (progress < next) return null;

    const validation = evaluateFullRetentionSuite(this.model, {
      episodesPerStage: CONFIG.curiosityAudit.episodesPerStage,
      seedBase: audit.seedBase || CONFIG.curiosityAudit.seedBase,
      deterministic: false,
      protocolTag: 'curiosity-ablation-v1',
    });
    const record = {
      role,
      checkpointSteps: next,
      experienceSteps: progress,
      globalSteps: this.totalSteps,
      validation,
    };
    audit.results.push(record);
    branch.progress = progress;
    branch.lastCheckpointSteps = next;
    branch.lastScore = validation.balancedScore;
    branch.nextCheckpointSteps = next >= target ? target : Math.min(target, next + interval);
    branch.complete = progress >= target;
    if (audit.branches.control.complete && audit.branches.curiosity.complete) {
      audit.completed = true;
      audit.completedAtSteps = this.totalSteps;
    }
    return record;
  }

  switchCuriosityAuditBranch(role) {
    if (!this.curiosityAudit?.active) throw new Error('No curiosity A/B audit is active');
    if (!['control', 'curiosity'].includes(role)) throw new Error(`Unknown audit branch: ${role}`);
    if (this.auditRole === role) return { unchanged: true, role, lineageId: this.learnerLineage?.id || null };
    const lineageId = this.curiosityAudit.branches?.[role]?.lineageId;
    if (!lineageId) throw new Error(`Audit branch ${role} is unavailable`);
    const event = this.switchToFrozenLearner(lineageId);
    this.autoCurriculum = false;
    return { ...event, role };
  }

  endCuriosityAudit() {
    if (!this.curiosityAudit?.active) return this.curiosityAuditSummary();
    const audit = this.curiosityAudit;
    if (this.auditRole && audit.branches?.[this.auditRole]) {
      audit.branches[this.auditRole].progress = Math.max(0, Number(this.learnerExperienceSteps) || 0);
    }
    audit.active = false;
    audit.endedAtSteps = this.totalSteps;
    this.autoCurriculum = audit.preAutoCurriculum !== false;
    this.curiosityRewardMode = ['reward', 'observe'].includes(audit.preCuriosityRewardMode) ? audit.preCuriosityRewardMode : 'reward';
    this.auditRole = null;
    // The experiment never promotes a winner. Resume ordinary validation from the
    // active policy on the next training cycle so Champion logic remains explicit.
    this.nextValidationStep = this.totalSteps;
    // Do not backfill ordinary milestone labels with A/B-branch weights. Resume
    // the single-learner historical schedule strictly after the audit's global age.
    this.nextMilestoneStep = nextHistoricalMilestoneAfter(this.totalSteps);
    return this.curiosityAuditSummary();
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
      curiosity: base?.curiosity ?? this.curiosity.serialize(),
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
      curiosity: cloneSerializable(brain.curiosity || null),
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
      episodeHistory: cloneSerializable(this.episodeHistory),
      model: this.model.serialize(),
      optimizer: this.trainer.serialize(),
      curiosity: this.curiosity.serialize(),
      curiosityRewardMode: this.curiosityRewardMode,
      curiosityEpisodeHistory: cloneSerializable(this.curiosityEpisodeHistory),
      curiosityBudgetResets: this.curiosityBudgetResets,
      auditRole: this.auditRole,
      auditId: this.curiosityAudit?.active ? this.curiosityAudit.id : null,
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
      curiosityRewardMode: x.curiosityRewardMode || 'reward',
      auditRole: x.auditRole || null,
      auditId: x.auditId || null,
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
    if (this.curiosityAudit?.active) throw new Error('End the active curiosity A/B audit before creating a different learner fork');
    if (!source?.model) throw new Error('Fork source has no model');
    const frozen = this.freezeCurrentLearner('preserved-before-fork');
    this.model.restore(source.model);
    if (source.optimizer) this.trainer.restore(source.optimizer);
    else this.trainer = new PPOTrainer(this.model, this.seed ^ 0x9e3779b9 ^ (this.lineageCounter + 1));
    this.curiosity = new CuriosityModule(this.seed ^ 0xc0decafe ^ (this.lineageCounter + 1));
    if (source.curiosity) this.curiosity.restore(source.curiosity);
    this.lastCuriosity = null;
    this.curiosityTrail = [];
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
    if (this.curiosityAudit?.active && target.auditId !== this.curiosityAudit.id) {
      throw new Error('End the active curiosity A/B audit before switching to an unrelated learner branch');
    }
    this.frozenLearners.splice(idx, 1);
    if (this.curiosityAudit?.active && this.auditRole && this.curiosityAudit.branches?.[this.auditRole]) {
      this.curiosityAudit.branches[this.auditRole].progress = Math.max(0, Number(this.learnerExperienceSteps) || 0);
    }
    const preserved = this.freezeCurrentLearner('preserved-before-lineage-switch');
    this.model.restore(target.model);
    this.trainer.restore(target.optimizer);
    this.curiosity = new CuriosityModule(this.seed ^ 0xc0decafe ^ (this.lineageCounter + 1));
    if (target.curiosity) this.curiosity.restore(target.curiosity);
    this.curiosityRewardMode = ['reward', 'observe'].includes(target.curiosityRewardMode) ? target.curiosityRewardMode : 'reward';
    this.curiosityEpisodeHistory = Array.isArray(target.curiosityEpisodeHistory) ? cloneSerializable(target.curiosityEpisodeHistory).slice(-160) : [];
    this.curiosityBudgetResets = Math.max(0, Number(target.curiosityBudgetResets) || 0);
    this.auditRole = target.auditRole || null;
    this.lastCuriosity = null;
    this.curiosityTrail = [];
    this.curriculum.restore(target.curriculum || {});
    this.envSeedCursor = Number(target.envSeedCursor) || this.envSeedCursor;
    if (Number.isFinite(Number(target.actionRngState))) this.actionRng.state = Number(target.actionRngState) >>> 0;
    this.rehearsalEpisodeHistory = Array.isArray(target.rehearsalEpisodeHistory)
      ? target.rehearsalEpisodeHistory.slice(-CONFIG.continual.recentMixWindow)
      : [];
    this.episodeHistory = Array.isArray(target.episodeHistory) ? cloneSerializable(target.episodeHistory).slice(-200) : [];
    this.learnerLineage = structuredLineage(target.lineage) || this.learnerLineage;
    this.learnerExperienceSteps = Math.max(0, Number(target.learnerExperienceSteps) || 0);
    if (!this.curiosityAudit?.active) this.resetBranchValidationState();
    else this.autoCurriculum = false;
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = {
      atSteps: this.totalSteps,
      automatic: false,
      type: this.curiosityAudit?.active ? 'curiosity-audit-switch' : 'manual-lineage-switch',
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
      schema: 8,
      seed: this.seed,
      totalSteps: this.totalSteps,
      totalEpisodes: this.totalEpisodes,
      envSeedCursor: this.envSeedCursor,
      actionRngState: this.actionRng.state >>> 0,
      curriculum: this.curriculum.serialize(),
      model: this.model.serialize(),
      optimizer: this.trainer.serialize(),
      curiosity: this.curiosity.serialize(),
      curiosityRewardMode: this.curiosityRewardMode,
      curiosityEpisodeHistory: cloneSerializable(this.curiosityEpisodeHistory),
      curiosityBudgetResets: this.curiosityBudgetResets,
      curiosityAudit: cloneSerializable(this.curiosityAudit),
      auditRole: this.auditRole,
      autoCurriculum: this.autoCurriculum,
      episodeHistory: cloneSerializable(this.episodeHistory),
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
    if (!data || ![1, 2, 3, 4, 5, 6, 7, 8].includes(data.schema)) throw new Error('Unsupported checkpoint schema');
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
    this.curiosity = new CuriosityModule(this.seed ^ 0xc0decafe);
    if (data.schema >= 7 && data.curiosity) this.curiosity.restore(data.curiosity);
    this.curiosityRewardMode = data.schema >= 8 && ['reward', 'observe'].includes(data.curiosityRewardMode) ? data.curiosityRewardMode : 'reward';
    this.curiosityEpisodeHistory = data.schema >= 8 && Array.isArray(data.curiosityEpisodeHistory)
      ? cloneSerializable(data.curiosityEpisodeHistory).slice(-160)
      : [];
    this.curiosityBudgetResets = data.schema >= 8 ? Math.max(0, Number(data.curiosityBudgetResets) || 0) : 0;
    this.curiosityAudit = data.schema >= 8 && data.curiosityAudit ? normalizeCuriosityAudit(data.curiosityAudit) : defaultCuriosityAudit();
    this.auditRole = data.schema >= 8 ? (data.auditRole || null) : null;
    this.autoCurriculum = this.curiosityAudit.active ? false : (data.schema >= 8 && typeof data.autoCurriculum === 'boolean' ? data.autoCurriculum : this.autoCurriculum);
    this.lastCuriosity = null;
    this.curiosityTrail = [];
    this.curriculum.restore(data.curriculum || {});
    if (data.schema < 3) this.curriculum.cooldownRemaining = Math.max(this.curriculum.cooldownRemaining, CONFIG.curriculum.transitionCooldownEpisodes);
    this.episodeHistory = data.schema >= 8 && Array.isArray(data.episodeHistory) ? cloneSerializable(data.episodeHistory).slice(-200) : [];
    this.metrics = Array.isArray(data.metrics) ? data.metrics.slice(-CONFIG.runtime.chartPoints) : [];
    this.milestones = new Map(Array.isArray(data.milestones) ? data.milestones : []);
    this.nextMilestoneStep = Number.isFinite(data.nextMilestoneStep) && data.nextMilestoneStep > this.totalSteps
      ? data.nextMilestoneStep
      : nextHistoricalMilestoneAfter(this.totalSteps);
    if (data.schema === 1 && this.totalSteps > 0 && !this.milestones.has(this.totalSteps)) this.saveMilestone(this.totalSteps);

    if (data.schema === 8 || data.schema === 7 || data.schema === 6 || data.schema === 5) {
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
      if (data.schema >= 6) {
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

function newCuriosityEpisodeStats() {
  return {
    externalReward: 0,
    potentialIntrinsicReward: 0,
    appliedIntrinsicReward: 0,
    samples: 0,
    noveltySum: 0,
    noveltyMax: 0,
    errorSum: 0,
    errorMax: 0,
    budgetExhaustedAtStep: null,
  };
}

function finalizeCuriosityEpisodeStats(stats, info, budgetRemaining, rewardMode) {
  const samples = Math.max(0, Number(stats?.samples) || 0);
  const steps = Math.max(1, Number(info?.steps) || 1);
  const remaining = Math.max(0, Math.min(CONFIG.curiosity.maxEpisodeBonus, Number(budgetRemaining) || 0));
  const used = CONFIG.curiosity.maxEpisodeBonus - remaining;
  const exhaustedAt = Number.isFinite(stats?.budgetExhaustedAtStep) ? Number(stats.budgetExhaustedAtStep) : null;
  return {
    stageId: Number(info?.stageId),
    steps,
    rewardMode,
    externalReward: Number(stats?.externalReward) || 0,
    potentialIntrinsicReward: Number(stats?.potentialIntrinsicReward) || 0,
    appliedIntrinsicReward: Number(stats?.appliedIntrinsicReward) || 0,
    samples,
    meanNovelty: samples ? (Number(stats?.noveltySum) || 0) / samples : 0,
    maxNovelty: Number(stats?.noveltyMax) || 0,
    meanPredictionError: samples ? (Number(stats?.errorSum) || 0) / samples : 0,
    maxPredictionError: Number(stats?.errorMax) || 0,
    budgetRemaining: remaining,
    budgetUsed: used,
    budgetUseFraction: CONFIG.curiosity.maxEpisodeBonus > 0 ? used / CONFIG.curiosity.maxEpisodeBonus : 0,
    budgetExhaustedAtStep: exhaustedAt,
    budgetExhaustionFraction: exhaustedAt == null ? null : Math.max(0, Math.min(1, exhaustedAt / steps)),
  };
}

function defaultCuriosityAudit() {
  return {
    active: false,
    completed: false,
    id: null,
    originLineageId: null,
    originSteps: null,
    startedAtSteps: null,
    targetStepsPerBranch: CONFIG.curiosityAudit.targetStepsPerBranch,
    checkpointInterval: CONFIG.curiosityAudit.checkpointInterval,
    seedBase: CONFIG.curiosityAudit.seedBase,
    preAutoCurriculum: true,
    preCuriosityRewardMode: 'reward',
    branches: {
      control: { lineageId: null, nextCheckpointSteps: CONFIG.curiosityAudit.checkpointInterval, complete: false, progress: 0 },
      curiosity: { lineageId: null, nextCheckpointSteps: CONFIG.curiosityAudit.checkpointInterval, complete: false, progress: 0 },
    },
    results: [],
  };
}

function normalizeCuriosityAudit(value) {
  const base = defaultCuriosityAudit();
  if (!value || typeof value !== 'object') return base;
  const out = { ...base, ...cloneSerializable(value) };
  out.active = Boolean(value.active);
  out.completed = Boolean(value.completed);
  out.targetStepsPerBranch = Math.max(100_000, Number(value.targetStepsPerBranch) || CONFIG.curiosityAudit.targetStepsPerBranch);
  out.checkpointInterval = Math.max(50_000, Number(value.checkpointInterval) || CONFIG.curiosityAudit.checkpointInterval);
  out.seedBase = value.seedBase || CONFIG.curiosityAudit.seedBase;
  out.branches = {
    control: { ...base.branches.control, ...(value.branches?.control || {}) },
    curiosity: { ...base.branches.curiosity, ...(value.branches?.curiosity || {}) },
  };
  out.results = Array.isArray(value.results) ? cloneSerializable(value.results).slice(-32) : [];
  return out;
}

function pairedAuditComparison(results) {
  const rows = Array.isArray(results) ? results : [];
  const controls = new Map(rows.filter(x => x.role === 'control').map(x => [Number(x.checkpointSteps) || 0, x]));
  const curiosity = new Map(rows.filter(x => x.role === 'curiosity').map(x => [Number(x.checkpointSteps) || 0, x]));
  const checkpoints = [...controls.keys()].filter(x => curiosity.has(x)).sort((a, b) => a - b);
  if (!checkpoints.length) return { checkpointSteps: null, controlScore: null, curiosityScore: null, delta: null, interpretation: 'awaiting-paired-result' };
  const checkpointSteps = checkpoints.at(-1);
  const c = controls.get(checkpointSteps);
  const q = curiosity.get(checkpointSteps);
  const controlScore = Number(c?.balancedScore);
  const curiosityScore = Number(q?.balancedScore);
  const delta = curiosityScore - controlScore;
  const margin = CONFIG.generalization.conflictMargin;
  let interpretation = 'no-material-difference';
  if (delta > margin) interpretation = 'curiosity-leading';
  else if (delta < -margin) interpretation = 'control-leading';
  return {
    checkpointSteps,
    controlScore,
    curiosityScore,
    delta,
    margin,
    interpretation,
  };
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
