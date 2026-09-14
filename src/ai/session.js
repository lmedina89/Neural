import { CONFIG } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
import { evaluateCurriculumSuite } from '../evaluation/evaluator.js';
import { RecurrentActorCritic } from './model.js';
import { PPOTrainer } from './ppo.js';
import { computeGAE } from './rollout.js';

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
    this.bestBrains = {};
    this.validationHistory = [];
    this.lastValidationStep = -1;
    this.nextValidationStep = nextValidationAfter(0);
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
            const event = this.curriculum.noteEpisode(info, this.totalEpisodes);
            if (event) curriculumEvent = event;
          }
          this.resetEnv(i);
        }
      }
    }

    const lastValues = new Map();
    for (let i = 0; i < this.envs.length; i++) {
      lastValues.set(i, this.model.forward(this.obs[i], this.hidden[i]).value);
    }
    const { advantages, returns } = computeGAE(transitions, lastValues);
    const ppo = this.trainer.update(transitions, advantages, returns);
    const elapsed = Math.max(1, performanceNow() - started);
    const recent = this.episodeHistory.slice(-40);
    const mean = (key) => recent.length ? recent.reduce((s, x) => s + (x[key] || 0), 0) / recent.length : 0;

    this.maybeSaveMilestones();
    const postValidation = this.maybeAutoValidate(Boolean(curriculumEvent));
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
      ...ppo,
    };
    this.metrics.push(metric);
    if (this.metrics.length > CONFIG.runtime.chartPoints) this.metrics.shift();
    return { metric, completed, transitions: transitions.length, validation, curriculumEvent };
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
    const validation = evaluateCurriculumSuite(this.model, this.curriculum.stage, {
      episodesPerStage: CONFIG.validation.episodesPerStage,
      seedBase: CONFIG.validation.seedBase,
      deterministic: false,
    });
    const priorBest = this.bestBrains[validation.protocol] || null;
    const priorScore = priorBest?.validation?.score ?? null;
    const deltaFromBest = priorScore == null ? null : validation.score - priorScore;
    const improved = priorScore == null || validation.score > priorScore + CONFIG.validation.improvementEpsilon;
    const regression = priorScore != null && deltaFromBest < -CONFIG.validation.regressionTolerance;

    if (improved) {
      this.bestBrains[validation.protocol] = {
        savedAtSteps: this.totalSteps,
        savedAtEpisodes: this.totalEpisodes,
        model: this.model.serialize(),
        optimizer: this.trainer.serialize(),
        curriculum: this.curriculum.serialize(),
        validation,
      };
    }
    this.bestBrain = this.bestBrains[validation.protocol] || priorBest;

    const record = {
      steps: this.totalSteps,
      episodes: this.totalEpisodes,
      curriculum: this.curriculum.stage,
      validation,
      improved,
      regression,
      deltaFromBest,
      bestSteps: this.bestBrain?.savedAtSteps ?? null,
      bestScore: this.bestBrain?.validation?.score ?? null,
    };
    this.validationHistory.push(record);
    if (this.validationHistory.length > 64) this.validationHistory.shift();
    this.lastValidationStep = this.totalSteps;
    this.nextValidationStep = nextValidationAfter(this.totalSteps);
    return record;
  }

  restoreBest() {
    if (!this.bestBrain?.model) throw new Error('No validated best brain is available yet');
    const sourceSteps = this.bestBrain.savedAtSteps;
    this.model.restore(this.bestBrain.model);
    if (this.bestBrain.optimizer) this.trainer.restore(this.bestBrain.optimizer);
    if (this.bestBrain.curriculum) this.curriculum.restore(this.bestBrain.curriculum);
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
    const event = { atSteps: this.totalSteps, sourceSteps };
    this.rollbackHistory.push(event);
    if (this.rollbackHistory.length > 32) this.rollbackHistory.shift();
    return event;
  }

  snapshot() {
    return {
      schema: 2,
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
      bestBrains: this.bestBrains,
      validationHistory: this.validationHistory,
      lastValidationStep: this.lastValidationStep,
      nextValidationStep: this.nextValidationStep,
      rollbackHistory: this.rollbackHistory,
    };
  }

  restore(data) {
    if (!data || (data.schema !== 1 && data.schema !== 2)) throw new Error('Unsupported checkpoint schema');
    this.seed = data.seed;
    this.actionRng = new PRNG(this.seed ^ 0xa5a5a5a5);
    this.totalSteps = data.totalSteps || 0;
    this.totalEpisodes = data.totalEpisodes || 0;
    this.envSeedCursor = data.envSeedCursor || 0;
    this.model.restore(data.model);
    this.trainer.restore(data.optimizer);
    this.curriculum.restore(data.curriculum || {});
    if (data.schema === 1) this.curriculum.cooldownRemaining = CONFIG.curriculum.transitionCooldownEpisodes;
    this.metrics = Array.isArray(data.metrics) ? data.metrics.slice(-CONFIG.runtime.chartPoints) : [];
    this.milestones = new Map(Array.isArray(data.milestones) ? data.milestones : []);
    // Never fabricate missed historical brains when upgrading an old long-running session.
    // Continue from the first milestone strictly after the restored step count.
    this.nextMilestoneStep = Number.isFinite(data.nextMilestoneStep) && data.nextMilestoneStep > this.totalSteps
      ? data.nextMilestoneStep
      : nextHistoricalMilestoneAfter(this.totalSteps);
    // Preserve the exact policy present at a legacy migration point before training changes it.
    if (data.schema === 1 && this.totalSteps > 0 && !this.milestones.has(this.totalSteps)) this.saveMilestone(this.totalSteps);
    this.bestBrains = data.schema >= 2 && data.bestBrains && typeof data.bestBrains === 'object' ? data.bestBrains : {};
    if (data.schema >= 2 && data.bestBrain?.model && data.bestBrain.validation?.protocol && !this.bestBrains[data.bestBrain.validation.protocol]) this.bestBrains[data.bestBrain.validation.protocol] = data.bestBrain;
    const protocol = currentValidationProtocol(this.curriculum.stage);
    this.bestBrain = this.bestBrains[protocol] || (data.schema >= 2 && data.bestBrain?.validation?.protocol === protocol ? data.bestBrain : null);
    this.validationHistory = data.schema >= 2 && Array.isArray(data.validationHistory) ? data.validationHistory.slice(-64) : [];
    this.lastValidationStep = data.schema >= 2 ? Number(data.lastValidationStep ?? -1) : -1;
    this.nextValidationStep = data.schema >= 2 && Number.isFinite(data.nextValidationStep)
      ? data.nextValidationStep
      : this.totalSteps; // v0.1.0 migrations validate immediately after training resumes.
    this.rollbackHistory = data.schema >= 2 && Array.isArray(data.rollbackHistory) ? data.rollbackHistory.slice(-32) : [];
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
  }
}

function currentValidationProtocol(stage) {
  return `${CONFIG.validation.seedBase}|0-${stage}|${CONFIG.validation.episodesPerStage}|seeded-stochastic`;
}

function compactValidation(record) {
  return {
    improved: record.improved,
    regression: record.regression,
    score: record.validation.score,
    bestScore: record.bestScore,
    bestSteps: record.bestSteps,
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
