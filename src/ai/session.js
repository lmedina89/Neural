import { CONFIG } from '../config.js';
import { domainSeed, PRNG } from '../utils/prng.js';
import { CURRICULUM, CurriculumManager } from '../sim/curriculum.js';
import { World } from '../sim/world.js';
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
    this.nextMilestoneIndex = 0;
    this.milestoneSteps = [0, 1000, 10000, 50000, 100000, 500000, 1000000];
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
    const started = performanceNow();
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
          if (this.autoCurriculum) this.curriculum.noteEpisode(info);
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
    const metric = {
      steps: this.totalSteps,
      episodes: this.totalEpisodes,
      curriculum: this.curriculum.stage,
      curriculumName: CURRICULUM[this.curriculum.stage].name,
      meanReturn: mean('totalReward'),
      meanFood: mean('food'),
      meanEnergy: mean('energy'),
      hazardHits: mean('hazardHits'),
      wallHits: mean('wallHits'),
      throughput: transitions.length / (elapsed / 1000),
      ...ppo,
    };
    this.metrics.push(metric);
    if (this.metrics.length > CONFIG.runtime.chartPoints) this.metrics.shift();
    this.maybeSaveMilestones();
    return { metric, completed, transitions: transitions.length };
  }

  maybeSaveMilestones() {
    for (const step of this.milestoneSteps) {
      if (step > 0 && this.totalSteps >= step && !this.milestones.has(step)) this.saveMilestone(step);
    }
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

  snapshot() {
    return {
      schema: 1,
      seed: this.seed,
      totalSteps: this.totalSteps,
      totalEpisodes: this.totalEpisodes,
      envSeedCursor: this.envSeedCursor,
      curriculum: this.curriculum.serialize(),
      model: this.model.serialize(),
      optimizer: this.trainer.serialize(),
      metrics: this.metrics,
      milestones: Array.from(this.milestones.entries()),
    };
  }

  restore(data) {
    if (!data || data.schema !== 1) throw new Error('Unsupported checkpoint schema');
    this.seed = data.seed;
    this.totalSteps = data.totalSteps || 0;
    this.totalEpisodes = data.totalEpisodes || 0;
    this.envSeedCursor = data.envSeedCursor || 0;
    this.model.restore(data.model);
    this.trainer.restore(data.optimizer);
    this.curriculum.restore(data.curriculum || {});
    this.metrics = Array.isArray(data.metrics) ? data.metrics.slice(-CONFIG.runtime.chartPoints) : [];
    this.milestones = new Map(Array.isArray(data.milestones) ? data.milestones : []);
    for (let i = 0; i < this.envs.length; i++) this.resetEnv(i);
  }
}

function performanceNow() {
  return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
}
