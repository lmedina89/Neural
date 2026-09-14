import { CONFIG } from '../config.js';

export const CURRICULUM = [
  { id: 0, name: 'Motor Nursery', foods: 3, hazards: 0, walls: 0, scarcity: 0.25 },
  { id: 1, name: 'Foraging', foods: 2, hazards: 0, walls: 1, scarcity: 0.45 },
  { id: 2, name: 'Obstacle Avoidance', foods: 2, hazards: 2, walls: 2, scarcity: 0.55 },
  { id: 3, name: 'Scarcity', foods: 1, hazards: 3, walls: 3, scarcity: 0.78 },
];

export class CurriculumManager {
  constructor(stage = 0) {
    this.stage = clampStage(stage);
    this.history = [];
    this.cooldownRemaining = 0;
    this.transitions = [];
  }

  current() { return CURRICULUM[this.stage]; }

  noteEpisode({ food, survived }, episodeNumber = 0) {
    const cfg = CONFIG.curriculum;
    const score = Math.min(1, food / Math.max(1, this.current().foods)) * 0.8 + (survived ? 0.2 : 0);
    this.history.push(score);
    if (this.history.length > cfg.historySize) this.history.shift();
    if (this.cooldownRemaining > 0) {
      this.cooldownRemaining--;
      return null;
    }
    if (this.history.length < cfg.minSamples) return null;

    const avg = this.history.reduce((a, b) => a + b, 0) / this.history.length;
    let to = this.stage;
    let reason = null;
    if (avg > cfg.promoteThreshold && this.stage < CURRICULUM.length - 1) {
      to = this.stage + 1;
      reason = 'promotion';
    } else if (avg < cfg.demoteThreshold && this.stage > 0) {
      to = this.stage - 1;
      reason = 'demotion';
    }
    if (!reason) return null;

    const from = this.stage;
    this.stage = to;
    this.history.length = 0;
    this.cooldownRemaining = cfg.transitionCooldownEpisodes;
    const event = { from, to, reason, avg, episode: episodeNumber };
    this.transitions.push(event);
    if (this.transitions.length > 64) this.transitions.shift();
    return event;
  }

  serialize() {
    return {
      stage: this.stage,
      history: [...this.history],
      cooldownRemaining: this.cooldownRemaining,
      transitions: this.transitions.map(x => ({ ...x })),
    };
  }

  restore(x = {}) {
    this.stage = clampStage(x.stage ?? 0);
    this.history = Array.isArray(x.history) ? x.history.filter(Number.isFinite).slice(-CONFIG.curriculum.historySize) : [];
    this.cooldownRemaining = Math.max(0, Number(x.cooldownRemaining) || 0);
    this.transitions = Array.isArray(x.transitions) ? x.transitions.slice(-64).map(t => ({ ...t })) : [];
  }
}

function clampStage(stage) {
  return Math.max(0, Math.min(CURRICULUM.length - 1, Number(stage) || 0));
}
