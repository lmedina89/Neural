export const CURRICULUM = [
  { id: 0, name: 'Motor Nursery', foods: 3, hazards: 0, walls: 0, scarcity: 0.25 },
  { id: 1, name: 'Foraging', foods: 2, hazards: 0, walls: 1, scarcity: 0.45 },
  { id: 2, name: 'Obstacle Avoidance', foods: 2, hazards: 2, walls: 2, scarcity: 0.55 },
  { id: 3, name: 'Scarcity', foods: 1, hazards: 3, walls: 3, scarcity: 0.78 },
];

export class CurriculumManager {
  constructor(stage = 0) {
    this.stage = Math.max(0, Math.min(CURRICULUM.length - 1, stage));
    this.history = [];
  }
  current() { return CURRICULUM[this.stage]; }
  noteEpisode({ food, survived }) {
    const score = Math.min(1, food / Math.max(1, this.current().foods)) * 0.8 + (survived ? 0.2 : 0);
    this.history.push(score);
    if (this.history.length > 40) this.history.shift();
    if (this.history.length >= 24) {
      const avg = this.history.reduce((a, b) => a + b, 0) / this.history.length;
      if (avg > 0.72 && this.stage < CURRICULUM.length - 1) {
        this.stage++;
        this.history.length = 0;
        return true;
      }
    }
    return false;
  }
  serialize() { return { stage: this.stage, history: [...this.history] }; }
  restore(x) { this.stage = x.stage ?? 0; this.history = Array.isArray(x.history) ? [...x.history] : []; }
}
