export const VERSION = '0.1.0.1.2';
export const BUILD_MARKER = 'SAVEREC-01012';

export const CONFIG = Object.freeze({
  world: {
    width: 1,
    height: 1,
    maxSteps: 700,
    foodRadius: 0.035,
    hazardRadius: 0.055,
    agentRadius: 0.026,
    wallMargin: 0.018,
    sensorRange: 0.36,
    rayAngles: [-0.72, 0, 0.72],
  },
  physics: {
    accel: 0.0038,
    turnAccel: 0.012,
    drag: 0.965,
    angularDrag: 0.82,
    maxSpeed: 0.016,
    maxAngularSpeed: 0.095,
  },
  energy: {
    initial: 1,
    baseDrain: 0.00125,
    thrustDrain: 0.00115,
    turnDrain: 0.00032,
    foodGain: 0.32,
  },
  rewards: {
    food: 1.0,
    death: -1.0,
    hazard: -0.55,
    wall: -0.035,
    survival: 0.0,
    energyScale: -0.18,
    approachScale: 1.15,
  },
  model: {
    obsSize: 10,
    hiddenSize: 24,
    actionSize: 7,
  },
  ppo: {
    gamma: 0.985,
    lambda: 0.94,
    clip: 0.2,
    entropyCoef: 0.03,
    valueCoef: 0.5,
    learningRate: 0.0003,
    adamBeta1: 0.9,
    adamBeta2: 0.999,
    adamEps: 1e-8,
    maxGradNorm: 0.5,
    epochs: 3,
    minibatchSize: 64,
  },
  curriculum: {
    historySize: 48,
    minSamples: 32,
    transitionCooldownEpisodes: 64,
    promoteThreshold: 0.72,
    demoteThreshold: 0.18,
  },
  validation: {
    seedBase: 'validation:v1',
    episodesPerStage: 8,
    regressionTolerance: 0.35,
    improvementEpsilon: 0.01,
    schedule: [10000, 50000, 100000, 250000, 500000, 750000, 1000000],
    intervalAfterSchedule: 250000,
  },
  runtime: {
    rolloutSteps: 64,
    trainEnvs: 8,
    evalEpisodes: 32,
    compareEpisodes: 24,
    chartPoints: 240,
  },
});

export const ACTIONS = ['COAST', 'THRUST', 'BRAKE', 'LEFT', 'RIGHT', 'THRUST+LEFT', 'THRUST+RIGHT'];
