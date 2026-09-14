import { CONFIG } from '../config.js';

export function computeGAE(transitions, lastValuesByEnv) {
  const { gamma, lambda } = CONFIG.ppo;
  const adv = new Float64Array(transitions.length);
  const ret = new Float64Array(transitions.length);
  const lastAdv = new Map();
  const nextValue = new Map(lastValuesByEnv);

  for (let i = transitions.length - 1; i >= 0; i--) {
    const t = transitions[i];
    const env = t.env;
    const nv = t.done ? 0 : (nextValue.get(env) ?? 0);
    const na = t.done ? 0 : (lastAdv.get(env) ?? 0);
    const delta = t.reward + gamma * nv - t.value;
    const a = delta + gamma * lambda * na;
    adv[i] = a;
    ret[i] = a + t.value;
    nextValue.set(env, t.value);
    lastAdv.set(env, a);
  }
  return { advantages: adv, returns: ret };
}
