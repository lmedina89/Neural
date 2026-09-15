import { CONFIG } from '../config.js';

export class WorldRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.world = null;
    this.showSensors = true;
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: rect.width, h: rect.height };
  }
  draw(world, mode = 'OBSERVE', noveltyTrail = []) {
    this.world = world;
    const { w, h } = this.resize(), c = this.ctx;
    c.clearRect(0, 0, w, h);
    c.fillStyle = '#071018'; c.fillRect(0, 0, w, h);
    c.strokeStyle = 'rgba(124,180,200,.08)'; c.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      c.beginPath(); c.moveTo((w*i)/10,0); c.lineTo((w*i)/10,h); c.stroke();
      c.beginPath(); c.moveTo(0,(h*i)/10); c.lineTo(w,(h*i)/10); c.stroke();
    }
    const sx = x => x*w, sy = y => y*h;
    // Real curiosity trail from the currently training environment only. These
    // marks are prediction-surprise samples, not decorative random particles.
    if (Array.isArray(noveltyTrail) && noveltyTrail.length) {
      for (let i = 0; i < noveltyTrail.length; i++) {
        const q = noveltyTrail[i];
        const age = noveltyTrail.length > 1 ? i / (noveltyTrail.length - 1) : 1;
        const n = Math.max(0, Math.min(1, Number(q.novelty) || 0));
        if (n <= 0) continue;
        c.beginPath();
        c.arc(sx(q.x), sy(q.y), (4 + 16 * n) * (0.55 + 0.45 * age), 0, Math.PI * 2);
        c.fillStyle = `rgba(246,187,91,${(0.015 + 0.11 * n) * age})`;
        c.fill();
      }
    }
    for (const wall of world.walls) {
      c.fillStyle='rgba(135,155,170,.36)'; c.fillRect(sx(wall.x),sy(wall.y),wall.w*w,wall.h*h);
      c.strokeStyle='rgba(190,220,230,.32)'; c.strokeRect(sx(wall.x),sy(wall.y),wall.w*w,wall.h*h);
    }
    for (const hz of world.hazards) {
      c.beginPath(); c.arc(sx(hz.x),sy(hz.y),hz.r*Math.min(w,h),0,Math.PI*2);
      c.fillStyle='rgba(234,76,120,.24)'; c.fill(); c.strokeStyle='rgba(255,105,145,.85)'; c.stroke();
    }
    for (const f of world.food) {
      c.beginPath(); c.arc(sx(f.x),sy(f.y),CONFIG.world.foodRadius*Math.min(w,h),0,Math.PI*2);
      c.fillStyle='rgba(91,231,196,.82)'; c.fill();
      c.beginPath(); c.arc(sx(f.x),sy(f.y),CONFIG.world.foodRadius*Math.min(w,h)*1.8,0,Math.PI*2);
      c.strokeStyle='rgba(91,231,196,.20)'; c.stroke();
    }
    const a=world.agent, px=sx(a.x), py=sy(a.y), scale=Math.min(w,h);
    if (this.showSensors) {
      c.strokeStyle='rgba(114,205,255,.18)'; c.lineWidth=1;
      for (const ang of CONFIG.world.rayAngles) {
        const d=world.rayDistance(ang);
        c.beginPath(); c.moveTo(px,py); c.lineTo(px+Math.cos(a.angle+ang)*d*w,py+Math.sin(a.angle+ang)*d*h); c.stroke();
      }
    }
    c.save(); c.translate(px,py); c.rotate(a.angle);
    c.beginPath(); c.moveTo(14,0); c.lineTo(-10,-8); c.lineTo(-6,0); c.lineTo(-10,8); c.closePath();
    c.fillStyle='#d9f7ff'; c.fill(); c.strokeStyle='#6dd9ff'; c.stroke(); c.restore();
    c.font='12px system-ui'; c.fillStyle='rgba(220,240,247,.75)';
    c.fillText(`${mode} • seed ${world.seed}`,10,18);
  }
}
