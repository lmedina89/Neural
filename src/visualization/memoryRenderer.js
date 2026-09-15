const TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

function fit(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const c = canvas.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, w: rect.width, h: rect.height };
}

export function projectHiddenState(hidden) {
  if (!hidden?.length) return { x: 0, y: 0, radius: 0 };
  let x = 0, y = 0, e = 0;
  const n = hidden.length;
  for (let i = 0; i < n; i++) {
    const v = Number(hidden[i]) || 0;
    // Fixed deterministic projection. No learned parameters and no RNG state.
    x += v * Math.sin((i + 1) * 1.731 + .41);
    y += v * Math.cos((i + 1) * 2.117 + .93);
    e += v * v;
  }
  const scale = Math.max(1, Math.sqrt(n) * .78);
  return { x: Math.tanh(x / scale), y: Math.tanh(y / scale), radius: Math.sqrt(e / n) };
}

export class MemoryRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.hitPoints = [];
    this.selectedIndex = -1;
  }

  draw(points = [], now = performance.now(), meta = {}) {
    const { c, w, h } = fit(this.canvas);
    c.clearRect(0, 0, w, h);
    this.hitPoints = [];

    const bg = c.createRadialGradient(w * .5, h * .47, 4, w * .5, h * .47, Math.max(w, h) * .65);
    bg.addColorStop(0, 'rgba(18,49,69,.42)');
    bg.addColorStop(.45, 'rgba(8,24,36,.25)');
    bg.addColorStop(1, 'rgba(2,7,12,0)');
    c.fillStyle = bg; c.fillRect(0, 0, w, h);

    // Subtle coordinate field helps the constellation read as a state-space map.
    c.save();
    c.strokeStyle = 'rgba(104,177,204,.055)'; c.lineWidth = .7;
    for (let i = 1; i < 6; i++) {
      const x = i * w / 6, y = i * h / 6;
      c.beginPath(); c.moveTo(x, 18); c.lineTo(x, h - 18); c.stroke();
      c.beginPath(); c.moveTo(18, y); c.lineTo(w - 18, y); c.stroke();
    }
    c.restore();

    if (!points.length) {
      c.fillStyle = 'rgba(167,205,218,.72)'; c.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = 'center';
      c.fillText('Collecting recurrent-state samples…', w / 2, h / 2 - 4);
      c.fillStyle = 'rgba(111,153,168,.62)'; c.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      c.fillText('Learn or Observe to populate the constellation.', w / 2, h / 2 + 16);
      return;
    }

    const padX = Math.max(32, w * .08), padY = Math.max(30, h * .10);
    const map = p => ({ x: padX + (p.x * .5 + .5) * (w - padX * 2), y: padY + (p.y * .5 + .5) * (h - padY * 2) });

    // Recent trajectory: an actual path through projected recurrent state space.
    c.save(); c.lineJoin = 'round'; c.lineCap = 'round';
    for (let i = 1; i < points.length; i++) {
      const a = map(points[i - 1]), b = map(points[i]);
      const age = i / points.length;
      c.strokeStyle = `rgba(94,207,235,${.025 + .13 * age})`;
      c.lineWidth = .45 + .65 * age;
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
    c.restore();

    // Experience ripples are only emitted by real novelty/reward/episode events.
    const recentStart = Math.max(0, points.length - 24);
    c.save(); c.globalCompositeOperation = 'lighter';
    for (let i = recentStart; i < points.length; i++) {
      const p = points[i];
      const strength = clamp(Number(p.eventStrength) || 0);
      if (strength < .08) continue;
      const q = map(p);
      const elapsed = now - Number(p.createdAt || now);
      if (elapsed < 0 || elapsed > 2400) continue;
      const phase = elapsed / 2400;
      const r = 7 + phase * (34 + 38 * strength);
      const alpha = (1 - phase) * (.05 + .28 * strength);
      c.strokeStyle = p.eventType === 'reward' ? `rgba(255,201,105,${alpha})` : p.eventType === 'danger' ? `rgba(255,93,160,${alpha})` : `rgba(132,112,255,${alpha})`;
      c.lineWidth = .7 + 1.4 * strength;
      c.beginPath(); c.arc(q.x, q.y, r, 0, TAU); c.stroke();
    }
    c.restore();

    // Older samples fade naturally; novelty and events brighten real samples.
    c.save(); c.globalCompositeOperation = 'lighter';
    for (let i = 0; i < points.length; i++) {
      const p = points[i], q = map(p), age = (i + 1) / points.length;
      const novelty = clamp(Number(p.novelty) || 0);
      const event = clamp(Number(p.eventStrength) || 0);
      const activity = clamp(Number(p.activity) || 0);
      const rr = 1.15 + 2.15 * age + 2.2 * novelty + 1.1 * event;
      const alpha = .08 + .42 * age + .24 * novelty;
      const hue = Number.isInteger(p.action) ? [190, 202, 220, 276, 318, 35, 155][p.action % 7] : 192;
      c.shadowBlur = (3 + 12 * novelty + 5 * event) * age;
      c.shadowColor = `hsla(${hue},90%,68%,${alpha})`;
      c.fillStyle = `hsla(${hue},90%,70%,${alpha})`;
      c.beginPath(); c.arc(q.x, q.y, rr, 0, TAU); c.fill();
      if (event > .15) {
        c.strokeStyle = `rgba(255,220,150,${.12 + .5 * event})`; c.lineWidth = .6;
        c.beginPath(); c.arc(q.x, q.y, rr + 3 + activity * 4, 0, TAU); c.stroke();
      }
      this.hitPoints.push({ ...q, radius: Math.max(9, rr + 5), index: i, point: p });
    }
    c.restore();

    const latest = points.at(-1), lastPos = map(latest);
    c.save(); c.globalCompositeOperation = 'lighter';
    const pulse = 1 + .16 * Math.sin(now * .008);
    c.strokeStyle = 'rgba(115,231,255,.9)'; c.lineWidth = 1.2; c.shadowBlur = 12; c.shadowColor = 'rgba(88,220,255,.9)';
    c.beginPath(); c.arc(lastPos.x, lastPos.y, (7 + 5 * clamp(latest.activity || 0)) * pulse, 0, TAU); c.stroke(); c.restore();

    c.save();
    c.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    c.fillStyle = 'rgba(116,167,185,.72)'; c.textAlign = 'left';
    c.fillText(`RECURRENT STATE SPACE • ${points.length} samples`, 12, 16);
    c.textAlign = 'right';
    c.fillText(`${meta.mode || 'LIVE'} • ${Number(meta.steps || 0).toLocaleString()} steps`, w - 12, 16);
    c.restore();

    if (this.selectedIndex >= 0 && points[this.selectedIndex]) {
      const p = points[this.selectedIndex], q = map(p);
      c.save(); c.strokeStyle = 'rgba(255,216,126,.92)'; c.lineWidth = 1.2; c.setLineDash([3, 3]);
      c.beginPath(); c.arc(q.x, q.y, 12, 0, TAU); c.stroke(); c.restore();
    }
  }

  inspectAt(clientX, clientY, points = []) {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    let best = null, d2 = Infinity;
    for (const hit of this.hitPoints) {
      const dx = x - hit.x, dy = y - hit.y, dd = dx * dx + dy * dy;
      if (dd < d2 && dd <= hit.radius * hit.radius) { best = hit; d2 = dd; }
    }
    this.selectedIndex = best?.index ?? -1;
    if (!best || !points[best.index]) return null;
    const p = points[best.index];
    return `step ${Number(p.steps || 0).toLocaleString()} • action ${p.actionLabel || '—'} • novelty ${(100 * (p.novelty || 0)).toFixed(1)}% • activity ${(p.activity || 0).toFixed(3)} • ${p.eventType || 'ordinary state'}`;
  }
}
