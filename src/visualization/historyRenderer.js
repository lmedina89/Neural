const TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));

function fit(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, globalThis.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const c = canvas.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, w: rect.width, h: rect.height };
}

function lineageId(x) { return x?.learnerLineage?.id || x?.lineage?.id || x?.id || null; }

export function buildHistoryScene(session) {
  const validations = (session.validationHistory || []).map((row, i) => ({
    type: 'validation', steps: Number(row.steps) || 0, score: Number(row.validation?.categoryScores?.balanced ?? row.validation?.balancedScore),
    lineage: lineageId(row), label: `validation ${i + 1}`, interpretation: row.interpretation || null,
    improved: Boolean(row.improved), regression: Boolean(row.regression),
  })).filter(x => Number.isFinite(x.steps));
  const champions = Object.entries(session.bestArchive || {}).map(([category, row]) => ({
    type: 'champion', steps: Number(row?.savedAtSteps) || 0, score: Number(row?.validation?.categoryScores?.[category]),
    lineage: row?.lineage?.id || null, label: `Champion ${category}`,
  })).filter(x => x.steps >= 0);
  const halls = (session.hallOfFame || []).map(row => ({
    type: 'hall', steps: Number(row.savedAtSteps) || 0, score: Number(row.validation?.categoryScores?.[row.category || 'balanced']),
    lineage: row.lineage?.id || null, label: row.id || 'Hall',
  }));
  const branches = (session.frozenLearners || []).map(row => ({
    type: 'branch', steps: Number(row.frozenAtSteps) || 0, score: null, lineage: row.id, parent: row.lineage?.parent?.lineageId || row.lineage?.parent?.sourceLineage || null,
    label: `${row.id}${row.auditRole ? ` • ${row.auditRole}` : ''}`,
  }));
  const lineages = (session.lineageHistory || []).map(row => ({
    type: 'lineage', steps: Number(row.startedAtSteps) || 0, score: null, lineage: row.id, parent: row.parent?.lineageId || row.parent?.sourceLineage || null,
    label: row.id || 'lineage', reason: row.reason || null,
  }));
  const current = { type: 'current', steps: Number(session.totalSteps) || 0, score: Number(session.lastSkillValidation?.categoryScores?.balanced), lineage: session.learnerLineage?.id || null, label: 'CURRENT LEARNER' };
  return { validations, champions, halls, branches, lineages, current };
}

export class HistoryRenderer {
  constructor(canvas) { this.canvas = canvas; this.hitNodes = []; this.selected = null; }

  draw(session, now = performance.now()) {
    const { c, w, h } = fit(this.canvas); c.clearRect(0, 0, w, h); this.hitNodes = [];
    const scene = buildHistoryScene(session);
    const allSteps = [0, scene.current.steps, ...scene.validations.map(x => x.steps), ...scene.champions.map(x => x.steps), ...scene.halls.map(x => x.steps), ...scene.lineages.map(x => x.steps), ...scene.branches.map(x => x.steps)];
    const maxSteps = Math.max(1, ...allSteps), minSteps = 0;
    const pad = Math.max(28, Math.min(64, w * .07));
    const tx = s => pad + clamp((s - minSteps) / Math.max(1, maxSteps - minSteps)) * (w - pad * 2);

    const bg = c.createLinearGradient(0, 0, w, h); bg.addColorStop(0, 'rgba(10,33,47,.26)'); bg.addColorStop(.5, 'rgba(9,18,31,.18)'); bg.addColorStop(1, 'rgba(35,13,44,.16)');
    c.fillStyle = bg; c.fillRect(0, 0, w, h);

    const timelineY = Math.max(96, h * .30), graphTop = Math.max(165, h * .48), graphBottom = h - 42;
    c.strokeStyle = 'rgba(116,184,207,.22)'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(pad, timelineY); c.lineTo(w - pad, timelineY); c.stroke();
    for (let i = 0; i <= 4; i++) {
      const x = pad + i * (w - pad * 2) / 4;
      c.strokeStyle = 'rgba(116,184,207,.07)'; c.beginPath(); c.moveTo(x, 28); c.lineTo(x, h - 28); c.stroke();
      c.fillStyle = 'rgba(112,153,168,.65)'; c.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = 'center';
      c.fillText(`${Math.round(maxSteps * i / 4 / 1000)}k`, x, timelineY + 19);
    }

    // Validation history: score controls vertical position, making recoveries and regressions visible.
    const scoreY = s => timelineY - 16 - clamp(Number.isFinite(s) ? s : .5) * Math.min(66, timelineY - 32);
    if (scene.validations.length > 1) {
      c.strokeStyle = 'rgba(83,205,233,.34)'; c.lineWidth = 1.1;
      c.beginPath();
      scene.validations.forEach((v, i) => { const x = tx(v.steps), y = scoreY(v.score); if (!i) c.moveTo(x, y); else c.lineTo(x, y); }); c.stroke();
    }
    for (const v of scene.validations) this.drawNode(c, tx(v.steps), scoreY(v.score), 4.2, v.regression ? '255,93,160' : v.improved ? '255,206,104' : '94,220,245', v);
    for (const ch of scene.champions) this.drawNode(c, tx(ch.steps), timelineY - 4, 6.3, '255,199,91', ch);
    for (const hall of scene.halls) this.drawNode(c, tx(hall.steps), timelineY + 5, 5.4, '192,126,255', hall);
    const cx = tx(scene.current.steps), pulse = 7.5 + 1.8 * Math.sin(now * .006);
    this.drawNode(c, cx, timelineY - 16, pulse, '106,237,255', scene.current);

    c.fillStyle = 'rgba(147,190,204,.78)'; c.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = 'left';
    c.fillText('LEARNING TIMELINE • score path + Champions + Hall', 12, 16);

    // Lineage map uses persisted lineage IDs and parent references only.
    const ids = [];
    for (const row of [...scene.lineages, ...scene.branches, scene.current]) if (row.lineage && !ids.includes(row.lineage)) ids.push(row.lineage);
    const laneY = new Map(ids.map((id, i) => [id, graphTop + (ids.length <= 1 ? .5 : i / (ids.length - 1)) * Math.max(10, graphBottom - graphTop)]));
    const byId = new Map([...scene.lineages, ...scene.branches, scene.current].filter(x => x.lineage).map(x => [x.lineage, x]));
    c.fillStyle = 'rgba(147,190,204,.72)'; c.fillText('BRAIN LINEAGE', 12, graphTop - 18);
    for (const row of byId.values()) {
      if (!row.parent || !laneY.has(row.parent)) continue;
      const parent = byId.get(row.parent); if (!parent) continue;
      const x1 = tx(parent.steps), y1 = laneY.get(row.parent), x2 = tx(row.steps), y2 = laneY.get(row.lineage);
      c.strokeStyle = 'rgba(138,113,255,.28)'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x1, y1); c.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2); c.stroke();
    }
    for (const row of byId.values()) {
      const x = tx(row.steps), y = laneY.get(row.lineage), rgb = row.type === 'current' ? '89,231,255' : row.type === 'branch' ? '192,126,255' : '125,184,205';
      this.drawNode(c, x, y, row.type === 'current' ? 6 : 4.6, rgb, row);
      c.fillStyle = 'rgba(168,205,218,.72)'; c.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace'; c.textAlign = x > w * .72 ? 'right' : 'left';
      c.fillText(row.lineage, x + (x > w * .72 ? -7 : 7), y + 3);
    }
  }

  drawNode(c, x, y, r, rgb, data) {
    c.save(); c.globalCompositeOperation = 'lighter'; c.shadowBlur = 10; c.shadowColor = `rgba(${rgb},.75)`;
    c.fillStyle = `rgba(${rgb},.80)`; c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill();
    c.restore(); this.hitNodes.push({ x, y, r: Math.max(10, r + 6), data });
  }

  inspectAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect(), x = clientX - rect.left, y = clientY - rect.top;
    let best = null, d2 = Infinity;
    for (const hit of this.hitNodes) { const dx = x - hit.x, dy = y - hit.y, dd = dx * dx + dy * dy; if (dd <= hit.r * hit.r && dd < d2) { best = hit; d2 = dd; } }
    this.selected = best?.data || null;
    const n = this.selected; if (!n) return null;
    const score = Number.isFinite(n.score) ? ` • balanced ${(n.score * 100).toFixed(1)}%` : '';
    return `${n.label || n.type} • step ${Number(n.steps || 0).toLocaleString()}${score} • lineage ${n.lineage || '—'}${n.reason ? ` • ${n.reason}` : ''}`;
  }
}
