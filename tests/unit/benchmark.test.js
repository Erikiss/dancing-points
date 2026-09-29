// benchmark.test.js - duo benchmark metrics: syncDistance (offset removal, mirroring),
// syncLag (cross-correlation sign and magnitude), winner, buildBenchmark, CSV/JSON export and
// the leaderboard helpers. Runs are derived from the synthetic choreography (make-choreo.js).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  syncDistance, syncDistanceDetails, syncLag, syncLagDetails, alignRuns, headSpeedSignal,
  winner, compareResults, playerSummary, buildBenchmark, toCSV, toJSON, csvCell,
  leaderboard, leaderboards, mergeResults, rankOf, resultKey, leaderboardToCSV, isResultLike,
  BENCHMARK_FORMAT,
} from '../../webxr/src/duo/benchmark.js';
import { GhostDuo, RUN_FORMAT } from '../../webxr/src/duo/ghost.js';
import { makeChoreo } from './helpers/make-choreo.js';

const choreo = makeChoreo({ id: 'bench', durationBeats: 16, bpm: 120 });   // 8 s, 241 frames

function baseRun(extra = {}) {
  return {
    format: RUN_FORMAT, choreoId: 'bench', fps: 30, frameCount: choreo.frames.head.length,
    head: choreo.frames.head, left: choreo.frames.left, right: choreo.frames.right,
    score: 90, player: 'A', playedAt: '2026-09-29T10:00:00.000Z', ...extra,
  };
}

/**
 * Derive a run from the base run: B(t) = A(t - lag) + offset, optionally mirrored
 * (x -> -x, hands swapped).
 */
function derivedRun({ lag = 0, offset = [0, 0, 0], mirror = false, fps = 30, noise = 0 } = {}) {
  const g = new GhostDuo(baseRun());
  const n = Math.round(g.duration * fps) + 1;
  const head = [], left = [], right = [];
  let seed = 7;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
  for (let i = 0; i < n; i++) {
    const s = g.sampleAt(i / fps - lag);
    const p = (v, k) => v[k] + offset[k] + noise * rnd();
    const hx = mirror ? -s.head.p[0] : s.head.p[0];
    head.push([hx + offset[0] + noise * rnd(), p(s.head.p, 1), p(s.head.p, 2), ...s.head.q]);
    const l = mirror ? s.right.p : s.left.p;
    const r = mirror ? s.left.p : s.right.p;
    left.push([(mirror ? -l[0] : l[0]) + offset[0] + noise * rnd(), p(l, 1), p(l, 2)]);
    right.push([(mirror ? -r[0] : r[0]) + offset[0] + noise * rnd(), p(r, 1), p(r, 2)]);
  }
  return { format: RUN_FORMAT, choreoId: 'bench', fps, frameCount: n, head, left, right, score: 70, player: 'B', playedAt: '2026-09-29T10:05:00.000Z' };
}

function makeResult(over = {}) {
  return {
    choreoId: 'bench', score: 80, stars: 4, maxCombo: 6, timingBias: 0.02, durationSec: 8,
    moves: [{ name: 'Sway', score: 0.9, grade: 'perfekt', startBeat: 0, endBeat: 8 }, { name: 'Arms', score: 0.7, grade: 'gut', startBeat: 8, endBeat: 16 }],
    perBeat: [0.9, 0.8], playedAt: '2026-09-29T10:00:00.000Z', mode: 'solo', player: 'Erik', ...over,
  };
}

test('syncDistance: identical runs 0, pure offset removed, offset kept on request', () => {
  assert.ok(syncDistance(baseRun(), baseRun()) < 1e-6);
  const shifted = derivedRun({ offset: [0.3, -0.05, 0.2] });
  const d = syncDistanceDetails(baseRun(), shifted);
  assert.ok(d.mean < 1e-3, `offset removed: ${d.mean}`);
  assert.equal(d.frames, choreo.frames.head.length);
  assert.ok(Math.abs(d.offset[0] - 0.3) < 1e-3 && Math.abs(d.offset[2] - 0.2) < 1e-3);
  const raw = syncDistance(baseRun(), shifted, { removeOffset: false });
  const expected = Math.hypot(0.3, 0.05, 0.2);
  assert.ok(Math.abs(raw - expected) < 1e-3, `raw ${raw} vs ${expected}`);
  // a different rate is resampled onto A's grid
  const slow = derivedRun({ fps: 20 });
  assert.ok(syncDistance(baseRun(), slow) < 2e-3);
  // lagged run has a visible distance which the lag option removes again (0.2 s = 6 frames:
  // exact; 0.15 s = 4.5 frames: double linear interpolation leaves a small residual)
  const late = derivedRun({ lag: 0.2 });
  const dLate = syncDistance(baseRun(), late);
  assert.ok(dLate > 0.01, `lag creates distance: ${dLate}`);
  assert.ok(syncDistance(baseRun(), late, { lagSec: 0.2 }) < 1e-3);
  const half = derivedRun({ lag: 0.15 });
  assert.ok(syncDistance(baseRun(), half, { lagSec: 0.15 }) < 0.01);
});

test('syncDistance: mirrored partner needs the mirror option', () => {
  const m = derivedRun({ mirror: true });
  const plain = syncDistance(baseRun(), m);
  const mirrored = syncDistance(baseRun(), m, { mirror: true });
  assert.ok(mirrored < 1e-3, `mirrored ${mirrored}`);
  assert.ok(plain > mirrored + 0.01, `unmirrored comparison is worse: ${plain}`);
  const al = alignRuns(baseRun(), baseRun(), { mirror: true });
  assert.equal(al.n, choreo.frames.head.length);
  assert.equal(al.mirror, true);
});

test('syncDistance: no overlap -> NaN / 0 frames', () => {
  const tiny = { choreoId: 'bench', fps: 30, frames: [[0, 1.6, 0, 0, 1, 0, 0, 1, 0]] };
  const d = syncDistanceDetails(baseRun(), tiny);
  assert.equal(d.frames, 1);
  assert.ok(Number.isFinite(d.mean));
  const none = syncDistanceDetails(baseRun(), tiny, { lagSec: 5 });
  assert.equal(none.frames, 0);
  assert.ok(Number.isNaN(none.mean));
});

test('syncLag: sign and magnitude (positive = B behind A), symmetric, zero for identical', () => {
  const late = derivedRun({ lag: 0.1 });
  const l = syncLagDetails(baseRun(), late);
  assert.equal(l.valid, true);
  assert.ok(Math.abs(l.lag - 0.1) < 0.02, `lag ${l.lag}`);
  assert.ok(l.correlation > 0.9, `correlation ${l.correlation}`);
  const back = syncLag(late, baseRun());
  assert.ok(Math.abs(back + 0.1) < 0.02, `reverse lag ${back}`);
  assert.ok(Math.abs(syncLag(baseRun(), baseRun())) < 1e-3, 'identical runs: lag below 1 ms');
  const noisy = derivedRun({ lag: -0.25, noise: 0.01 });
  assert.ok(Math.abs(syncLag(baseRun(), noisy) + 0.25) < 0.04);
  // a lag beyond maxLagSec is clamped to the search window
  const far = derivedRun({ lag: 0.8 });
  const clamped = syncLagDetails(baseRun(), far, { maxLagSec: 0.3 });
  assert.ok(Math.abs(clamped.lag) <= 0.3 + 1e-9);
});

test('syncLag: degenerate signals are reported invalid', () => {
  const still = makeChoreo({ id: 'still', still: true });
  const stillRun = { choreoId: 'still', fps: 30, head: still.frames.head, left: still.frames.left, right: still.frames.right };
  const l = syncLagDetails(stillRun, stillRun);
  assert.equal(l.valid, false);
  assert.equal(l.lag, 0);
  const tiny = { choreoId: 'bench', fps: 30, frames: [[0, 1.6, 0, 0, 1, 0, 0, 1, 0], [0.1, 1.6, 0, 0, 1, 0, 0, 1, 0]] };
  assert.equal(syncLagDetails(baseRun(), tiny).valid, false);
  const sig = headSpeedSignal(baseRun(), 30, 10);
  assert.equal(sig.length, 10);
  assert.ok(sig.every((v) => v >= 0));
});

test('winner and compareResults', () => {
  const a = makeResult({ score: 80 });
  const b = makeResult({ score: 75, player: 'Anna' });
  assert.equal(winner(a, b), 'A');
  assert.equal(winner(b, a), 'B');
  assert.equal(winner(a, makeResult({ score: 80, maxCombo: 9 })), 'B', 'combo breaks the tie');
  assert.equal(winner(a, makeResult({ score: 80, maxCombo: 6, timingBias: 0.1 })), 'A', 'timing breaks the tie');
  assert.equal(winner(a, makeResult({})), 'tie');
  assert.equal(winner(a, null), 'A');
  assert.equal(winner(null, b), 'B');
  assert.equal(winner(null, null), 'tie');
  const sorted = [makeResult({ score: 50 }), makeResult({ score: 90 }), makeResult({ score: 90, stars: 5 }), makeResult({ score: 70 })].sort(compareResults);
  assert.deepEqual(sorted.map((r) => `${r.score}/${r.stars}`), ['90/5', '90/4', '70/4', '50/4']);
});

test('buildBenchmark + exports', () => {
  const a = makeResult({ score: 88, player: 'Erik' });
  const b = makeResult({ score: 74, stars: 3, maxCombo: 4, player: 'Anna', moves: [{ name: 'Sway', score: 0.8, grade: 'gut' }, { name: 'Arms', score: 0.6, grade: 'ok' }] });
  const bm = buildBenchmark(a, b, baseRun(), derivedRun({ lag: 0.1, offset: [0.2, 0, 0] }), { mode: 'online', createdAt: '2026-09-29T11:00:00.000Z' });
  assert.equal(bm.format, BENCHMARK_FORMAT);
  assert.equal(bm.choreoId, 'bench');
  assert.equal(bm.mode, 'online');
  assert.equal(bm.winner, 'A');
  assert.equal(bm.players.length, 2);
  assert.equal(bm.players[0].name, 'Erik');
  assert.equal(bm.players[1].name, 'Anna');
  assert.equal(bm.players[0].perMove.length, 2);
  assert.equal(bm.players[0].perMove[1].grade, 'gut');
  assert.equal(bm.pair.scoreDiff, 14);
  assert.equal(bm.pair.comboDiff, 2);
  assert.ok(Math.abs(bm.pair.syncLag - 0.1) < 0.02);
  assert.ok(bm.pair.syncDistance > 0.01, 'lagged partner is apart');
  assert.ok(bm.pair.syncDistanceAligned < bm.pair.syncDistance, 'aligned distance is smaller');
  assert.ok(bm.pair.syncCorrelation > 0.9);
  assert.equal(bm.pair.framesCompared, choreo.frames.head.length);
  // without runs the pair metrics are null
  const noRuns = buildBenchmark(a, b);
  assert.equal(noRuns.pair.syncDistance, null);
  assert.equal(noRuns.pair.syncLag, null);
  assert.equal(noRuns.pair.scoreDiff, 14);
  // a missing partner result
  const solo = buildBenchmark(a, null, null, null, { nameB: 'Ghost' });
  assert.equal(solo.winner, 'A');
  assert.equal(solo.players[1].score, null);
  assert.equal(solo.players[1].name, 'Ghost');
  // JSON
  const parsed = JSON.parse(toJSON(bm));
  assert.equal(parsed.winner, 'A');
  // CSV
  const csv = toCSV(bm);
  const lines = csv.split('\r\n');
  assert.equal(lines[0], 'section,key,A,B');
  assert.ok(lines.includes('player,name,Erik,Anna'));
  assert.ok(lines.includes('player,score,88,74'));
  assert.ok(lines.includes('move,Sway,0.9,0.8'));
  assert.ok(lines.includes('move,Arms,0.7,0.6'));
  assert.ok(lines.includes('pair,winner,A,'));
  assert.ok(lines.some((l) => l.startsWith('pair,syncLag,')));
  assert.ok(csv.endsWith('\r\n'));
  const semi = toCSV(bm, { separator: ';' });
  assert.ok(semi.startsWith('section;key;A;B'));
  assert.equal(csvCell('a,b'), '"a,b"');
  assert.equal(csvCell('say "hi"'), '"say ""hi"""');
  assert.equal(csvCell(null), '');
  assert.equal(csvCell(1.5), '1.5');
  const ps = playerSummary(a);
  assert.equal(ps.name, 'Erik');
  assert.equal(ps.timingBias, 0.02);
});

test('leaderboard: top N per choreo, ranks, dedupe of local + server copies, filters', () => {
  const results = [];
  for (let i = 0; i < 12; i++) results.push(makeResult({ score: 40 + i * 5, player: `P${i}`, playedAt: `2026-09-${String(i + 1).padStart(2, '0')}T10:00:00.000Z` }));
  results.push(makeResult({ choreoId: 'other', score: 99, player: 'Z' }));
  results.push({ garbage: true });
  const serverCopy = results.slice(0, 3).map((r) => ({ ...r, source: 'server' }));   // duplicates
  const top = leaderboard([...results, ...serverCopy], { choreoId: 'bench' });
  assert.equal(top.length, 10);
  assert.equal(top[0].rank, 1);
  assert.equal(top[0].score, 95);
  assert.equal(top[0].player, 'P11');
  assert.equal(top[9].rank, 10);
  assert.equal(top[9].score, 50);
  assert.ok(top.every((e) => e.choreoId === 'bench'));
  assert.equal(leaderboard(results, { choreoId: 'bench', top: 3 }).length, 3);
  assert.equal(leaderboard(results, { choreoId: 'bench', top: 0 }).length, 12, 'top 0 = all');
  assert.equal(leaderboard(results, { choreoId: 'bench', player: 'P3' }).length, 1);
  assert.equal(leaderboard(results, { choreoId: 'bench', mode: 'duo' }).length, 0);
  const all = leaderboards(results, { top: 5 });
  assert.deepEqual(Object.keys(all), ['bench', 'other']);
  assert.equal(all.bench.length, 5);
  assert.equal(all.other[0].score, 99);
  // merge / keys / rank
  assert.equal(mergeResults(results, serverCopy).length, 13);
  assert.equal(resultKey(results[0]), resultKey(serverCopy[0]));
  assert.equal(rankOf(results, results[11]), 1);
  assert.equal(rankOf(results, results[0]), 12);
  assert.equal(rankOf(results, makeResult({ score: 200 })), 1, 'a new result is ranked as if inserted');
  assert.equal(rankOf(results, null), -1);
  assert.equal(isResultLike({ choreoId: 'x', score: 1 }), true);
  assert.equal(isResultLike({ choreoId: 'x' }), false);
  // ties keep the earlier play first
  const tie = leaderboard([makeResult({ player: 'late', playedAt: '2026-09-02T00:00:00Z' }), makeResult({ player: 'early', playedAt: '2026-09-01T00:00:00Z' })], { choreoId: 'bench' });
  assert.deepEqual(tie.map((e) => e.player), ['early', 'late']);
  const csv = leaderboardToCSV(top);
  assert.ok(csv.startsWith('rank,player,score,stars,maxCombo,timingBias,playedAt,mode,choreoId\r\n1,P11,95,'));
});
