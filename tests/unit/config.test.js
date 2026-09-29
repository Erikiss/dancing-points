// config.test.js - URL parameter parsing, constants and text tables.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseParams, DEFAULT_PARAMS, APP_VERSION, SCORING, STAGE, STORAGE_KEYS, params } from '../../webxr/src/config.js';
import { de, en, getTexts, formatText, GRADE_KEYS, LANGUAGES } from '../../webxr/src/ui/texts.js';
import { EventEmitter } from '../../webxr/src/util/events.js';

test('parseParams defaults and parsing', () => {
  assert.deepEqual(parseParams(''), DEFAULT_PARAMS);
  assert.deepEqual(params, DEFAULT_PARAMS, 'no location in Node');
  const p = parseParams('?emu=playback&noise=0.35&lag=0.1&choreo=snoop-cwalk&autostart=1&speed=10&server=wss://host:8443&avatar=points&style=free&telemetry=1&lang=en&seed=5');
  assert.equal(p.emu, 'playback');
  assert.equal(p.noise, 0.35);
  assert.equal(p.lag, 0.1);
  assert.equal(p.choreo, 'snoop-cwalk');
  assert.equal(p.autostart, true);
  assert.equal(p.speed, 10);
  assert.equal(p.server, 'wss://host:8443');
  assert.equal(p.avatar, 'points');
  assert.equal(p.style, 'free');
  assert.equal(p.telemetry, true);
  assert.equal(p.lang, 'en');
  assert.equal(p.seed, 5);
  assert.equal(parseParams('?emu=1').emu, 'desktop');
  assert.equal(parseParams('?emu=true').emu, 'desktop');
  assert.equal(parseParams('?emu=0').emu, null);
  assert.equal(parseParams('?emu').emu, 'desktop');
  // invalid values fall back
  const bad = parseParams('?noise=-1&speed=0&choreo=../etc&server=ftp://x&avatar=huge&lang=fr&autostart=maybe');
  assert.equal(bad.noise, 0);
  assert.equal(bad.speed, 1);
  assert.equal(bad.choreo, null);
  assert.equal(bad.server, null);
  assert.equal(bad.avatar, 'neural');
  assert.equal(bad.lang, 'de');
  assert.equal(bad.autostart, false);
  assert.equal(APP_VERSION, '0.1.0');
  assert.equal(SCORING.sigmaHead, 0.10);
  assert.equal(SCORING.sigmaHand, 0.18);
  assert.equal(SCORING.sigmaVel, 0.80);
  assert.equal(STAGE.teacherDistance, 2.5);
  assert.equal(STAGE.duoSideOffset, 1.5);
  assert.equal(STORAGE_KEYS.results, 'dp.results');
});

test('texts: German primary, English fallback with identical keys', () => {
  assert.deepEqual(LANGUAGES, ['de', 'en']);
  const deKeys = Object.keys(de).sort();
  const enKeys = Object.keys(en).sort();
  assert.deepEqual(enKeys, deKeys, 'every German key has an English counterpart');
  assert.equal(getTexts('de'), de);
  assert.equal(getTexts('xx'), de);
  const t = getTexts('en');
  assert.equal(t.menuPlay, 'Dance');
  assert.equal(de.menuPlay, 'Tanzen');
  assert.equal(de.gradePerfekt, 'Perfekt');
  assert.equal(de[GRADE_KEYS.daneben], 'Daneben');
  assert.equal(formatText(de, 'nope'), 'nope');
  assert.equal(formatText({ hello: 'Hallo {name}!' }, 'hello', { name: 'Erik' }), 'Hallo Erik!');
  for (const k of deKeys) assert.ok(typeof de[k] === 'string' && de[k].length > 0, k);
});

test('EventEmitter on/off/once/emit', () => {
  const e = new EventEmitter();
  const seen = [];
  const off = e.on('x', (v) => seen.push(v));
  e.once('x', (v) => seen.push('once:' + v));
  assert.equal(e.emit('x', 1), 2);
  assert.equal(e.emit('x', 2), 1);
  assert.deepEqual(seen, [1, 'once:1', 2]);
  off();
  assert.equal(e.emit('x', 3), 0);
  assert.equal(e.listenerCount('x'), 0);
  // a throwing listener does not break the others
  const warn = console.warn;
  let warned = 0;
  console.warn = () => { warned++; };
  e.on('y', () => { throw new Error('boom'); });
  e.on('y', () => seen.push('ok'));
  assert.equal(e.emit('y'), 2);
  console.warn = warn;
  assert.equal(warned, 1);
  assert.equal(seen[seen.length - 1], 'ok');
  e.off('y');
  assert.equal(e.emit('y'), 0);
  assert.throws(() => e.on('z', null), TypeError);
  e.on('z', () => {});
  e.removeAllListeners();
  assert.equal(e.listenerCount('z'), 0);
});
