// gen_choreos.js - procedural choreography generator (DESIGN.md section 5). Builds the shipped
// `dancing-points-choreo/1` files (snoop-cwalk, tutorial-basics) from a small library of
// beat-synchronous motion primitives and writes webxr/choreos/<id>.json plus index.json.
// Fully deterministic: no randomness, no wall-clock timestamps - running it twice yields
// byte-identical files (the unit test checks that). Node ES module: importable without side
// effects (primitives, definitions, buildChoreo, writeChoreos) and runnable as a CLI:
//   node tools/gen_choreos.js [--out <dir>] [--only <id,id>] [--no-index] [--quiet]
// Must not import three.js; only depends on webxr/src/util/math.js (quaternions) and
// webxr/src/game/choreo.js (validateChoreo, as a safety net before writing).
//
// Body model (stage frame S: right-handed, y up, forward -Z, origin on the floor at the
// calibration spot, referenceHeight 1.70 m): the head (tracker) rests at 1.60 m and dips with
// knee bends; both shoulders are 0.20 m beside and 0.22 m below the head; hands are authored
// relative to the head and clamped to the arm reach (0.62 m from the shoulder, < 0.65 m), so
// they are always plausible even when several primitives add up.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { quatFromYaw, quatFromAxisAngle, quatMul, quatNormalize } from '../webxr/src/util/math.js';
import { validateChoreo, CHOREO_FORMAT } from '../webxr/src/game/choreo.js';

export const GENERATOR_VERSION = 1;
export const FPS = 30;
export const REFERENCE_HEIGHT = 1.70;
// Fixed timestamp so that the output is reproducible (meta.createdAt is informational only).
export const GENERATED_AT = '2026-09-29T00:00:00.000Z';

export const BODY = Object.freeze({
  headRestY: 1.60,          // head tracker height of a 1.70 m person standing upright
  shoulderHalfWidth: 0.20,  // lateral offset of each shoulder joint from the head centre
  shoulderDrop: 0.22,       // shoulder joint below the head centre
  armReach: 0.62,           // max distance controller <-> shoulder joint (< 0.65 m)
  maxHandHeadDist: 0.88,    // hard cap (the file validator allows 0.90)
  torsoLever: 0.55,         // hip -> head lever arm used by `lean`
});

// Hand anchors relative to the head (metres). `rest` = arms hanging relaxed, elbows soft.
export const ANCHORS = Object.freeze({
  rest: Object.freeze({ left: [-0.24, -0.72, -0.10], right: [0.24, -0.72, -0.10] }),
  chest: Object.freeze({ left: [-0.15, -0.32, -0.30], right: [0.15, -0.32, -0.30] }),
  hip: Object.freeze({ left: [-0.20, -0.50, -0.25], right: [0.20, -0.50, -0.25] }),
  // crossed in front of the belly, right forearm over the left (hands never pass through each other)
  crossed: Object.freeze({ left: [0.12, -0.50, -0.27], right: [-0.12, -0.36, -0.35] }),
});
const SHOULDER_L = [-BODY.shoulderHalfWidth, -BODY.shoulderDrop, 0];
const SHOULDER_R = [BODY.shoulderHalfWidth, -BODY.shoulderDrop, 0];
// hanging arm vectors (rest hand - shoulder), rotated by the pendulum-style primitives
const ARM_L = [ANCHORS.rest.left[0] - SHOULDER_L[0], ANCHORS.rest.left[1] - SHOULDER_L[1], ANCHORS.rest.left[2] - SHOULDER_L[2]];
const ARM_R = [ANCHORS.rest.right[0] - SHOULDER_R[0], ANCHORS.rest.right[1] - SHOULDER_R[1], ANCHORS.rest.right[2] - SHOULDER_R[2]];

const TWO_PI = Math.PI * 2;

// ---------------------------------------------------------------------------------------------
// Waves and easing (pure functions of the cycle position u; one cycle = one period)

export function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function smoothstep(x) {
  const t = clamp01(x);
  return t * t * (3 - 2 * t);
}

export function fract(x) {
  return x - Math.floor(x);
}

/** Cycle position for beat `b`: `phase` is in beats (the pattern's reference point). */
export function cycles(b, timesPerBeat, phase) {
  return (b - phase) * timesPerBeat;
}

/** Sinusoid, 0 at the cycle start, +1 at a quarter cycle. */
export function sineWave(u) {
  return Math.sin(TWO_PI * u);
}

/**
 * Smooth square wave: +1 from the cycle start, -1 from the half cycle; each level is reached
 * exactly at the cycle boundary / half cycle after a smoothstep transition of `blend` cycles
 * (0 < blend <= 0.5). Models a weight shift that lands on the beat and is then held.
 */
export function stepWave(u, blend = 0.3) {
  const x = fract(u);
  if (x < 0.5 - blend) return 1;
  if (x < 0.5) return 1 - 2 * smoothstep((x - (0.5 - blend)) / blend);
  if (x < 1 - blend) return -1;
  return -1 + 2 * smoothstep((x - (1 - blend)) / blend);
}

/** Raised-cosine pulse (0..1) centred on the cycle start with total `width` (in cycles). */
export function bump(u, width = 0.5) {
  let d = fract(u);
  if (d >= 0.5) d -= 1;
  const h = width * 0.5;
  if (d <= -h || d >= h) return 0;
  return 0.5 * (1 + Math.cos(Math.PI * d / h));
}

function oscillator(shape, blend) {
  if (shape === 'step') return (u) => stepWave(u, blend);
  if (shape === 'sin') return sineWave;
  throw new Error(`unknown wave shape "${shape}"`);
}

// ---------------------------------------------------------------------------------------------
// Pose accumulator: offsets from the rest pose, all primitives ADD into it.

export function createPose() {
  return { head: [0, 0, 0], yaw: 0, pitch: 0, roll: 0, left: [0, 0, 0], right: [0, 0, 0] };
}

export function resetPose(p) {
  p.head[0] = p.head[1] = p.head[2] = 0;
  p.yaw = p.pitch = p.roll = 0;
  p.left[0] = p.left[1] = p.left[2] = 0;
  p.right[0] = p.right[1] = p.right[2] = 0;
  return p;
}

/** out = a + (b - a) * w (component-wise); `out` may alias `a`. */
export function mixPose(out, a, b, w) {
  for (let i = 0; i < 3; i++) {
    out.head[i] = a.head[i] + (b.head[i] - a.head[i]) * w;
    out.left[i] = a.left[i] + (b.left[i] - a.left[i]) * w;
    out.right[i] = a.right[i] + (b.right[i] - a.right[i]) * w;
  }
  out.yaw = a.yaw + (b.yaw - a.yaw) * w;
  out.pitch = a.pitch + (b.pitch - a.pitch) * w;
  out.roll = a.roll + (b.roll - a.roll) * w;
  return out;
}

export function scalePose(p, s) {
  for (let i = 0; i < 3; i++) { p.head[i] *= s; p.left[i] *= s; p.right[i] *= s; }
  p.yaw *= s; p.pitch *= s; p.roll *= s;
  return p;
}

function rotateX(out, v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const y = v[1], z = v[2];
  out[0] = v[0];
  out[1] = y * c - z * s;
  out[2] = y * s + z * c;
  return out;
}

function rotateZ(out, v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const x = v[0], y = v[1];
  out[0] = x * c - y * s;
  out[1] = x * s + y * c;
  out[2] = v[2];
  return out;
}

const tmpV = [0, 0, 0];

/** Adds (R * arm - arm) to `hand`, i.e. the displacement of a hand when its arm rotates. */
function addArmRotation(hand, arm, rotate, angle) {
  rotate(tmpV, arm, angle);
  hand[0] += tmpV[0] - arm[0];
  hand[1] += tmpV[1] - arm[1];
  hand[2] += tmpV[2] - arm[2];
}

// ---------------------------------------------------------------------------------------------
// Motion primitives. Each returns `(beat, pose) => void` that adds its contribution.
// Conventions: amplitudes in metres, angles in radians, `timesPerBeat` = cycles per beat,
// `phase` in beats. Head yaw > 0 turns the face to the player's left (-X); pitch > 0 looks up;
// roll > 0 tilts the head to the left.

/** Knee-bend bounce: the head dips by `amplitude`, lowest on the beat, with a small nod. */
export function bounce(amplitude, timesPerBeat = 1, phase = 0, { nod = 0.8 } = {}) {
  return (b, pose) => {
    const dy = -amplitude * 0.5 * (1 + Math.cos(TWO_PI * cycles(b, timesPerBeat, phase)));
    pose.head[1] += dy;
    pose.pitch += nod * dy;
  };
}

/** Lateral sway / weight shift (+ = right). `look` couples a small yaw towards the side. */
export function sway(amplitude, timesPerBeat = 0.5, phase = 0, { shape = 'sin', blend = 0.3, look = 0.6 } = {}) {
  const osc = oscillator(shape, blend);
  return (b, pose) => {
    const dx = amplitude * osc(cycles(b, timesPerBeat, phase));
    pose.head[0] += dx;
    pose.yaw -= look * dx;
  };
}

/** Fore/aft rock (+ wave = forward, i.e. -Z), with a small forward nod. */
export function rock(amplitude, timesPerBeat = 0.5, phase = 0, { shape = 'step', blend = 0.35, nod = 0.4 } = {}) {
  const osc = oscillator(shape, blend);
  return (b, pose) => {
    const dz = -amplitude * osc(cycles(b, timesPerBeat, phase));
    pose.head[2] += dz;
    pose.pitch += nod * dz;
  };
}

/**
 * Lateral hop: the body alternates between x = -lateral and +lateral, one hop per cycle. The
 * flight (last `flight` of the cycle) ends exactly on the cycle boundary (landing on the beat),
 * with a `height` arc in the air and a small knee `dip` after landing.
 */
export function hop(lateral, height, timesPerBeat = 1, phase = 0, { flight = 0.4, dip = 0.03, dipLen = 0.3, look = 0.25 } = {}) {
  const side = (k) => ((((k % 2) + 2) % 2) === 0 ? 1 : -1);
  return (b, pose) => {
    const u = cycles(b, timesPerBeat, phase);
    const n = Math.floor(u);
    const x = u - n;
    const from = side(n - 1), to = side(n);
    let dx, dy;
    if (x >= 1 - flight) {
      const f = (x - (1 - flight)) / flight;
      dx = lateral * (from + (to - from) * smoothstep(f));
      dy = height * Math.sin(Math.PI * f);
    } else {
      dx = lateral * from;
      dy = x < dipLen ? -dip * Math.sin(Math.PI * x / dipLen) : 0;
    }
    pose.head[0] += dx;
    pose.head[1] += dy;
    pose.yaw -= look * dx;
  };
}

/** Lateral lean from the hips (+ wave = lean right): head roll + displacement on the lever. */
export function lean(angle, timesPerBeat = 0.25, phase = 0, { shape = 'sin', blend = 0.3 } = {}) {
  const osc = oscillator(shape, blend);
  return (b, pose) => {
    const a = angle * osc(cycles(b, timesPerBeat, phase));
    pose.roll -= a;
    pose.head[0] += Math.sin(a) * BODY.torsoLever;
    pose.head[1] -= (1 - Math.cos(a)) * BODY.torsoLever;
  };
}

/** Head turn (+ wave = look to the player's right, +X) with an optional pitch component. */
export function headTurn(angle, timesPerBeat = 0.25, phase = 0, { shape = 'sin', blend = 0.3, pitch = 0 } = {}) {
  const osc = oscillator(shape, blend);
  return (b, pose) => {
    const w = osc(cycles(b, timesPerBeat, phase));
    pose.yaw -= angle * w;
    pose.pitch += pitch * w;
  };
}

/** Pendulum arm swing about the shoulders (+ wave = left arm forward); alternating by default. */
export function armSwing(angle, timesPerBeat = 0.5, phase = 0, { alternate = true, shape = 'sin', blend = 0.3 } = {}) {
  const osc = oscillator(shape, blend);
  return (b, pose) => {
    const aL = angle * osc(cycles(b, timesPerBeat, phase));
    const aR = alternate ? -aL : aL;
    addArmRotation(pose.left, ARM_L, rotateX, aL);
    addArmRotation(pose.right, ARM_R, rotateX, aR);
  };
}

/** Both arms flare outward (abduction) by `angle` at the pulse centre, once per cycle. */
export function armFlare(angle, timesPerBeat = 1, phase = 0, { width = 0.6 } = {}) {
  return (b, pose) => {
    const a = angle * bump(cycles(b, timesPerBeat, phase), width);
    addArmRotation(pose.left, ARM_L, rotateZ, -a);
    addArmRotation(pose.right, ARM_R, rotateZ, a);
  };
}

/**
 * Raise a hand above the head on a front arc (shoulder rotation up to `maxAngle`): with
 * `alternate` the left hand rises and falls during the first half cycle, the right during the
 * second half; otherwise both over the full cycle. `lookUp` adds head pitch while a hand is up.
 */
export function armRaise(amount = 1, timesPerBeat = 0.25, phase = 0, { alternate = true, lookUp = 0.12, maxAngle = 2.8 } = {}) {
  return (b, pose) => {
    const x = fract(cycles(b, timesPerBeat, phase));
    let vL, vR;
    if (alternate) {
      const s = Math.sin(TWO_PI * x);
      vL = x < 0.5 ? s * s : 0;
      vR = x < 0.5 ? 0 : s * s;
    } else {
      const s = Math.sin(Math.PI * x);
      vL = vR = s * s;
    }
    addArmRotation(pose.left, ARM_L, rotateX, maxAngle * amount * vL);
    addArmRotation(pose.right, ARM_R, rotateX, maxAngle * amount * vR);
    pose.pitch += lookUp * (vL + vR);
  };
}

/** Both hands pump down by `depth` (and `forward`) on the pulse centre, once per cycle. */
export function armPump(depth, timesPerBeat = 1, phase = 0, { width = 0.5, forward = 0 } = {}) {
  return (b, pose) => {
    const v = bump(cycles(b, timesPerBeat, phase), width);
    pose.left[1] -= depth * v; pose.right[1] -= depth * v;
    pose.left[2] -= forward * v; pose.right[2] -= forward * v;
  };
}

/** Hands cross in front of the body (`ANCHORS.crossed`) at the pulse centre, once per cycle. */
export function armCross(amount = 1, timesPerBeat = 0.5, phase = 0, { width = 0.7 } = {}) {
  const dL = ANCHORS.crossed.left.map((v, i) => v - ANCHORS.rest.left[i]);
  const dR = ANCHORS.crossed.right.map((v, i) => v - ANCHORS.rest.right[i]);
  return (b, pose) => {
    const v = amount * bump(cycles(b, timesPerBeat, phase), width);
    for (let i = 0; i < 3; i++) { pose.left[i] += dL[i] * v; pose.right[i] += dR[i] * v; }
  };
}

/** Constant hand base: move both hands from `rest` towards an anchor (`ANCHORS.<name>` or {left,right}). */
export function handsAt(anchor, weight = 1) {
  const a = typeof anchor === 'string' ? ANCHORS[anchor] : anchor;
  if (!a) throw new Error(`unknown hand anchor ${anchor}`);
  const dL = a.left.map((v, i) => (v - ANCHORS.rest.left[i]) * weight);
  const dR = a.right.map((v, i) => (v - ANCHORS.rest.right[i]) * weight);
  return (_b, pose) => {
    for (let i = 0; i < 3; i++) { pose.left[i] += dL[i]; pose.right[i] += dR[i]; }
  };
}

// ---------------------------------------------------------------------------------------------
// Shipped choreography definitions

const SNOOP_NOTES = 'Prozedurale v0 der sechs Basic C-Walk Combos aus dem TikTok-Video "Snoop Dogg doing ' +
  'C-Walk" (Electro Breakers). C-Walk ist Fusstechnik; mit Kopf + Haenden sind nur Wippen, ' +
  'Seitwaerts-/Vor-Zurueck-Bewegung, Huepfer und Armschwung erfassbar - genau das kodiert jeder ' +
  'Move mit einer eigenen Signatur. Fuer eine originalgetreue Version den Tanz im Headset ' +
  'aufnehmen (Recorder) und diese Datei ersetzen. / Procedural v0; record in the headset for a ' +
  'faithful version.';

export const CHOREO_DEFS = [
  {
    id: 'snoop-cwalk',
    title: 'Snoop Dogg C-Walk – 6 Basic Combos',
    artist: 'nach Electro Breakers (TikTok)',
    bpm: 90,
    beatsPerBar: 4,
    countInBeats: 4,
    durationBeats: 48,
    mirror: true,
    difficulty: 2,
    audio: { url: null, synth: 'hiphop', offsetSec: 0, gain: 0.8 },
    notes: SNOOP_NOTES,
    moves: [
      {
        name: 'Shoe Vibe',
        beats: 8,
        hint: 'Füße locker abwechselnd nach vorn wischen (Shoe Vibe): bei jedem Beat leicht in die Knie federn, den Kopf sanft ca. 6 cm nach links und rechts pendeln lassen und die Arme locker gegengleich mitschwingen.',
        parts: [
          bounce(0.04, 1, 0),
          sway(0.06, 0.5, -0.5),
          armSwing(0.32, 0.5, -0.5),
        ],
      },
      {
        name: 'Restep',
        beats: 8,
        hint: 'Mit einem Fuß nach vorn steppen und wieder zurücksetzen (Restep): der Oberkörper wippt pro Beat etwa 12 cm vor und zurück, die Arme schwingen jeweils entgegengesetzt zum Körper.',
        parts: [
          rock(0.12, 0.5, 0, { shape: 'step', blend: 0.35 }),
          bounce(0.02, 1, 0),
          armSwing(-0.30, 0.5, 0, { alternate: false, shape: 'step', blend: 0.35 }),
        ],
      },
      {
        name: 'Heel Toe',
        beats: 8,
        hint: 'Ferse und Fußspitze im schnellen Wechsel aufsetzen (Heel Toe): in Achteln kurz federn, bei jedem Beat das Gewicht ca. 8 cm zur anderen Seite verlagern, die Hände bleiben vor der Brust und pumpen leicht mit.',
        parts: [
          bounce(0.04, 2, 0),
          sway(0.08, 0.5, 0, { shape: 'step', blend: 0.3 }),
          handsAt('chest'),
          armPump(0.06, 1, 0, { width: 0.5, forward: 0.02 }),
        ],
      },
      {
        name: 'Side Hopping',
        beats: 8,
        hint: 'Mit beiden Füßen bei jedem Beat ca. 30 cm zur Seite hüpfen, abwechselnd nach rechts und links: der Kopf springt dabei etwa 6 cm hoch und die Arme schwingen bei jedem Hüpfer nach außen.',
        parts: [
          hop(0.30, 0.06, 1, 0, { flight: 0.4, dip: 0.03 }),
          armFlare(0.60, 1, -0.2, { width: 0.6 }),
        ],
      },
      {
        name: 'Shuffle Legs',
        beats: 8,
        hint: 'Die Füße schnell im Wechsel über den Boden schieben (Shuffle Legs): kleine, schnelle Achtel-Wipper mit leichtem Seitenpendeln, die Hände drücken bei jedem Beat kräftig nach unten.',
        parts: [
          bounce(0.03, 2, 0),
          sway(0.06, 0.5, -0.5),
          handsAt('hip'),
          armPump(0.12, 1, 0, { width: 0.5, forward: 0.03 }),
        ],
      },
      {
        name: 'Gangster Two Step',
        beats: 8,
        hint: 'Nach rechts steppen und den anderen Fuß heranziehen, dann nach links (Two Step): der Kopf schwingt langsam ca. 20 cm zur Seite und lehnt sich leicht hinein, auf 2 und 4 kreuzen sich die Arme vor dem Körper.',
        parts: [
          sway(0.20, 0.25, 1, { shape: 'step', blend: 0.25, look: 0 }),
          lean(0.14, 0.25, 0),
          headTurn(0.15, 0.25, 0),
          bounce(0.03, 1, 0),
          armCross(1, 0.5, 1, { width: 0.7 }),
          armFlare(0.25, 0.5, 0, { width: 0.6 }),
        ],
      },
    ],
  },
  {
    id: 'tutorial-basics',
    title: 'Tutorial: Grundschritte',
    artist: 'Dancing Points VR',
    bpm: 80,
    beatsPerBar: 4,
    countInBeats: 4,
    durationBeats: 32,
    mirror: false,
    difficulty: 1,
    audio: { url: null, synth: 'metronome', offsetSec: 0, gain: 0.8 },
    notes: 'Einsteiger-Tutorial: vier Grundbewegungen zu je 8 Beats, die alle weiteren Taenze ' +
      'benutzen. Prozedural erzeugt. / Beginner tutorial, procedurally generated.',
    moves: [
      {
        name: 'Wippen',
        beats: 8,
        hint: 'Bei jedem Beat leicht in die Knie gehen und wieder hochkommen: der Kopf wippt etwa 5 cm auf und ab, die Arme hängen locker an der Seite.',
        parts: [
          bounce(0.05, 1, 0),
          armSwing(0.10, 0.5, -0.5),
        ],
      },
      {
        name: 'Seitwärts',
        beats: 8,
        hint: 'Das Gewicht im Takt von einem Bein aufs andere verlagern: der Kopf pendelt ca. 15 cm nach rechts und links, die Arme schwingen locker mit.',
        parts: [
          sway(0.15, 0.5, -0.5),
          bounce(0.02, 1, 0),
          armSwing(0.18, 0.5, -0.5),
        ],
      },
      {
        name: 'Arme hoch',
        beats: 8,
        hint: 'Ruhig stehen und abwechselnd die linke und die rechte Hand hoch über den Kopf strecken (je zwei Beats hoch und wieder runter), der Blick folgt der Hand leicht nach oben.',
        parts: [
          armRaise(1, 0.25, 0, { lookUp: 0.12 }),
          bounce(0.02, 1, 0),
        ],
      },
      {
        name: 'Freestyle',
        beats: 8,
        hint: 'Alles zusammen: im Takt wippen, das Gewicht seitlich verlagern und dabei abwechselnd die Hände hochstrecken – locker bleiben und im Beat bleiben.',
        parts: [
          bounce(0.04, 1, 0),
          sway(0.10, 0.5, -0.5),
          armRaise(1, 0.25, 0, { lookUp: 0.10 }),
          armSwing(0.15, 0.5, -0.5),
        ],
      },
    ],
  },
];

/** Index entries for choreographies that are shipped but not produced by this generator. */
export const EXTERNAL_INDEX_ENTRIES = [
  {
    id: 'mocap-freestyle',
    title: 'Freestyle (Mocap-Demo)',
    artist: 'Dancing Points Datensatz',
    bpm: 100,
    durationBeats: 50,
    difficulty: 3,
    file: 'mocap-freestyle.json',
  },
];

// ---------------------------------------------------------------------------------------------
// Building frames

/** Frame count for a header: index 0 = beat 0, inclusive end frame. */
export function frameCountFor(def) {
  return Math.round(def.durationBeats * 60 / def.bpm * FPS) + 1;
}

function movesWithBeats(def) {
  let start = 0;
  return def.moves.map((m) => {
    const entry = { ...m, startBeat: start, endBeat: start + m.beats };
    start += m.beats;
    return entry;
  });
}

function evalMove(move, localBeat, pose) {
  resetPose(pose);
  for (const part of move.parts) part(localBeat, pose);
  return pose;
}

/**
 * Pose offsets at `beat` for a definition: the containing move, crossfaded (smoothstep over
 * +-blendBeats around each move boundary) with its neighbour, and eased in/out from the rest
 * pose over `edgeBeats` at the very start and end.
 */
export function samplePose(def, moves, beat, pose, tmp, { blendBeats = 0.5, edgeBeats = 0.5 } = {}) {
  let i = moves.length - 1;
  for (let k = 0; k < moves.length; k++) {
    if (beat < moves[k].endBeat) { i = k; break; }
  }
  const m = moves[i];
  const local = beat - m.startBeat;
  evalMove(m, local, pose);
  if (i + 1 < moves.length && local > m.beats - blendBeats) {
    evalMove(moves[i + 1], local - m.beats, tmp);
    mixPose(pose, pose, tmp, smoothstep((local - (m.beats - blendBeats)) / (2 * blendBeats)));
  } else if (i > 0 && local < blendBeats) {
    const prev = moves[i - 1];
    evalMove(prev, local + prev.beats, tmp);
    mixPose(pose, tmp, pose, smoothstep((local + blendBeats) / (2 * blendBeats)));
  }
  const env = smoothstep(beat / edgeBeats) * smoothstep((def.durationBeats - beat) / edgeBeats);
  if (env < 1) scalePose(pose, env);
  return pose;
}

function round4(v) {
  const r = Math.round(v * 1e4) / 1e4;
  return Object.is(r, -0) ? 0 : r;
}

/** Clamp `hand` (absolute) to the arm reach around `shoulder` and to the head distance cap. */
function constrainHand(hand, shoulder, head) {
  let dx = hand[0] - shoulder[0], dy = hand[1] - shoulder[1], dz = hand[2] - shoulder[2];
  let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (d > BODY.armReach) {
    const s = BODY.armReach / d;
    hand[0] = shoulder[0] + dx * s; hand[1] = shoulder[1] + dy * s; hand[2] = shoulder[2] + dz * s;
  }
  dx = hand[0] - head[0]; dy = hand[1] - head[1]; dz = hand[2] - head[2];
  d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (d > BODY.maxHandHeadDist) {
    const s = BODY.maxHandHeadDist / d;
    hand[0] = head[0] + dx * s; hand[1] = head[1] + dy * s; hand[2] = head[2] + dz * s;
  }
  return hand;
}

const qYaw = [0, 0, 0, 1], qPitch = [0, 0, 0, 1], qRoll = [0, 0, 0, 1], qTmp = [0, 0, 0, 1];
const AXIS_X = [1, 0, 0], AXIS_Z = [0, 0, 1];

/** Head quaternion [x,y,z,w] from yaw (about +Y), pitch (about local +X) and roll (about local +Z). */
export function headQuat(out, yaw, pitch, roll) {
  quatFromYaw(qYaw, yaw);
  quatFromAxisAngle(qPitch, AXIS_X, pitch);
  quatFromAxisAngle(qRoll, AXIS_Z, roll);
  quatMul(qTmp, qYaw, qPitch);
  quatMul(out, qTmp, qRoll);
  return quatNormalize(out, out);
}

/** Build the complete choreography JSON object for a definition. */
export function buildChoreo(def) {
  const moves = movesWithBeats(def);
  const total = moves.length ? moves[moves.length - 1].endBeat : 0;
  if (Math.abs(total - def.durationBeats) > 1e-9) {
    throw new Error(`${def.id}: moves cover ${total} beats, durationBeats is ${def.durationBeats}`);
  }
  const n = frameCountFor(def);
  const head = new Array(n), left = new Array(n), right = new Array(n);
  const pose = createPose(), tmp = createPose();
  const q = [0, 0, 0, 1];
  const hp = [0, 0, 0], lp = [0, 0, 0], rp = [0, 0, 0], sl = [0, 0, 0], sr = [0, 0, 0];
  const beatsPerSec = def.bpm / 60;
  for (let i = 0; i < n; i++) {
    const beat = (i / FPS) * beatsPerSec;
    samplePose(def, moves, beat, pose, tmp);
    hp[0] = pose.head[0]; hp[1] = BODY.headRestY + pose.head[1]; hp[2] = pose.head[2];
    headQuat(q, pose.yaw, pose.pitch, pose.roll);
    for (let k = 0; k < 3; k++) {
      sl[k] = hp[k] + SHOULDER_L[k];
      sr[k] = hp[k] + SHOULDER_R[k];
      lp[k] = hp[k] + ANCHORS.rest.left[k] + pose.left[k];
      rp[k] = hp[k] + ANCHORS.rest.right[k] + pose.right[k];
    }
    constrainHand(lp, sl, hp);
    constrainHand(rp, sr, hp);
    head[i] = [round4(hp[0]), round4(hp[1]), round4(hp[2]), round4(q[0]), round4(q[1]), round4(q[2]), round4(q[3])];
    left[i] = [round4(lp[0]), round4(lp[1]), round4(lp[2])];
    right[i] = [round4(rp[0]), round4(rp[1]), round4(rp[2])];
  }
  const json = {
    format: CHOREO_FORMAT,
    id: def.id,
    title: def.title,
    artist: def.artist,
    bpm: def.bpm,
    beatsPerBar: def.beatsPerBar,
    countInBeats: def.countInBeats,
    durationBeats: def.durationBeats,
    fps: FPS,
    referenceHeight: REFERENCE_HEIGHT,
    mirror: def.mirror,
    difficulty: def.difficulty,
    audio: { ...def.audio },
    moves: moves.map((m) => ({ name: m.name, startBeat: m.startBeat, endBeat: m.endBeat, hint: m.hint })),
    frames: { head, left, right },
    meta: {
      source: 'procedural',
      createdAt: GENERATED_AT,
      author: `tools/gen_choreos.js v${GENERATOR_VERSION}`,
      notes: def.notes || '',
      generatorVersion: GENERATOR_VERSION,
      headRestY: BODY.headRestY,
    },
  };
  const v = validateChoreo(json);
  if (!v.ok) throw new Error(`${def.id}: generated choreography is invalid: ${v.errors.join('; ')}`);
  return json;
}

/** Index entry for a built choreography (DESIGN.md section 5). */
export function indexEntry(json) {
  return {
    id: json.id,
    title: json.title,
    artist: json.artist,
    bpm: json.bpm,
    durationBeats: json.durationBeats,
    difficulty: json.difficulty,
    file: `${json.id}.json`,
  };
}

/**
 * Serialise a choreography: readable header (2-space JSON) with the frame rows on one line
 * each, so diffs stay reviewable and the file small. Trailing newline included.
 */
export function serializeChoreo(json) {
  const lines = ['{'];
  const keys = Object.keys(json);
  keys.forEach((key, idx) => {
    const comma = idx < keys.length - 1 ? ',' : '';
    if (key === 'frames') {
      lines.push('  "frames": {');
      const fkeys = Object.keys(json.frames);
      fkeys.forEach((fk, fi) => {
        const rows = json.frames[fk];
        lines.push(`    "${fk}": [`);
        for (let i = 0; i < rows.length; i++) {
          lines.push(`      [${rows[i].join(',')}]${i < rows.length - 1 ? ',' : ''}`);
        }
        lines.push(`    ]${fi < fkeys.length - 1 ? ',' : ''}`);
      });
      lines.push(`  }${comma}`);
    } else {
      const text = JSON.stringify(json[key], null, 2).split('\n').map((l, li) => (li === 0 ? l : `  ${l}`)).join('\n');
      lines.push(`  "${key}": ${text}${comma}`);
    }
  });
  lines.push('}');
  return `${lines.join('\n')}\n`;
}

export function serializeIndex(entries) {
  return `${JSON.stringify({ choreos: entries }, null, 2)}\n`;
}

/**
 * Generate and write the choreographies (all, or `only` ids) into `outDir`, plus index.json
 * (unless `index === false`). Returns `[{ id, file, frames, bytes }]`.
 */
export function writeChoreos(outDir, { only = null, index = true } = {}) {
  fs.mkdirSync(outDir, { recursive: true });
  const defs = only ? CHOREO_DEFS.filter((d) => only.includes(d.id)) : CHOREO_DEFS;
  const report = [];
  const entries = [];
  for (const def of defs) {
    const json = buildChoreo(def);
    const text = serializeChoreo(json);
    const file = path.join(outDir, `${json.id}.json`);
    fs.writeFileSync(file, text, 'utf8');
    entries.push(indexEntry(json));
    report.push({ id: json.id, file, frames: json.frames.head.length, bytes: Buffer.byteLength(text, 'utf8') });
  }
  if (index && !only) {
    const text = serializeIndex(entries.concat(EXTERNAL_INDEX_ENTRIES));
    const file = path.join(outDir, 'index.json');
    fs.writeFileSync(file, text, 'utf8');
    report.push({ id: 'index', file, frames: 0, bytes: Buffer.byteLength(text, 'utf8') });
  }
  return report;
}

export const DEFAULT_OUT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'webxr', 'choreos');

function parseArgs(argv) {
  const opts = { out: DEFAULT_OUT_DIR, only: null, index: true, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') opts.out = path.resolve(argv[++i]);
    else if (a === '--only') opts.only = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--no-index') opts.index = false;
    else if (a === '--quiet') opts.quiet = true;
    else if (a === '--help' || a === '-h') {
      console.log('usage: node tools/gen_choreos.js [--out <dir>] [--only <id,id>] [--no-index] [--quiet]');
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  return opts;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const report = writeChoreos(opts.out, { only: opts.only, index: opts.index });
  if (!opts.quiet) {
    for (const r of report) {
      console.log(`${r.id.padEnd(18)} ${String(r.frames).padStart(5)} frames  ${String(r.bytes).padStart(7)} bytes  ${r.file}`);
    }
  }
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) main();
