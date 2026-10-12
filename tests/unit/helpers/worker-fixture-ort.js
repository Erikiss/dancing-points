// Deterministic ORT stand-in for worker-regressions.test.js. It replays the existing Python /
// fp32 model output fixtures; only the asynchronous completion of create/run is controlled.
// No production pipeline or worker code is replaced by this helper.
import { readFileSync } from 'node:fs';
import { MAPPING_INPUT_NAMES, MAPPING_OUTPUT_NAMES, TRACKING_INPUT_NAMES,
  TRACKING_OUTPUT_NAMES } from '../../../webxr/src/net/pipeline.js';

const readFixture = (name) => JSON.parse(readFileSync(new URL('../../fixtures/' + name, import.meta.url), 'utf8'));
const fixtures = { mapping: readFixture('mapping_io.json'), tracking: readFixture('tracking_io.json') };
const pending = new Map();
const counts = new Map();
let gateRuns = false;
let hold = new Set();

export const env = { wasm: {}, versions: { web: 'controlled-fixture' }, logLevel: 'warning' };
export class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

export function configure(options) {
  if (options.gateRuns !== undefined) gateRuns = options.gateRuns;
  if (options.hold) hold = new Set(options.hold);
}

export function release(id, error) {
  const p = pending.get(id);
  if (!p) throw new Error('no pending fixture operation: ' + id);
  pending.delete(id);
  if (error) p.reject(new Error(error)); else p.resolve();
}

async function operation(model, kind, feeds) {
  const key = model + '.' + kind;
  const count = (counts.get(key) || 0) + 1;
  counts.set(key, count);
  const id = key + '.' + count;
  const shouldHold = hold.has(id) || (gateRuns && kind === 'run');
  const waiting = shouldHold ? new Promise((resolve, reject) => pending.set(id, { resolve, reject })) : null;
  const inputs = feeds ? Object.fromEntries(Object.entries(feeds).map(([name, tensor]) => [name, Array.from(tensor.data)])) : undefined;
  self.postMessage({ type: 'test:stage', id, phase: 'start', feeds: inputs });
  try {
    if (waiting) await waiting;
  } finally {
    self.postMessage({ type: 'test:stage', id, phase: 'end' });
  }
}

export class InferenceSession {
  static async create(bytes) {
    const model = bytes[0] === 1 ? 'mapping' : 'tracking';
    await operation(model, 'create');
    const names = model === 'mapping' ? MAPPING_OUTPUT_NAMES : TRACKING_OUTPUT_NAMES;
    const outputs = Object.fromEntries(names.map((name) => [name, {
      data: Float32Array.from(fixtures[model].outputsFp32[name].flat(Infinity)),
    }]));
    let activeRuns = 0;
    let released = false;
    return {
      inputNames: model === 'mapping' ? MAPPING_INPUT_NAMES : TRACKING_INPUT_NAMES,
      outputNames: names,
      async run(feeds) {
        if (released) throw new Error(model + ' session already released');
        activeRuns++;
        try {
          await operation(model, 'run', feeds);
          return outputs;
        } finally { activeRuns--; }
      },
      async release() {
        self.postMessage({ type: 'test:released', model, activeRuns, alreadyReleased: released });
        released = true;
      },
    };
  }
}
