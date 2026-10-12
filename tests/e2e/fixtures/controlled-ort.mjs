// Test-only replacement for the ORT/session boundary. The app, NeuralAvatarDriver,
// module Worker, DPPipeline, transferable pose buffers, and render loop are unchanged.
// Outputs are the existing Python-generated regression fixtures; no ONNX inference or
// WASM execution is performed here. Real asynchronous delays exercise message ordering.
import {
  MAPPING_INPUT_NAMES, MAPPING_OUTPUT_NAMES, TRACKING_INPUT_NAMES, TRACKING_OUTPUT_NAMES,
} from '/src/net/pipeline.js';

export const env = { wasm: {}, versions: { web: 'controlled-regression-fixture' } };

export class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}

const config = await (await fetch('/__guard__/config')).json();
const fixture = await (await fetch('/__guard__/outputs')).json();
const outputs = {};
for (const kind of ['mapping', 'tracking']) {
  outputs[kind] = Object.fromEntries(Object.entries(fixture[kind]).map(([name, data]) =>
    [name, { data: Float32Array.from(data) }]));
}

let blockNextRun = false;
let serial = 0;
const observation = (event, data = {}) => self.postMessage({ type: '__guard_observation', event, ...data });
self.addEventListener('message', ({ data: msg }) => {
  if (!msg || msg.type !== '__guard_control') return;
  if (msg.delayMs !== undefined) config.delayMs = msg.delayMs;
  if (msg.blockNextRun !== undefined) blockNextRun = msg.blockNextRun;
  observation('configured', {
    id: msg.id, delayMs: config.delayMs,
    isolated: self.crossOriginIsolated, sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    numThreads: env.wasm.numThreads,
  });
});

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export const InferenceSession = {
  async create(bytes) {
    // The HTTP fixture sends one-byte model identifiers in place of the large ONNX files.
    const kind = bytes[0] === 1 ? 'mapping' : bytes[0] === 2 ? 'tracking' : null;
    if (!kind) throw new Error('unexpected test model identifier');
    if (kind === 'mapping') await fetch('/__guard__/init-gate');
    let runs = 0;
    return {
      inputNames: kind === 'mapping' ? MAPPING_INPUT_NAMES : TRACKING_INPUT_NAMES,
      outputNames: kind === 'mapping' ? MAPPING_OUTPUT_NAMES : TRACKING_OUTPUT_NAMES,
      async run() {
        // Keep warm-up separate from guard samples, just as the production worker does.
        const warmup = runs++ === 0;
        if (!warmup && kind === 'tracking') {
          const current = ++serial;
          if (blockNextRun) {
            blockNextRun = false;
            observation('run-blocked', { serial: current });
            await fetch('/__guard__/run-gate?serial=' + current);
            observation('run-released', { serial: current });
          }
          await delay(config.delayMs);
        }
        return outputs[kind];
      },
      async release() {},
    };
  },
};
