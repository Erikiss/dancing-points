// Node worker_threads bridge for the *actual* ES-module inference worker. JSON assets are read
// from the repository; tiny model identifiers replace the binary downloads consumed by the
// fixture ORT module. The Web Worker protocol and structured-clone / transfer semantics run
// across a real worker thread, including messages arriving while an inference is suspended.
import { parentPort } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { configure, release } from './worker-fixture-ort.js';

globalThis.self = globalThis;
globalThis.postMessage = (msg, transfer) => parentPort.postMessage(msg, transfer);
globalThis.fetch = async (input) => {
  const url = new URL(input);
  if (url.protocol !== 'file:') throw new Error('fixture fetch expects a local file URL');
  if (url.pathname.endsWith('.onnx')) {
    return new Response(new Uint8Array([url.pathname.includes('mapping_') ? 1 : 2]));
  }
  return new Response(await readFile(url));
};

await import('../../../webxr/src/net/worker.js');

parentPort.on('message', (msg) => {
  if (msg.type === 'test:configure') configure(msg);
  else if (msg.type === 'test:release') release(msg.id, msg.error);
  else if (msg.type !== 'test:barrier') self.onmessage({ data: msg });
  // A barrier after release observes all promise continuations from the previous message.
  if (msg.type === 'test:barrier') parentPort.postMessage(msg);
});
parentPort.postMessage({ type: 'test:host-ready' });
