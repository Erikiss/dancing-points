# End-to-end tests / running the app without a headset (emu mode)

The WebXR app under `webxr/` is a static site. Everything the headset does can be exercised in
a normal browser (or headless Chromium) through the **emu mode** URL parameters, which replace
`XRInput` by `EmulatedInput` (`webxr/src/emu/emulated-input.js`).

## 1. Serve `webxr/`

Any static server works (ES modules need `http://`, not `file://`):

```bash
# with the LAN server of this repository (also serves /api for results, runs, own dances)
node server/server.js --http --http-port 8090

# or the plain Python one-liner
python3 -m http.server 8090 --directory webxr
```

Then open `http://localhost:8090/`.

## 2. URL parameters (see `webxr/src/config.js: parseParams` + `src/app.js: parseExtraParams`)

| parameter | effect |
|---|---|
| `emu=1` (or `desktop`) | no WebXR: the head follows the mouse/keyboard (`W/A/S/D` move, `R/F` height, mouse-look with the left button held, `Q/E` raise the left/right hand, `Space`/click = trigger, `Esc` = abort a run). The menu is used with the mouse (hover + click) or the keyboard (arrows, `Enter`, `Esc`, digits `1..9`). |
| `emu=playback` | scripted input: a virtual player performs the choreography itself (mirrored for `mirror: true` dances, i.e. the ideal performance). Deterministic: the clock is virtual and advanced in sub-steps of one scoring tick, so scores do not depend on the frame rate. No audio in this mode. |
| `noise=<m>` / `lag=<s>` / `seed=<n>` | playback only: smooth gaussian noise per axis (metres), a reaction lag (seconds), PRNG seed. |
| `height=<m>` / `yaw=<rad>` | playback only: the virtual player's standing head height and facing (exercises the calibration transform). |
| `choreo=<id or URL>` | choreography to select/start: an id from `choreos/index.json` (or a locally recorded one, or one on the server), **or an absolute/relative URL** of a `dancing-points-choreo/1` JSON file. |
| `autostart=1` | start the run immediately after boot (with `emu=…`). |
| `speed=<f>` | clock speed factor (e.g. `10`). In emu mode the automatic calibration wait (3 s) is divided by the speed as well. |
| `cam=third` | third-person camera (for screenshots); default is the first-person view from the emulated head. |
| `avatar=neural|points|off`, `lang=de|en`, `server=<ws(s)://host:port>`, `telemetry=1` | as in `docs/DESIGN.md` section 4. |

Example, the canonical smoke run (finishes in a few seconds):

```
http://localhost:8090/?emu=playback&choreo=snoop-cwalk&autostart=1&speed=10
```

## 3. What to assert (the smoke test contract, DESIGN.md section 10)

The app exposes `window.__dp = { app, version }`:

* `app.state` runs `MENU → CALIBRATE → COUNTDOWN → PLAYING → RESULTS`; wait until it is `'RESULTS'`.
* `app.lastResult.score >= 90` for exact playback (measured: `snoop-cwalk` 100, `tutorial-basics`
  100, `mocap-freestyle` 99), `<= 70` with `noise=0.35` (measured: 1).
* no `console.error` and no uncaught exception; `warning`s from SwiftShader ("GL Driver Message")
  are harmless.
* `localStorage['dp.results']` (array of results) and `localStorage['dp.runs']` (array of
  `dancing-points-run/1` objects, at most 3 per choreography) each gained one entry.
* useful handles: `app.startSolo(choreoIdOrUrl, {mode})`, `app.abortToMenu()`, `app.playAgain()`,
  `app.clock`, `app.input`, `app.session` (the `PlaySession`), `app.menu.getPanel(id).hits`
  (clickable regions), `app.on('state', fn)`.

## 4. Headless Chromium (Playwright)

`playwright-core` with the pre-installed Chromium works without a GPU:

```js
import { chromium } from 'playwright-core';
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',   // or the default
  args: ['--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(e.message));
await page.goto('http://localhost:8090/?emu=playback&choreo=snoop-cwalk&autostart=1&speed=10');
await page.waitForFunction(() => window.__dp && window.__dp.app.state === 'RESULTS', null, { timeout: 120000 });
const score = await page.evaluate(() => window.__dp.app.lastResult.score);
```

SwiftShader renders the scene at roughly 20 fps at 1280×720; a `speed=10` run of the 32 s
C-Walk takes about 5 s of wall time. `npm run test:e2e` runs `tests/e2e/smoke.test.js`
(`node --test`, playwright-core from the root `devDependencies`) against this contract: it starts
`server/server.js --http --http-port 0 --no-https --data tests/.tmp/...` so the same-origin REST
API and the duo relay are covered as well, and skips with a clear message when no Chromium is
found (`PLAYWRIGHT_CHROMIUM=<chrome>`, `/opt/pw-browsers/chromium-*`, or
`npx playwright-core install chromium`).

## 5. Recording and other flows without a headset

* Record: menu → "Neuen Tanz aufnehmen" → set title/BPM/bars → "Aufnahme starten" → calibrate
  (automatic in emu mode) → count-in → the desktop head/hands are recorded → review → "Speichern"
  stores the dance in `localStorage['dp.choreos']` (and downloads the JSON when no server is
  configured); it then appears under the list as "Eigener Tanz".
* Duo modes appear in the menu once the duo lane registers them (`app.registerMode('ghost', …)`,
  `app.registerMode('online', …)`); without registration the buttons show "Noch nicht verfügbar".
* Programmatic use from the console: `__dp.app.startRecordSetup()`, `__dp.app.beginRecording()`,
  `__dp.app.saveRecording()`, `__dp.app.openSettings()`, `__dp.app.openHighscores()`.
