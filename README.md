# Dancing Points VR – Tanzspiel für Meta Quest 2

Ein Just‑Dance‑artiges VR‑Tanzspiel für die **Meta Quest 2**, gebaut auf der *Dancing Points*‑Technologie
dieses Repositories: Aus den drei getrackten Punkten der Brille (Kopf + beide Controller/Hände)
werden Tänze bewertet und ein Ganzkörper‑Avatar berechnet. Die App läuft **ohne Installation im
Quest‑Browser** (WebXR), lässt sich als PWA/APK für den Arcade‑Betrieb paketieren und ist für
den WLAN‑Betrieb mit eigenen Songs und eigenen (TikTok‑)Tänzen ausgelegt.

Was drin ist:

- **Menü mit Tänzen**, Kalibrierung, Count‑in, Live‑Scoring (Punkte, Combo, Bewertung pro Move),
  Ergebnisbildschirm mit Sternen und Bestenliste.
- **Snoop Dogg C‑Walk – 6 Basic Combos** (Shoe Vibe, Restep, Heel Toe, Side Hopping, Shuffle Legs,
  Gangster Two Step; 48 Beats bei 90 BPM ≈ 32 s), ein **Tutorial** und eine **Mocap‑Demo**.
- **Duo‑Modus**: gegen einen „Geist“ (gespeicherter Lauf) oder **online gegen eine zweite Brille im
  selben WLAN**, mit Benchmark‑Metriken (Punkte, Synchronität, Verzögerung) und JSON/CSV‑Export.
- **Aufnahme‑Modus**: eigene Tänze direkt im Headset aufnehmen (z. B. ein TikTok‑Video nachtanzen)
  und als Choreografie speichern.
- **Neuronaler Avatar**: die Mapping‑ und Tracking‑Netze aus dem Paper laufen im Browser
  (onnxruntime‑web, int8‑quantisiert) und zeigen einen Ganzkörper‑Spiegel‑Avatar des Spielers.
- **LAN‑Server** (Node.js) für WLAN‑Betrieb: statische App, Upload von Tänzen, Ergebnisse,
  WebSocket‑Relay für den Online‑Duo.

Entwicklerdoku (Architektur, Formate, Pipeline‑Spezifikation): [`docs/DESIGN.md`](docs/DESIGN.md).
Nativer Port (Unity/Meta XR): [`docs/NATIVE_PORT.md`](docs/NATIVE_PORT.md).
Änderungen: [`docs/CHANGELOG.md`](docs/CHANGELOG.md).

## Sofort auf der Quest 2 ausprobieren

WebXR braucht eine **HTTPS‑Adresse** (oder `localhost`). Drei Wege:

**1. GitHub Pages (einfachster Weg, echtes HTTPS, Offline‑Cache funktioniert)**

Einmalig vom Repo‑Besitzer: *Settings → Pages → Source: „GitHub Actions“*. Der Workflow
`.github/workflows/pages.yml` veröffentlicht danach bei jedem Push auf `main` den Ordner `webxr/`.
Die Adresse lautet dann `https://<github-nutzer>.github.io/dancing-points/` – im Quest‑Browser
öffnen, **„VR starten“** drücken. Die Modelle (≈ 85 MB) werden beim ersten Start geladen und vom
Service Worker gecacht; danach startet die App auch ohne Internet.

**2. Eigener PC im WLAN (Arcade‑Betrieb, eigene Tänze, Online‑Duo)**

```bash
npm install            # installiert auch server/ (ws, selfsigned)
node server/server.js  # HTTPS auf :8443, HTTP auf :8080 (nur localhost sinnvoll)
```

Der Server druckt die LAN‑Adresse, z. B. `https://192.168.1.20:8443/`. Im Quest‑Browser öffnen und
die Zertifikatswarnung **einmalig** bestätigen („Erweitert → Weiter zu …“). Hinweis: Mit dem
selbstsignierten Zertifikat registriert der Browser **keinen Service Worker** (kein Offline‑Cache,
keine PWA‑Installation); die App funktioniert trotzdem, Modelle liegen dann im normalen
Browser‑Cache. Für Offline/PWA im LAN ein vertrauenswürdiges Zertifikat verwenden
(`--cert/--key`, Anleitung in [`server/README.md`](server/README.md)).

**3. Am PC testen (ohne Brille)**

`npm run start:http` und `http://localhost:8080/?emu=1` öffnen: Desktop‑Modus mit Maus/Tastatur
(WASD/Maus = Kopf, Tasten heben die Hände). `?emu=playback&choreo=snoop-cwalk&autostart=1` spielt
die Referenz selbst nach (Score ≈ 100); `&noise=0.35` zeigt das Gegenteil.

## Bedienung in VR

1. **Kalibrieren**: auf den Ring am Boden stellen, Richtung Bühne schauen, Trigger 1 s halten
   (Abbruch mit B/Y). Die Kalibrierung merkt sich Kopfhöhe und Blickrichtung; alle Tänze werden auf
   eine Referenzgröße von 1,70 m normiert, damit große und kleine Spieler vergleichbar sind.
2. **Menü**: Laserpointer mit dem Controller, Trigger = auswählen. Punkte: *Tanzen*, *Duo‑Modus*,
   *Neuen Tanz aufnehmen*, *Bestenliste*, *Einstellungen* (Avatar Neural/Punkte/Aus, Lautstärke,
   Spielername, Server, Telemetrie‑Export, Sprache, neu kalibrieren).
3. **Tanzen**: Count‑in (4 Beats mit Klick), dann tanzt die Lehrer‑Figur 2,5 m vor dir; bei
   TikTok‑Tänzen (`mirror: true`) gespiegelt und dir zugewandt. Das HUD zeigt Punkte, Combo, aktuellen
   und nächsten Move. **B/Y 2 s halten** bricht ab.
4. **Ergebnis**: Gesamtpunkte (0–100), Sterne, Bewertung pro Move (Perfekt/Gut/OK/Daneben), Combo,
   Timing‑Abweichung (früh/spät). Ergebnisse landen in der Bestenliste (lokal und, falls vorhanden,
   auf dem Server).

Bewertet werden Kopfposition, Handpositionen relativ zum Kopf und die Bewegungsgeschwindigkeit
gegen die Referenz, mit ±200 ms Toleranz. Wer stillsteht, bekommt maximal 40 Punkte pro Move
(„Energie‑Sperre“). Die Toleranzen stehen in `webxr/src/config.js` (`SCORING`) und sollten nach den
ersten Tests mit echten Spielern nachjustiert werden – die Voreinstellung ist streng.

## Die mitgelieferten Tänze

| ID | Titel | Beats/BPM | Hinweis |
|---|---|---|---|
| `snoop-cwalk` | Snoop Dogg C‑Walk – 6 Basic Combos | 48 @ 90 | **Prozedurale v0**: Kopf‑Bounce/Sway/Hops und Armschwung je Combo sind aus der Beschreibung der Schritte konstruiert, nicht aus dem Video. Die echte Version im Headset aufnehmen (siehe unten) – die Aufnahme ersetzt die Datei. |
| `tutorial-basics` | Tutorial: Grundschritte | 32 @ 80 | Wippen, Seitwärts, Arme hoch, Freestyle |
| `mocap-freestyle` | Freestyle (Mocap‑Demo) | 50 @ 100 | 30‑s‑Ausschnitt aus dem Forschungs‑Datensatz mit echter Ganzkörper‑Referenz. **Nur Demo/Test** – der Datensatz ist Forschungsmaterial der Paper‑Autoren, keine kommerzielle Lizenz. Bei Bedarf Datei und Eintrag in `webxr/choreos/index.json` entfernen. |

C‑Walk ist Fußarbeit. Die Brille sieht nur Kopf und Hände, deshalb bewertet das Spiel das, was die
Fußarbeit im Oberkörper auslöst (Wippen, Verlagerung, Hüpfer, Armschwung). Genau darum ist die
Aufnahme im Headset der richtige Weg zu einer glaubwürdigen Referenz.

## Eigene Tänze und Songs (WLAN‑Betrieb)

**Tanz aufnehmen (empfohlen):** Menü → *Neuen Tanz aufnehmen* → Titel, BPM (Tap‑Tempo), Takte,
Count‑in → kalibrieren → mit dem Metronom tanzen (z. B. das TikTok‑Video am Monitor mitlaufen
lassen) → ansehen → *Speichern*. Mit Server wird die Choreografie hochgeladen
(`POST /api/choreos`) und erscheint auf allen Brillen unter *Eigene Tänze*; ohne Server bleibt sie
im Browser (localStorage) und kann als JSON heruntergeladen werden. Move‑Namen (Standard
„Teil 1…n“) lassen sich später in der JSON ändern.

**Musik:** Die App liefert **keine urheberrechtlich geschützte Musik** mit, sondern synthetische
Beats (`audio.synth`: `hiphop`, `house`, `metronome`). Ein eigener Song wird per `audio.url` in der
Choreografie eingetragen und vom Server ausgeliefert (`server/data/…` oder `webxr/choreos/`).
Die Lizenz (GEMA/Rechteinhaber) ist Sache des Betreibers.

**Format:** Choreografien sind JSON (`dancing-points-choreo/1`): Kopfpose und Handpositionen mit
30 fps im Bühnen‑Koordinatensystem plus Move‑Liste in Beats. Details und ein Generator für
prozedurale Tänze: [`webxr/choreos/README.md`](webxr/choreos/README.md), `tools/gen_choreos.js`.

## Duo‑Modus und Benchmark

- **Duo (Geist)**: eine Brille, der zweite Tänzer ist ein gespeicherter Lauf (eigener Bestwert oder
  ein Lauf vom Server). Beide Punktzahlen live im HUD.
- **Duo (Online)**: zwei Brillen im selben WLAN + `server/`. Host erstellt einen Raum (4‑Buchstaben‑
  Code), Gast tritt bei, beide kalibrieren, der Host startet; die Uhren werden über den Server
  synchronisiert (NTP‑artig, Median aus 5 Messungen). Der Partner wird als Avatar neben der
  Lehrer‑Figur gezeigt. Bei Verbindungsabbruch wird neu verbunden; ohne Partner‑Daten bricht die
  Runde nach einem Timeout sauber ab.
- **Benchmark**: pro Spieler Punkte/Sterne/Moves; als Paar `syncDistance` (mittlerer 3‑Punkt‑Abstand
  nach Offset‑Bereinigung), `syncLag` (Kreuzkorrelation der Kopfgeschwindigkeit, Sekunden) und Sieger.
  Export als JSON/CSV im Ergebnisbildschirm; Speicherung lokal und unter `POST /api/results`.

## Der neuronale Avatar (Dancing‑Points‑Netze im Browser)

Pro Tick (30 Hz) sagt `mapping_leader` aus den letzten 0,5 s der drei Punkte die kommende Sekunde
voraus, und `tracking_leader` erzeugt daraus autoregressiv die nächste Ganzkörper‑Pose
(34 Gelenke). Die Laufzeit‑Pipeline (Root aus der Kopfpose, relative Root‑Motion, kanal‑major
Tensoren, Nachbearbeitung) ist in `webxr/src/net/pipeline.js` portiert und gegen eine
numpy‑Referenz (`tools/dp_pipeline.py`) und den Original‑Datensatz verifiziert:

| Prüfung (Stil „free“, Datensatz‑Clip) | Ergebnis |
|---|---|
| Laufzeit‑Preprocessing vs. Trainings‑Preprocessing | Abweichung < 2e‑6 |
| Mapping: Vorhersage der nächsten 30 Frames vs. Ground Truth | 12,6 cm (statische Baseline 35 cm) |
| Tracking: nächste Pose bei bekannter Zukunft | 3,5 cm mittlerer Gelenkfehler |
| Geschlossener Regelkreis 10 s (mit Root‑Korrektur) | 9,4 cm; Drift 1,3 cm |
| int8 vs. fp32 | < 0,5 cm Unterschied, 4× kleiner, 4× schneller |

Modelle: `webxr/models/free/` (int8, 35 MB + 51 MB, plus `meta.json`, `skeleton.json`,
`init_pose.json`). Im Browser läuft die Inferenz in einem Web Worker; ein **Performance‑Guard**
misst die Inferenzzeit und schaltet bei > 25 ms auf 15 Hz und bei > 60 ms auf den leichten
Punkte‑Avatar um (HUD‑Hinweis). Auf einem Desktop‑Kern braucht ein Tick ≈ 30 ms, mit mehreren
WASM‑Threads ≈ 16 ms; Threads gibt es nur bei *Cross‑Origin‑Isolation* (der LAN‑Server setzt die
nötigen COOP/COEP‑Header, GitHub Pages nicht). **Erwartung für die Quest 2:** ohne Threads greift
der Guard vermutlich und zeigt den Punkte‑Avatar; mit dem LAN‑Server ist der Ganzkörper‑Avatar bei
15 Hz realistisch. Das ist mit der echten Brille zu prüfen (`?avatar=neural` erzwingt den Versuch).

**Andere Stile / eigene Netze:** Checkpoints (`checkpoints.tar`, 10 GB) vom Google‑Drive‑Link im
Original‑README laden, dann

```bash
pip install -r tools/requirements.txt
python tools/prepare_models.py --checkpoints checkpoints.tar --style chacha --set both --out webxr/models
```

erzeugt `webxr/models/chacha/…`; `?style=chacha` wählt den Stil. Das Netz ist eine Drop‑in‑ONNX‑
Datei mit `meta.json`. Für Analysen (Interpretability, Fine‑Tuning‑Daten) gibt es in den
Einstellungen **„Telemetrie exportieren“**: alle Netz‑Ein‑ und Ausgaben der letzten 2000 Ticks als
JSON. Referenzimplementierung, Fixtures und Parity‑Tests: `tools/dp_pipeline.py`,
`tools/gen_fixtures.py`, `tests/unit/pipeline*.test.js`, `tests/net/`.

## Arcade‑Betrieb

- **Installation als App**: Von GitHub Pages (oder einem Server mit vertrauenswürdigem Zertifikat)
  lässt sich die PWA im Quest‑Browser installieren („Zum Startbildschirm“). Für eine echte APK
  Meta’s `ovr-platform-util create-pwa` mit `webxr/manifest.webmanifest` verwenden; die Adresse
  bleibt die Web‑URL, Updates kommen automatisch (Service Worker ist network‑first für die App‑
  Shell und cacht Modelle/Bibliotheken dauerhaft).
- **Offline**: Nach dem ersten Start sind App, Modelle und mitgelieferte Tänze gecacht (nur mit
  vertrauenswürdigem HTTPS). Server‑Funktionen (Upload, Bestenliste, Online‑Duo) brauchen das WLAN.
- **Server absichern**: `node server/server.js --token <geheim>` schützt schreibende API‑Aufrufe;
  die Brillen bekommen den Token einmalig über `?token=…` in der URL. Zertifikat, Firewall und
  Optionen: [`server/README.md`](server/README.md).
- **Kiosk‑Tipps**: Kalibrierring auf dem Boden markieren (Spieler stehen automatisch richtig);
  Spielername in den Einstellungen setzen; Bestenliste pro Tanz; Ergebnisse liegen in
  `server/data/results.json` (CSV‑Export im Ergebnisbildschirm).
- **Leistung Quest 2**: Szene ist bewusst leicht (kein Schatten, Pixel‑Ratio 1, ein Draw‑Call pro
  Avatar, HUD‑Texturen nur bei Änderungen), Ziel 72 Hz. Die Netze laufen im Worker, siehe oben.
- **Bekannte Grenzen**: kein Fuß‑Tracking (Fußarbeit wird aus Kopf/Händen abgeleitet); die C‑Walk‑
  Referenz ist bis zur Aufnahme im Headset prozedural; Handtracking wird genutzt, wenn aktiv
  (Handgelenk), aber die Menü‑Bedienung ist auf Controller ausgelegt; Song‑Synchronisation hängt
  vom korrekten BPM/Offset in der Choreografie ab; Texteingabe in VR ist auf Presets beschränkt.

## Entwicklung

```bash
npm install                 # Root + server/
npm test                    # Unit-Tests (node --test): Scoring, Choreos, Pipeline-Parität, Server, Duo
npm run test:e2e            # Playwright, headless Chromium: Boot, Scoring, Neural-Worker, Aufnahme, Duo
npm run gen:choreos         # prozedurale Choreografien neu erzeugen
node tests/net/worker-smoke.mjs   # Inferenz-Worker im Browser messen (optional)
```

CI (`.github/workflows/ci.yml`) führt Unit‑ und E2E‑Tests aus; `pages.yml` deployt `webxr/`.
Alles ist ohne Bundler in ES‑Modulen geschrieben (three.js r170 und onnxruntime‑web 1.20.1 sind
unter `webxr/vendor/` eingebunden), sodass die App auch von einem USB‑Stick‑Server läuft – nur nicht
direkt von `file://`.

## Roadmap

1. C‑Walk im Headset aufnehmen und als Referenz einchecken; Toleranzen mit echten Spielern kalibrieren.
2. Quest‑2‑Messung des Neural‑Avatars (Threads via LAN‑Server) und ggf. kleineres Tracking‑Netz.
3. Nativer Port (Unity 2022.3 + Meta XR + Sentis) auf Basis des vorhandenen `dancing-points-unity`
   Frameworks, siehe [`docs/NATIVE_PORT.md`](docs/NATIVE_PORT.md).
4. Partner‑KI‑Modus (Follower‑Netze) für Ballroom‑Stile; Song‑Beat‑Erkennung; Handtracking‑Menü.

## Lizenzen

three.js (MIT), onnxruntime‑web (MIT), ws/selfsigned (MIT). Die vortrainierten Gewichte und der
Datensatz stammen von den Autoren des Papers *Dancing Points* (Li, Starke, Ye, Sorkine‑Hornung,
CGF 2026) und sind für Forschungszwecke veröffentlicht – **eine kommerzielle Nutzung der Gewichte
und Daten muss mit den Autoren geklärt werden; dieses Repository erteilt diese Rechte nicht.**
Musik ist nicht enthalten.

---

## Original research README (Dancing Points, Python training code)

# Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs

![Python](https://img.shields.io/badge/Python->=3.11-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=2.1-Red?logo=pytorch)

This repository provides the implementation for our leader/follower ballroom-dancing mapping and
tracking networks. Given only the three-point (head + both wrists) trajectory of a VR headset and
controllers, our method predicts both the counterpart dancer's three-point trajectory and each
dancer's full-body motion, with a purely deterministic, non-generative model. It is based on our
work [Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs](https://peizhuoli.github.io/dancing-points/).

For the Unity project for visualization and real-time playback, a separate repository is provided
[here](https://github.com/PeizhuoLi/dancing-points-unity).

```bibtex
@inproceedings{Li2026dancingpoints,
  title={Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs},
  author={Li, Peizhuo and Starke, Sebastian and Ye, Yuting and Sorkine-Hornung, Olga},
  booktitle = {Computer Graphics Forum},
  doi = {https://doi.org/10.1111/cgf.70588},
  year = {2026}
}
```

## Prerequisites

This code has been tested under Ubuntu 20.04 with Python 3.11. Please install the following
packages (and their dependencies):

- pytorch == 2.1.1
- onnx == 1.13.1
- numpy == 1.26.0
- matplotlib == 3.8.0
- tqdm == 4.65.0
- tensorboard == 2.12.1

## Quick Start

We provide pre-trained checkpoints for six ballroom-dancing styles (balboa, cha-cha, foxtrot, free,
hustle, viennese waltz), one `mapping_leader`/`mapping_follower`/`tracking_leader`/`tracking_follower`
set per style. Download `checkpoints.tar` [here](https://drive.google.com/file/d/1l8pY825MRyPjIqVu025L07C1I_2uLlQP/view?usp=share_link)
and extract it at the root of the repository:

```bash
tar -xf checkpoints.tar -C results/release
```

so that e.g. `results/release/dancing/chacha/tracking_follower/args.txt` exists.

Download the corresponding pre-processed dataset, `datasets.tar`
[here](https://drive.google.com/file/d/1Kx1uK9pQ3h3veSlN_EolNIBHDxuUewpM/view?usp=share_link), and
extract it under `./Datasets`:

```bash
tar -xf datasets.tar -C Datasets
```

which produces `Datasets/Dance-All-2-3pt` (leader) and `Datasets/Dance-All-1` (follower) — the
combined, multi-style datasets used by every style above (see `--data_name_filter` below).
Checkpoints and datasets for [LaFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
(a single `mapping`/`tracking` pair trained on the whole dataset) are not yet bundled in these
archives.

To evaluate a checkpoint and re-export its ONNX model, run:

```bash
python test_unified_autoregressive_mapping.py --save=./results/release/dancing/chacha/tracking_follower --export_onnx=1
```

`--save` also works for any `mapping_*` checkpoint. The script reads `args.txt` and the newest
`.pt` file from `--save`, rebuilds the matching dataset/model, and reports reconstruction losses.
Prebuilt `model.onnx` files for real-time playback in Unity are included with each checkpoint, and
are also shipped directly with the [Unity project](https://github.com/PeizhuoLi/dancing-points-unity).

## Training from Scratch

Each of the four networks per style/character pair — `mapping_leader`, `mapping_follower`,
`tracking_leader`, `tracking_follower` — is trained as a separate run via
`train_unified_autoregressive_mapping.py`. Every run takes `--paths=<dataset_a>,<dataset_b>`, two
dataset folder names under `./Datasets` separated by a comma; *which* dataset goes first/second
depends on the network:

| Network            | `--model_type` | `--paths`             | Predicts                                                                    |
|---------------------|-----------------|------------------------|------------------------------------------------------------------------------|
| `mapping_leader`   | `mlp_pose`      | `<leader>,<leader>`   | the leader's own future 3-point trajectory, from the leader's 3-point history |
| `mapping_follower` | `mlp_pose`      | `<leader>,<follower>` | the follower's future 3-point trajectory, from the leader's 3-point history   |
| `tracking_leader`  | `cvae`          | `<leader>,<leader>`   | the leader's full-body motion, from the leader's own future 3-point + current pose |
| `tracking_follower`| `cvae`          | `<follower>,<follower>` | the follower's full-body motion, from the follower's own future 3-point + current pose |

In other words: tracking is always self-tracking (both `--paths` entries are the same character's
data), so `<leader>`/`<follower>` above just tells you which character's dataset to point it at.
Mapping always takes the leader's data as its *first* `--paths` entry (the input); the second entry
is the *target* — the leader's own dataset again for `mapping_leader`, or the follower's dataset for
`mapping_follower`. See `option.py`, `models/CVAE.py` and `models/MLPPoseMapping.py` for the full
set of training/architecture flags — in particular `--data_name_filter` to restrict training to a
single dance style within a combined multi-style dataset.

The rest of each network's recipe (epochs, network size, learning rate, the 3-point input joints,
loss weights, ...) is baked in as the argparse defaults for `--model_type=cvae`
(`AutoregressiveCVAEOption` in `models/CVAE.py`) and `--model_type=mlp_pose`
(`OneFrameMappingOption` in `models/MLPPoseMapping.py`), so a training command only needs to state
what actually varies per run: `--paths`/`--data_name_filter` (which dataset(s)) and `--save` (where
to write it). Every flag is still overridable on the command line if you want to deviate from the
shipped recipe.

### Full example: training all four networks for balboa

All four networks use the same combined, multi-style `Dance-All-2-3pt` (leader) /
`Dance-All-1` (follower) datasets, restricted to one style via `--data_name_filter`. This is a
single unified recipe — swap `balboa` for any other style name (chacha, foxtrot, free, hustle,
vwaltz) in `--data_name_filter` and `--save` to reproduce that style instead, everything else
unchanged.

```bash
# tracking_leader: leader's full-body motion, from the leader's own future 3pt + current pose
python train_unified_autoregressive_mapping.py --model_type=cvae \
    --paths=Dance-All-2-3pt,Dance-All-2-3pt \
    --data_name_filter=balboa --save=./results/balboa/tracking_leader

# tracking_follower: same recipe, follower's own dataset instead of the leader's
python train_unified_autoregressive_mapping.py --model_type=cvae \
    --paths=Dance-All-1,Dance-All-1 \
    --data_name_filter=balboa --save=./results/balboa/tracking_follower

# mapping_leader: leader's own future 3pt trajectory, from the leader's 3pt history
python train_unified_autoregressive_mapping.py --model_type=mlp_pose \
    --paths=Dance-All-2-3pt,Dance-All-2-3pt \
    --data_name_filter=balboa --save=./results/balboa/mapping_leader

# mapping_follower: same recipe, follower's dataset as the second --paths entry (the target)
python train_unified_autoregressive_mapping.py --model_type=mlp_pose \
    --paths=Dance-All-2-3pt,Dance-All-1 \
    --data_name_filter=balboa --save=./results/balboa/mapping_follower
```

> **Note:** this is the recommended recipe for reproducing/retraining balboa going forward. The
> checkpoint actually shipped under `results/release/dancing/balboa/mapping_*` was trained with an
> older, balboa-specific recipe instead (standalone `Balboa-2-V3-3pt`/`Balboa-1-V3` datasets and a
> different root-position loss weight) — see the caveat under Released Checkpoints below.

Checkpoints are written to each `--save` dir every `--save_freq` epochs. After training, run
`test_unified_autoregressive_mapping.py --save=<save_dir> --export_onnx=1` on each of the four
dirs, same as Quick Start, to evaluate and export its ONNX model. At inference time, chaining
`mapping_follower`'s output into `tracking_follower`'s input (and `mapping_leader`'s into
`tracking_leader`'s) is what lets a single VR three-point stream drive both avatars.

## Released Checkpoints

```
results/release/
  dancing/<style>/{tracking_leader, tracking_follower, mapping_leader, mapping_follower}/
  lafan/{tracking, mapping}/
```

`<style>` is one of `balboa`, `chacha`, `foxtrot`, `free`, `hustle`, `vwaltz`. Each checkpoint
directory holds the training `args.txt`, the final `.pt` weights, and a `model.onnx` export.

A couple of caveats about this specific set of checkpoints:
- Balboa's `mapping_leader`/`mapping_follower` pair was trained slightly differently from the other
  five styles (an earlier sweep, different root-position loss weight); its ONNX input/output
  signature is identical to the others, so it is still a drop-in replacement, just not from the
  exact same training recipe.
- The LaFAN checkpoints were trained with a longer, cyclic-LR schedule than they ended up running
  for, so the learning-rate restart never fully completed. They are shipped as-is and still perform
  well, but a from-scratch retrain with the schedule matched to the actual epoch count would be
  expected to do slightly better.

## Original Motion Data

The preprocessed datasets above are derived from raw motion-captured skeletal animation. The
original data, in FBX format, is available [here](https://drive.google.com/file/d/1FqiUV1l014aDtSXyd-tE3aIEs70Y5rlg/view?usp=share_link).

## Acknowledgments

The optimizer and cyclic learning-rate scheduler in `Library/AdamWR/` are adapted from an
implementation of ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) and
["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186).
