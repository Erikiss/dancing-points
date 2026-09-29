# Choreografien (`webxr/choreos/`)

Hier liegen die Tänze, die das Spiel im Menü anbietet. Jeder Tanz ist eine JSON-Datei im Format
`dancing-points-choreo/1` (vollständige Spezifikation: `docs/DESIGN.md`, Abschnitt 5) und wird in
`index.json` eingetragen.

| Datei | Inhalt |
|---|---|
| `index.json` | Liste der mitgelieferten Tänze (`{"choreos": [{id, title, artist, bpm, durationBeats, difficulty, file}]}`) |
| `snoop-cwalk.json` | Snoop Dogg C-Walk – 6 Basic Combos (prozedural erzeugt, 48 Beats @ 90 bpm, 32 s) |
| `tutorial-basics.json` | Tutorial: Grundschritte (prozedural erzeugt, 32 Beats @ 80 bpm, 24 s) |
| `mocap-freestyle.json` | Freestyle-Ausschnitt aus dem Dancing-Points-Datensatz (Motion Capture, mit Ganzkörperdaten) |

Die beiden prozeduralen Dateien werden von `tools/gen_choreos.js` erzeugt (`npm run gen:choreos`).
Handänderungen darin gehen beim nächsten Generatorlauf verloren – wer sie verändern will, ändert
den Generator oder nimmt den Tanz neu auf.

## Eigenen Tanz hinzufügen

### Weg 1: Im Headset aufnehmen (empfohlen)

1. Im Menü **„Neuen Tanz aufnehmen“** wählen.
2. Titel, BPM (Tap-Tempo-Knopf oder eintippen), Anzahl Takte, Count-in, Beat-Muster
   (`hiphop`, `house`, `metronome`), „Spiegeln“ (für TikTok-Videos, die man wie im Spiegel
   nachtanzt) und optional eine Audio-URL eingeben.
3. Auf den markierten Punkt stellen, kalibrieren, Count-in abwarten und den Tanz vortanzen –
   z. B. während das TikTok-Video auf einem Monitor läuft.
4. In der Vorschau prüfen und **„Speichern“**: mit angeschlossenem Server (`server/`) landet der
   Tanz per `POST /api/choreos` in `data/choreos/<id>.json` und erscheint bei allen Headsets im
   WLAN unter „Eigene Tänze“; ohne Server wird die JSON-Datei heruntergeladen und zusätzlich lokal
   im Browser gespeichert.
5. Die Moves heißen zunächst „Teil 1 … n“ (je 8 Beats). Namen und Hinweise (`hint`) kann man
   danach in der JSON-Datei anpassen.

### Weg 2: JSON schreiben oder generieren

Ein Tanz besteht aus dem Kopf (Titel, BPM, Länge, Moves) und 30-fps-Frames für Kopf und beide
Hände im „Bühnen-Koordinatensystem“ (Meter, y nach oben, Blickrichtung −Z, Ursprung am
Kalibrierpunkt auf dem Boden, Referenzgröße 1,70 m):

```jsonc
{
  "format": "dancing-points-choreo/1",
  "id": "mein-tanz",                  // nur a-z, 0-9 und "-"; Dateiname = id + ".json"
  "title": "Mein Tanz", "artist": "…",
  "bpm": 100, "beatsPerBar": 4, "countInBeats": 4,
  "durationBeats": 32, "fps": 30, "referenceHeight": 1.70,
  "mirror": true, "difficulty": 2,
  "audio": { "url": null, "synth": "hiphop", "offsetSec": 0.0, "gain": 0.8 },
  "moves": [ { "name": "Teil 1", "startBeat": 0, "endBeat": 8, "hint": "…" } ],
  "frames": {
    "head":  [[x, y, z, qx, qy, qz, qw], …],   // durationBeats*60/bpm*fps + 1 Einträge, Index 0 = Beat 0
    "left":  [[x, y, z], …],
    "right": [[x, y, z], …]
  },
  "meta": { "source": "procedural|recorded|mocap", "createdAt": "…", "author": "", "notes": "" }
}
```

Regeln: Frame `i` liegt bei `i / fps` Sekunden nach Beat 0; die Moves sind lückenlos und
aufsteigend und enden bei `durationBeats`; keine `NaN`; Werte auf 4 Nachkommastellen runden.
Der Kopf steht bei einer 1,70 m großen Person in Ruhe auf ca. 1,60 m. Für prozedurale Tänze ist
`tools/gen_choreos.js` die Vorlage: dort werden Bewegungen aus Bausteinen (Wippen, Pendeln,
Hüpfen, Armschwung, Pumpen, Kreuzen, Lehnen, Kopfdrehung) pro Move zusammengesetzt. Ob eine
Datei gültig ist, prüft `node --test tests/unit/choreos-files.test.js` bzw. `validateChoreo`
in `webxr/src/game/choreo.js`.

Optional kann `tools/precompute_teacher.py` aus den drei Punkten mit dem Tracking-Netz
Ganzkörperdaten (`fullBody`) berechnen, damit der Vortänzer als ganze Figur erscheint. Die
mitgelieferten Tänze `snoop-cwalk` und `tutorial-basics` enthalten diesen Block bereits
(`python3 tools/precompute_teacher.py --choreo webxr/choreos/<id>.json --models webxr/models/free
--net-scale 1.1553 --out webxr/choreos/<id>.json`; der Netz-Maßstab 1,1553 = 1,70 m / Kopfhöhe
der Datensatz-Figur). `node tools/gen_choreos.js` behält einen vorhandenen `fullBody` beim
Neuerzeugen bei, solange die Frame-Zahl gleich bleibt; nach einer Änderung der Bewegungen den
Block also neu berechnen. Für eigene Aufnahmen ist er optional (ohne ihn tanzt der Vortänzer als
drei Punkte).

### Wo die Dateien hingehören

* **Mitgeliefert (im App-Ordner):** `webxr/choreos/<id>.json` ablegen **und** in `index.json`
  eintragen (`file` relativ zu diesem Ordner). Danach ggf. den Service-Worker-Cache leeren
  (Seite neu laden).
* **Über den LAN-Server:** `POST /api/choreos` (JSON-Body) oder die Datei direkt nach
  `data/choreos/` legen; der Server mischt sie automatisch in `GET /api/choreos`.
* **Nur lokal im Headset:** Aufnahmen ohne Server bleiben im Browser-Speicher
  (`localStorage['dp.choreos']`) des jeweiligen Headsets.

## Musik und Lizenzen

Das Spiel liefert **keine urheberrechtlich geschützte Musik** mit. Die mitgelieferten Tänze
laufen zu einem synthetischen Beat (`audio.synth`). Wer ein echtes Lied verwenden will, trägt
unter `audio.url` eine Audio-Datei (MP3/OGG/WAV, absolute URL oder relativ zur Choreo-Datei)
ein – z. B. auf dem LAN-Server abgelegt – und passt `bpm`/`offsetSec` an.

Die Rechte dafür liegen beim Betreiber: Für die öffentliche Wiedergabe in einer Spielhalle oder
VR-Arcade sind in Deutschland in der Regel GEMA-Lizenzen (und ggf. GVL) nötig; Sounds aus
TikTok/YouTube sind **nicht** für gewerbliche Nutzung freigegeben. Empfehlung: lizenzfreie bzw.
eigens lizenzierte Musik (z. B. Creative-Commons- oder Stock-Musik mit gewerblicher Lizenz)
verwenden. Auch die Tanzvideos selbst (z. B. der C-Walk von Electro Breakers) sind nur die
Inspiration für die Bewegungen – es werden keine Videos, Bilder oder Tonspuren daraus
ausgeliefert.
