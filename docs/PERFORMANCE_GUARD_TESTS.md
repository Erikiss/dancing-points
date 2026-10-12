# Performance-Guard: Regressionen und Validierung

Stand: 12. Oktober 2026. Ausgangspunkt ist der unveränderte Merge von
[PR #3](https://github.com/Erikiss/dancing-points/pull/3), Commit
`59d96c6772b66335582662006d80e357f356c0ce`.

## Was diese Prüfung nachweist

Die neuronale Pipeline lässt sich mit kontrollierten Verzögerungen durch ihre Zustände
führen: volle Inferenzrate, reduzierte Rate und Punkte-Avatar. Rendering und die vorhandene
Playback-Emulation laufen auch während einer angehaltenen Modellinitialisierung oder einer
ausbleibenden Worker-Antwort weiter. Die Regressionen verwenden die produktive App, den
Driver und den Worker; nur die ORT-Sessions werden in den kontrollierten Tests ersetzt.

Das ist kein Quest-2-Benchmark. Insbesondere erzeugt ein asynchroner 35-ms-Test-Delay
keine mit ONNX-Matrixmultiplikationen vergleichbare CPU-Last. Die Threadauswahl und echte
Browser-Isolation werden geprüft, die Leistung der WASM-Threads wird damit nicht simuliert.

## Bereits vorhandene Abdeckung

| Vorhandener Test | Abdeckung vor dieser Änderung | Fehlender Aspekt |
|---|---|---|
| `tests/unit/pipeline.test.js` | 72-Hz-Samples auf 30-Hz-Eingangsraster, Koordinatenumrechnung, Pose-Interpolation, Buffer-Recycling | Verzögerte Antworten über Stop/Reset hinweg |
| Inferenz-Guard im selben Test | Gemeldete 40 → 5 → 90 ms: Drosseln, Erholen, Deaktivieren | Der Fake-Worker antwortet ohne entsprechende Verzögerung und ignoriert `inferEvery`; außerdem verkürztes Testfenster |
| `tests/unit/fixes.test.js` | Threadbudget ohne Isolation, Quest-/Desktop-Caps; Render-Guard mit verkürzten Fenstern | Gemeinsames Verhalten von Render- und Inferenz-Guard, alte Teilfenster nach Neustart |
| `tests/e2e/smoke.test.js` | Acht Emulationsszenarien einschließlich echter ONNX-Pose oder begründetem Fallback, Scoring, Aufnahme, Ghost und Online-Duo | Kontrollierte Verzögerungen und reale Isolation an/aus als Matrix |
| Pipeline-/Mathe-Fixtures | Numerische Parität zur vorhandenen Python-Referenz | Asynchroner Worker-Lifecycle |
| `tests/net/worker-smoke.mjs` | Optionaler Browserlauf mit echten int8-Modellen, Poseprüfung und Telemetrie | Kein deterministischer Test eines bestimmten Guardzustands |

Die vorhandenen Tests bleiben erhalten. Im Neural-E2E wurde nur die Zustandsassertion
korrigiert: Ein späterer Performance-Fallback ist auch dann zulässig, wenn zuvor bereits
neuronale Posen empfangen wurden. Diese früheren Posen werden weiterhin auf Plausibilität
geprüft; die aktuelle Spiegelansicht richtet sich nach dem aktuellen Driver-Zustand.

## Schwellen und tatsächliche Aktualisierung

Der Inferenz-Guard wertet den Mittelwert der letzten **60 Antworten** aus, erstmals nach
**zehn Antworten**. Die ursprünglichen Schwellen bleiben erhalten:

| Bedingung | Zustand / Verhalten |
|---|---|
| Frischer Lauf mit 20 ms oder genau 25 ms | `running`, `inferEvery = 1`, nominal 30 Hz |
| Mittelwert strikt größer als 25 ms, höchstens 60 ms | `slow`, `inferEvery = 2`, nominal 15 Hz |
| Mittelwert strikt größer als 60 ms | `disabled`, Worker beendet, Punkte-Avatar |
| Bereits gedrosselt, Mittelwert 20 ms oder genau 17,5 ms | Weiter 15 Hz: Recovery-Hysterese |
| Bereits gedrosselt, Mittelwert unter 17,5 ms und kein fortbestehender Render-Befund | Zurück zu 30 Hz |

Eine einzelne langsame Antwort deaktiviert daher nicht sofort einen zuvor gesunden
60er-Mittelwert. 20 ms ist ein passender Test für den frischen Normalbetrieb, aber kein
passender Wert, um die Erholung aus 15 Hz zu erzwingen.

Der deterministische Worker-Test sendet 90 Inputticks mit `t = seq / 30` und wartet kontrolliert
auf den Abschluss der jeweils erwarteten Inferenz. Das ergibt exakt:

| Konfiguration | Inputticks | Posen |
|---|---:|---:|
| `inferEvery = 1` | 30 | 30 |
| `inferEvery = 2` | 30 | 15 |
| Wieder `inferEvery = 1` | 30 | 30 |

Alle 90 Eingangsframes bleiben in der Mapping-History; deren Tensoren werden gegen die
Referenz geprüft. Das überprüft das Scheduling bei freiem Worker. Im Browser können
zusätzlich Busy-Frames ausfallen. Dort wird geprüft, dass die Pose-Sequenzen nach Drosselung
tatsächlich auf jedem zweiten Eingabetick liegen. Daraus wird keine garantierte Browser-
oder Headset-Framerate abgeleitet.

## Neue Testfälle

### Deterministische CPU-Tests

`tests/unit/avatar-driver-performance.test.js` verwendet manuell kontrollierte
Worker-Antworten und virtuelle Timer für:

- Produktionsfenster, erste zehn Antworten und strikte 25-/60-/17,5-ms-Grenzen;
- Verhalten bei Einzelspitzen und Hysterese;
- gemeinsame Entscheidung von Render- und Inferenz-Guard;
- korrekten 15-Hz-Zustand nach Stop/Start und frische Render-Fenster nach Reset;
- alte, doppelte und ungeordnete Posen;
- verspätete `ready`-/`error`-Nachrichten nach terminalen Zuständen;
- erfolgreiche langsame Initialisierung, Initialisierungstimeout und Antwortstimeout;
- Abbruch ausstehender Initialisierungs-/Telemetrie-Promises und Timer-Cleanup.

`tests/unit/worker-regressions.test.js` importiert den produktiven Modul-Worker in einen
echten Node-Worker-Thread. Structured Clone und Transferables überschreiten die echte
Thread-Grenze. Der ORT-Adapter liefert die bereits vorhandenen Python-Fixture-Ausgaben;
einzelne `create()`-/`run()`-Aufrufe werden gezielt angehalten oder abgelehnt. Geprüft werden:

- exakte Inferenzanzahl bei `inferEvery = 1 → 2 → 1` bei vollständiger Eingangshistory;
- Eingangs-/Root-/Pose-/Tick-Konsistenz bei weiteren Frames während einer Inferenz;
- Reset während Mapping und während Tracking, einschließlich verspäteter Fehler;
- Dispose während Mapping/Tracking sowie während beider Session-Erzeugungen und Warm-ups.

### Vier Headless-Browser-Szenarien

`tests/e2e/performance-guard.test.js` startet einen eigenen lokalen HTTP-Server und die
vorhandene Playback-Emulation. Er serviert die echte App samt Worker und Pipeline. Die
Modellbytes und das ORT-Modul werden ausschließlich auf diesem Testserver ersetzt.

1. **COOP/COEP aus:** 20 → 35 → 70 ms echte asynchrone Wartezeit, volle Rate → 15 Hz →
   Punkte-Spiegel; anschließend expliziter Settings-Retry mit neuem Driver.
2. **COOP/COEP an:** dieselbe Sequenz unter tatsächlicher Cross-Origin-Isolation.
3. **Langsames Modellladen und verzögerter Neustart:** `session.create()` bleibt blockiert,
   während ein ganzer Tanz bis `RESULTS` mit Score mindestens 90 durchläuft. Nach Freigabe
   startet der Avatar. Eine später angehaltene Inferenz darf Stop/Reset/Neustart nicht überleben.
4. **Keine weitere Worker-Antwort:** Nach der Antwortfrist folgt Fehlerfallback mit
   Punkte-Spiegel und konkreter Ursache. Renderframes und Scoring laufen weiter.

Die Headerprüfung liest `crossOriginIsolated` und `SharedArrayBuffer` in Dokument **und**
Worker. Ohne Header wird ein Thread gewählt; mit Headern wird das reguläre Hardwarebudget
an die ORT-Grenze weitergegeben. Die Tests ändern diese Browserwerte nicht künstlich und
deaktivieren keine Web-Sicherheitsbedingungen. Der Software-Renderer dient nur dazu, die
Emulation ohne GPU/Headset auszuführen.

## Reproduzierte Fehler und Korrekturen

- **Guard-Konflikt:** 90 Renderintervalle von 18 ms drosselten; zehn schnelle Inferenzantworten
  konnten das trotz weiterhin schlechtem Render-Median wieder aufheben. Recovery verlangt
  nun bei einem Render-Befund ein vollständiges gesundes Render-Fenster.
- **Alte Renderdaten und Status:** Ein teilweise gefülltes langsames Fenster konnte nach
  Stop/Start einen gesunden neuen Lauf deaktivieren. Reset verwirft diese Teilfenster.
  Eine beibehaltene 15-Hz-Rate wird nach Start auch wieder als `slow` angezeigt.
- **Alte Antworten:** Vor Stop/Reset gestartete Posen konnten den neuen Avatar befüllen und
  dessen Guard auslösen. Monotone Sequenzgrenzen im Driver und interne Worker-Generationen
  schließen das aus. Veraltete ORT-Fehler werden ebenfalls verworfen.
- **Tick-Zuordnung:** Die Pose eines alten Inputframes wurde mit dem inzwischen fortgeschrittenen
  History-Tick gemeldet. Tick, Sequenz, Zeitpunkt und Telemetrie beziehen sich nun auf denselben
  Inferenzstart. Die laufenden Pose-/Root-/Feed-Snapshots selbst waren bereits konsistent.
- **Lifecycle und fehlende Antworten:** Dispose während Init ließ die Promise offen;
  Error/Disable konnten Telemetrieanfragen offenlassen. Shutdown beendet ausstehende Anfragen,
  räumt Timer auf und verhindert Wiederbelebung durch späte Nachrichten.
- **Session-Freigabe:** Der Worker konnte laufende ORT-Sessions freigeben oder nach Dispose
  neu erzeugte Sessions behalten. Freigabe erfolgt nun nach Abschluss des jeweiligen
  asynchronen Aufrufs; nach Dispose wird kein `ready` oder `pose` mehr gesendet.
- **Erneuter Versuch und Diagnose:** Die Settings-Aktion ersetzt jetzt einen terminalen Driver.
  `neural.fallback` bewahrt die konkrete Fehlerursache, statt sie durch die allgemeine
  HUD-Meldung zu verdecken.

### Neue Wartefristen

`NeuralAvatarDriver` hat zwei getrennte, konfigurierbare Wall-Clock-Fristen:

- `initTimeoutMs = 120000`: maximal zwei Minuten für Laden, Session-Erzeugung und Warm-up;
- `responseTimeoutMs = 2000`: maximal zwei Sekunden ohne erwartete Pose während der
  Frame-Zufuhr. Bei 15 Hz wird erst nach zwei neuen Inputframes eine Antwort erwartet.

Weitere Eingangsframes verschieben eine laufende Antwortfrist nicht. Eine gültige Pose
setzt sie zurück; Stop/Reset hebt sie auf. Ein Timeout führt zu `error` und Punkte-Fallback,
**nicht** zu einem dauerhaft gespeicherten Performance-Verdict. Für absichtlich sehr langsame
Modelldownloads lässt sich die Initialisierungsfrist am Driver erhöhen.

## Reproduktion

```bash
npm ci
npx playwright-core install chromium

# Nur die gezielte CPU- und Browser-Regression
npm run test:performance

# Vollständige bestehende und ergänzte Regression
npm test
npm run test:e2e
```

`test:e2e` führt Testdateien nacheinander aus, damit der echte ONNX-Smoke-Test nicht mit
den kontrollierten Delay-Fällen um CPU-Zeit konkurriert. Ist Chromium anderweitig installiert,
kann sein Pfad über `PLAYWRIGHT_CHROMIUM` übergeben werden. Fehlende Browser werden ausdrücklich
als **Skip** gemeldet; ein solcher Lauf ist kein bestandener Browsertest.

Der ergänzende Lauf mit echten Modellen verwendet das vorhandene optionale Werkzeug:

```bash
CHROMIUM=/pfad/zu/chromium node tests/net/worker-smoke.mjs --threads 1 --ticks 30
CHROMIUM=/pfad/zu/chromium node tests/net/worker-smoke.mjs --coi --threads 4 --ticks 30
```

`CHROMIUM` ist die bereits vorhandene Pfadoption dieses separaten Tools; die E2E-Suite
verwendet `PLAYWRIGHT_CHROMIUM`.

## Ausgeführte Validierung

| Lauf | Ergebnis |
|---|---|
| Unveränderter Ausgangscommit, vollständige Unit-Baseline | **119/119 bestanden** |
| Unveränderter Ausgangscommit, vollständige E2E-Baseline | **8/8 bestanden** |
| Neuer Stand, vollständige Unit-Suite unter Node **22.23.3** | **151/151 bestanden**, 0 Skips |
| Neuer Stand, vollständige E2E-Suite unter Chromium **153.0.8010.0** | **12/12 bestanden**, 0 Skips |
| Numerische Fixture-Parität, Eingabe/Dekodierung | Maximal **5,72 × 10⁻⁷**, Grenze 10⁻⁵ |
| Numerische Fixture-Parität, Postprocessing/Weltkoordinaten | Maximal **5,64 × 10⁻⁷**, Grenze 10⁻⁴ |
| JavaScript-Syntax, Python-Tool-Kompilierung, `git diff --check` | Bestanden |

32 neue CPU-Testfälle und vier neue Browser-Szenarien ergänzen die Baseline. Der separate
Browserlauf der vier Szenarien bestand ebenfalls vollständig. Vor den Korrekturen schlugen
gezielte Driver-/Worker-Lifecycle-Tests fehl; die beiden Browser-Matrixen scheiterten nach
korrektem 70-ms-Fallback am Settings-Retry, der Timeout-Test an der verdeckten Fehlerursache.

### Zusätzlicher Lauf mit echten ONNX-Modellen

Das vorhandene Worker-Smoke-Tool wurde mit **unveränderten int8-Modellen und ORT 1.20.1**
je einmal für 30 Posen ausgeführt. Diese kurzen Stichproben stammen aus der lokalen
Desktop-/Container-CPU-Umgebung; sie sind keine reproduzierbaren Leistungszusagen:

| Browserbedingung | WASM-Threads | Mittel je Inferenz | Median | Guardzustand nach 30 Posen |
|---|---:|---:|---:|---|
| Ohne Cross-Origin-Isolation | 1 | **37,6 ms** | 36,2 ms | `slow`, `inferEvery = 2` |
| Mit Cross-Origin-Isolation | 4 | **16,4 ms** | 13,4 ms | `running`, `inferEvery = 1` |

Beide Läufe bestanden die vorhandenen Prüfungen auf endliche Posen, plausible Kopfpositionen,
Knochenlängen und Telemetrie. Beide exportierten 30 Telemetrie-Ticks, ohne Konsolenfehler.
Die modellabhängige Inferenz wurde damit auch ohne Isolation tatsächlich ausgeführt; der
kontrollierte ORT-Adapter allein könnte diese Aussage nicht belegen. Die Gesamt-Posezahl pro
Sekunde während der kurzen Drosselungsphase wäre kein sinnvoller Messwert für einen
eingeschwungenen 15-Hz-Betrieb.

### Besonderheiten der Ausführungsumgebung

Die hier verwendete Sandbox erlaubt lokale HTTP-/WebSocket-Verbindungen, blockiert aber
die Aufzählung physischer Netzwerkadapter (`os.networkInterfaces`, Fehler
`uv_interface_addresses`). Für die unveränderten Server-Baselines wurde deshalb ein
ausdrücklich außerhalb des Repositories liegender Test-Preload verwendet, der ausschließlich
diese fehlschlagende Abfrage durch Loopback ersetzt. Serverlogik, HTTP-Header und Browserwerte
wurden dafür nicht verändert. Reale LAN-Adapter-Erkennung ist durch diesen Lauf nicht geprüft.
Der neue Browser-Testserver benötigt diesen Preload nicht.

Der reguläre Playwright-Browserdownload lieferte in dieser Umgebung kein gültiges Archiv.
Verwendet wurde daher eine regulär über das npm-Paket `@sparticuz/chromium` bezogene
Chromium-153-Binary mit den vorhandenen E2E-Startoptionen. Die zusätzlichen Startoptionen
dieses Pakets, die Web-Sicherheitsbedingungen verändern würden, wurden nicht übernommen.
COI und SharedArrayBuffer wurden in Seite und Worker mit tatsächlichen Headern verifiziert.
Auf einer üblichen Entwicklungsmaschine genügen die oben angegebenen Standardkommandos.

## Grenzen der Aussage

Eine echte Quest 2 wurde weiterhin nicht benutzt. Ungeprüft bleiben insbesondere die
anhaltende Headset-Framerate, CPU-/Compositor-Konkurrenz unter realer ONNX-Last, thermische
Drosselung sowie die natürliche Suspend-/Resume-Reihenfolge beim Absetzen oder Öffnen des
Systemmenüs. In der Emulation wird der XR-Render-Guard bewusst nicht aus SwiftShader-
Frameraten gespeist; seine Entscheidungslogik wird separat mit kontrollierten Intervallen
geprüft. Die Tests liefern belastbare Zustands- und Fehlerregressionen, keine neue
Leistungsprognose für die Quest 2.
