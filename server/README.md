# Dancing Points VR – LAN-Server

Kleiner Node-Server für den Betrieb im WLAN (z. B. VR-Arcade): liefert die WebXR-App per HTTPS
an die Quest, speichert hochgeladene Tänze, Ergebnisse und Aufzeichnungen und vermittelt den
Duo-Modus zwischen zwei Brillen (WebSocket-Relay). Ohne Server läuft die App auch von GitHub
Pages / einem USB-Stick – der Server wird nur für eigene Tänze, Bestenliste und Online-Duo gebraucht.

## Schnellstart

Voraussetzung: Node.js ≥ 18 (https://nodejs.org) auf einem PC/Laptop im selben WLAN wie die Quest.

```bash
cd server
npm install          # einmalig: installiert ws + selfsigned
node server.js       # startet HTTPS auf Port 8443 und HTTP auf Port 8080
```

Beim ersten Start wird ein selbstsigniertes Zertifikat in `server/data/cert/` erzeugt (dauert ein
paar Sekunden). Der Server gibt dann die Adressen aus, z. B.:

```
[dp 18:02:11] Auf der Quest im Browser oeffnen:
[dp 18:02:11]     https://192.168.178.42:8443/
[dp 18:02:11]     http://192.168.178.42:8080/
[dp 18:02:11]     https://localhost:8443/
```

Alternativ aus dem Repository-Wurzelverzeichnis: `node server/server.js` oder `npm start`.

## Auf der Quest öffnen

1. Quest und Server-PC müssen im **selben WLAN** sein (kein Gäste-/Client-Isolation-Netz).
2. Im Quest-Browser die **https://**-Adresse mit der LAN-IP öffnen, z. B. `https://192.168.178.42:8443/`.
   WebXR funktioniert nur über HTTPS (Ausnahme: `localhost`), deshalb nicht die http-Adresse nehmen.
3. **Zertifikatswarnung bestätigen** (einmalig pro Brille): Der Quest-Browser zeigt „Dies ist keine
   sichere Verbindung“ / „Your connection is not private“. Unten auf **„Erweitert“ (Advanced)** tippen
   und dann **„Weiter zu 192.168.… (unsicher)“ / „Proceed to … (unsafe)“**. Das Zertifikat ist selbst
   erstellt und enthält die LAN-IP als SAN; die Warnung kommt nur, weil es von keiner Zertifizierungs-
   stelle unterschrieben ist. Der Browser merkt sich die Ausnahme, bis sich die IP ändert.
4. „VR starten“ tippen. Beim ersten Laden werden die Netzmodelle (~90 MB) geholt; danach sind sie
   im Service-Worker-Cache und der Start geht schnell.
5. Optional als App installieren: Browser-Menü → „Zum Startbildschirm hinzufügen“ (PWA).

Tipp: Dem Server-PC im Router eine feste IP (DHCP-Reservierung) geben, sonst ändert sich die
Adresse und die Zertifikatsausnahme muss neu bestätigt werden (der Server erzeugt bei neuer IP
automatisch ein neues Zertifikat).

## Firewall

Der Server lauscht auf **TCP 8443 (HTTPS + WebSocket)** und **TCP 8080 (HTTP)**. Diese Ports in der
Firewall des Server-PCs freigeben, sonst kommt die Quest nicht durch:

* Windows: beim ersten Start fragt die Windows-Firewall nach – „Zugriff zulassen“ für private
  Netzwerke anklicken. Nachträglich: Windows-Sicherheit → Firewall → „Eine App durch die Firewall
  zulassen“ → Node.js (privat). Oder als Administrator:
  `netsh advfirewall firewall add rule name="Dancing Points VR" dir=in action=allow protocol=TCP localport=8443,8080`
* Linux (ufw): `sudo ufw allow 8443/tcp && sudo ufw allow 8080/tcp`
* macOS: Systemeinstellungen → Netzwerk → Firewall → Node erlauben.

Es gibt **keine Authentifizierung** – der Server ist nur für ein vertrauenswürdiges LAN gedacht, nicht
für das offene Internet.

## Optionen

```
node server.js [--port 8443] [--http-port 8080] [--http] [--no-http] [--no-https]
               [--host 0.0.0.0] [--dir ../webxr] [--data ./data] [--cert x.pem --key y.pem]
               [--no-coep] [--verbose] [--quiet]
```

| Option | Bedeutung |
|---|---|
| `--port` | HTTPS-Port (Standard 8443) |
| `--http-port` | HTTP-Port (Standard 8080, `0` = freier Port) |
| `--http` / `--no-http` | HTTP-Listener an (Standard) / aus |
| `--no-https` | kein HTTPS, kein Zertifikat (nur für Tests/Entwicklung am PC) |
| `--host` | Bind-Adresse (Standard: alle Interfaces) |
| `--dir` | Verzeichnis der Web-App (Standard: `../webxr` neben `server.js`) |
| `--data` | Datenverzeichnis (Standard: `server/data`) – Zertifikat, Uploads, Ergebnisse, Aufzeichnungen |
| `--cert`, `--key` | eigenes Zertifikat (PEM) statt des selbstsignierten |
| `--no-coep` | die Cross-Origin-Isolation-Header (COOP/COEP) weglassen, falls externe Audio-URLs ohne CORS geladen werden sollen |
| `--verbose` / `--quiet` | jede Anfrage loggen / nur Fehler |

Nach dem Start schreibt der Server eine Zeile `READY {"https":8443,"http":8080,"urls":[…]}` auf
stdout (für Skripte/Tests). `Ctrl+C` beendet ihn sauber.

## Was der Server macht

* **Statische Dateien** aus `--dir` mit korrekten MIME-Typen (`.mjs`/`.js` → `text/javascript`,
  `.wasm` → `application/wasm`, `.onnx` → `application/octet-stream`, `.webmanifest`), `Cache-Control:
  no-cache` für HTML/JSON/`sw.js`, ETag/304, Range-Requests (Audio), und die Header
  `Cross-Origin-Opener-Policy: same-origin` / `Cross-Origin-Embedder-Policy: require-corp`
  (schaltet `SharedArrayBuffer` frei, damit onnxruntime-web später mit Threads laufen kann).
  Gleiche-Origin-Dateien laden damit uneingeschränkt; fremde Origins müssen CORS/CORP liefern
  (Ausweg: `--no-coep`).
* **REST-API** (JSON, CORS offen):
  * `GET /api/info` – Version, URLs, `https`-Flag, Limits
  * `GET /api/choreos` – Index der mitgelieferten Tänze (`webxr/choreos/index.json`) zusammengeführt
    mit den hochgeladenen (`source: "upload"`, `url: /api/choreos/<id>`); ein Upload mit gleicher id
    ersetzt den mitgelieferten Eintrag (`replaces: "shipped"`)
  * `GET /api/choreos/:id`, `POST /api/choreos` (Body ≤ 5 MB, wird validiert, landet in
    `data/choreos/<id>.json`), `DELETE /api/choreos/:id` (nur Uploads)
  * `GET /api/results?choreoId=&player=&limit=` (neueste zuerst), `POST /api/results` (≤ 1 MB;
    `data/results.json`, max. 5000 Einträge)
  * `GET /api/runs?choreoId=&limit=` (nur Metadaten), `GET /api/runs/:id` (komplette Aufzeichnung),
    `POST /api/runs` (≤ 5 MB; `data/runs/<choreoId>/<id>.json`, **max. 50 pro Tanz**, älteste fliegen
    raus), `DELETE /api/runs/:id`
  * `GET /api/rooms` – offene Duo-Räume (Debug/Lobby)
* **WebSocket-Relay** unter `wss://<ip>:8443/ws` (bzw. `ws://…:8080/ws`) für den Online-Duo-Modus:
  Räume mit 4-Buchstaben-Code, max. 2 Spieler, `ping`/`pong` für die Uhrensynchronisation,
  `start` mit server-gestempeltem `startAt`, `state`-Nachrichten werden an den Mitspieler
  weitergereicht. Nachrichten ≤ 4 KB, max. 30 Nachrichten/s pro Client, leere Räume werden gelöscht.
  Das Protokoll ist in `docs/DESIGN.md` („Appendix: server & duo API“) beschrieben.

## Eigene Tänze und Musik

Aufgenommene Tänze aus der Brille werden mit „Speichern“ per `POST /api/choreos` hochgeladen und
erscheinen sofort bei allen Brillen im Menü unter „Eigene Tänze“. Manuell: eine Choreo-JSON-Datei
(Format siehe `webxr/choreos/README.md`) mit `curl` hochladen:

```bash
curl -k -X POST -H "Content-Type: application/json" --data-binary @mein-tanz.json https://localhost:8443/api/choreos
```

Musikdateien (`audio.url` in der Choreo) am einfachsten nach `webxr/choreos/` legen und relativ
referenzieren – der Server liefert sie mit Range-Unterstützung aus. Die Lizenz für eigene Musik
liegt beim Betreiber.

## Tests

```bash
npm test                       # im Repository-Wurzelverzeichnis: enthält tests/unit/server.test.js
node --test tests/unit/server.test.js
```

Der Server-Test startet `server.js` mit `--http --http-port 0 --no-https --data <tmp>` als
Kindprozess und braucht das `ws`-Paket aus `server/node_modules` (`npm install` in `server/`).
