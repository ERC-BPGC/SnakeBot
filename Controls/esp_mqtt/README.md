# esp_mqtt

Wireless, multi-ESP32 servo control over MQTT. One ESP32 per snake segment;
a single PC-side script plans the gait and broadcasts synchronized,
timestamped commands that each ESP32 filters down to its own segment.

This doc is written so you can get this running from a blanexk laptop and a
box of fresh ESP32s — setup steps included below the protocol/pieces
overview.

## Protocol

A single MQTT message on topic `servos/sync_command`:

```json
{
  "ts": 1700000000000,
  "data": {
    "ESP_01": [servo1_angle, servo2_angle],
    "ESP_02": [servo1_angle, servo2_angle]
  }
}
```

`ts` is a Unix timestamp in **milliseconds**, in the future. Every ESP32
subscribes to the same topic and receives every message, but each one
(`src/main.cpp`) is identified by its own `THIS_ESP_ID`, and only queues the
entry under its own key. It executes that command once its own
NTP-synced clock reaches `ts`. Because all the ESP32s independently sync
their clocks to the same NTP servers and then just wait for the same
target timestamp, they move in lockstep despite each having a different
WiFi/MQTT round-trip latency.

## Pieces

- **`src/main.cpp`** — firmware, flashed once per ESP32 with a distinct
  `THIS_ESP_ID` (`ESP_01`, `ESP_02`, ...) and pins set for that board.
  Connects to WiFi + NTP, then runs two FreeRTOS tasks: one
  receiving/queuing MQTT commands (`networkTask`, core 0), one executing
  due commands against its two servos (`servoTask`, core 1).
- **`controller_full.py`** — the PC-side driver. Fits the snake's segments
  to a travelling sine wave (this is the merged/evolved `figure_out_snake`
  path-planner — see [`../README.md`](../README.md) for its earlier
  standalone history), converts each segment's pose to a servo angle pair,
  and publishes them on a rolling schedule. While it's running: `a` pauses,
  `q` quits.
- **`tools/mqtt_manual_tester.py`** — a minimal interactive sender for
  bench-testing the firmware directly (single moves or a canned 3-move
  sequence) without running the full path-planner. Useful for confirming a
  freshly flashed ESP32 is receiving and moving before wiring it into a full
  gait.

## One-time machine setup

### 1. VS Code + PlatformIO (to build/flash the firmware)

1. Install [VS Code](https://code.visualstudio.com/).
2. Inside VS Code, open the Extensions panel (`Ctrl+Shift+X` /
   `Cmd+Shift+X`) and install **PlatformIO IDE**. Opening this folder
   (`esp_mqtt/`) in VS Code will also prompt you to install it, since it's
   listed in `.vscode/extensions.json`.
3. Restart VS Code when PlatformIO finishes installing (it needs to set up
   its own Python environment and toolchains — first run can take a few
   minutes).
4. Open `esp_mqtt/` itself as the VS Code workspace folder (PlatformIO
   looks for `platformio.ini` at the root of what you open) — not the repo
   root, and not `Controls/`.

### 2. An MQTT broker (Mosquitto)

The broker is the message hub — it must run somewhere reachable by both
your laptop and every ESP32 (same WiFi network/hotspot). Running it on your
laptop is the easiest setup.

**Windows**
1. Download the installer from
   [mosquitto.org/download](https://mosquitto.org/download/) and run it
   (keep the default install path).
2. Mosquitto 2.x refuses remote connections by default — it only listens on
   `localhost` and requires auth unless told otherwise. Edit
   `C:\Program Files\mosquitto\mosquitto.conf` (as Administrator) and add:
   ```
   listener 1883 0.0.0.0
   allow_anonymous true
   ```
3. Restart the service: open Services (`services.msc`), find
   "Mosquitto Broker", restart it. Or run it in a terminal directly:
   `"C:\Program Files\mosquitto\mosquitto.exe" -c "C:\Program Files\mosquitto\mosquitto.conf" -v`
4. Allow it through Windows Firewall when prompted (or manually allow TCP
   1883, private networks).

**macOS**
1. `brew install mosquitto`
2. Same default-is-localhost-only caveat applies. Edit the config (path
   printed by `brew info mosquitto`, typically
   `/opt/homebrew/etc/mosquitto/mosquitto.conf` on Apple Silicon or
   `/usr/local/etc/mosquitto/mosquitto.conf` on Intel) and add:
   ```
   listener 1883 0.0.0.0
   allow_anonymous true
   ```
3. `brew services restart mosquitto`

**Linux (Debian/Ubuntu)**
1. `sudo apt update && sudo apt install mosquitto mosquitto-clients`
2. Edit `/etc/mosquitto/mosquitto.conf`, add:
   ```
   listener 1883 0.0.0.0
   allow_anonymous true
   ```
3. `sudo systemctl restart mosquitto` (and `sudo systemctl enable mosquitto`
   if you want it to survive a reboot).

**Verify the broker is reachable** (run from the laptop, then again from
another machine on the same network to check it's not localhost-locked):
```
mosquitto_sub -h <broker-ip> -t test &
mosquitto_pub -h <broker-ip> -t test -m "hello"
```
You should see `hello` printed by the subscriber. `mosquitto_sub`/`_pub`
ship with the broker install above (on Windows they're in the same install
folder — add it to `PATH`, or run them from there directly).

> `allow_anonymous true` with no listener restrictions is fine for a local
> lab/hotspot network you control, but don't do this on a network you don't
> trust — anyone on it could publish servo commands.

### 3. Python environment (to run the controller scripts)

```
pip install paho-mqtt numpy matplotlib scipy keyboard
```
(`keyboard` needs to run as Administrator/root on some platforms to read
global key presses — that's what `controller_full.py` uses for its
pause/quit hotkeys.)

## Running it

1. **Find your broker's IP**: on the machine running Mosquitto, check
   `ipconfig` (Windows) / `ifconfig` or `ip addr` (Mac/Linux) for its
   address on the shared WiFi network — this is what everything else
   connects to.
2. **Set that IP** in two places:
   - `MQTT_BROKER` near the top of `controller_full.py`
   - `mqtt_server` near the top of `src/main.cpp`
3. **Set your WiFi network** in `src/main.cpp`: `ssid` / `password`.
4. **Flash each ESP32**, one at a time, over USB:
   - Before each flash, edit `src/main.cpp`: give it a unique
     `THIS_ESP_ID` (`"ESP_01"`, `"ESP_02"`, ...) and the correct
     `SERVO_PIN_1`/`SERVO_PIN_2` for that board's wiring.
   - Plug the ESP32 in via USB, then in VS Code use the PlatformIO toolbar
     (checkmark icon = Build, right-arrow icon = Upload), or from a
     terminal in this folder: `pio run -t upload`. PlatformIO
     auto-detects the USB serial port; if it picks the wrong one, set
     `upload_port` in `platformio.ini` to your board's COM port /
     `/dev/tty*` device.
   - Open the Serial Monitor (plug icon in PlatformIO, or `pio device
     monitor`, 115200 baud) to confirm it connects to WiFi, syncs NTP time,
     and connects to the MQTT broker.
5. **Sanity-check each ESP32** with the manual tester before trusting the
   full gait:
   ```
   python tools/mqtt_manual_tester.py
   ```
   (update the broker IP/ESP keys at the top of that script to match what
   you just flashed), and confirm the servos move.
6. **Run the full controller**:
   ```
   python controller_full.py
   ```
   Set `num_segments` near the top of the script to match how many ESP32s
   you actually flashed — it builds one `ESP_XX` key per segment, so a
   mismatch here means some ESP32s never get commanded (too few) or some
   commands go to keys nothing is listening for (too many).

## Troubleshooting

- **ESP32 connects to WiFi but never reaches MQTT / times out on NTP**:
  it needs a route to the public internet for NTP (`pool.ntp.org`), not
  just to the broker. A phone hotspot with data works; a fully offline
  router does not — point `ntpServer` in `main.cpp` at a local NTP server
  if you're fully air-gapped.
- **Python script connects to the broker but the ESP32 never receives
  anything**: almost always the Mosquitto "localhost only" default — check
  the `listener`/`allow_anonymous` lines above.
- **One segment doesn't move**: check its `THIS_ESP_ID` actually matches an
  `ESP_XX` key the controller is sending (`num_segments` in
  `controller_full.py` vs. how many boards you flashed), and that its
  Serial Monitor shows `Queued cmd for ...` when the controller runs.
- **Movements are jerky/desynced across segments**: usually one board's NTP
  sync failed silently or its WiFi is flaky — check its Serial Monitor for
  repeated reconnect messages.
