# esp_ota

A minimal test project for updating an ESP32's firmware **over WiFi**
instead of over USB, using the Arduino core's built-in `ArduinoOTA` library.

## Why this exists

Once an ESP32 is wired up and closed inside a snake segment, plugging in a
USB cable to push a firmware update means partially disassembling the
robot. OTA (over-the-air) updates let you push new firmware to an ESP32
that's already powered on and on the WiFi network, no cable required. This
project is a small, isolated sketch to prove out that OTA flow — it doesn't
drive any servos or talk MQTT (see [`../esp_mqtt`](../esp_mqtt) for that);
it just blinks the onboard LED, so you have a simple visual way to confirm
an OTA push actually changed the running firmware.

There's a chicken-and-egg catch: OTA only works if the ESP32 is *already*
running firmware built with `ArduinoOTA.begin()` in it. **A brand new
ESP32, or one currently running MQTT/servo firmware without OTA support,
must get this sketch onto it over USB first.** After that first USB flash,
all later updates (including replacing this sketch with a different one, as
long as the replacement also calls `ArduinoOTA.begin()`) can go over WiFi.

## What it does

- Connects to WiFi (`ssid`/`password` at the top of `src/main.cpp`).
- Starts `ArduinoOTA` with hostname `esp32-ota-test` (so you can reach it as
  `esp32-ota-test.local` on networks that support mDNS — most do; if yours
  doesn't, use its IP instead).
- Blinks the onboard LED (GPIO 2) at `interval` milliseconds, non-blockingly
  (so blinking doesn't interfere with listening for OTA pushes).
- Prints WiFi status, OTA progress, and its IP address to Serial (115200
  baud) — useful the first time, over USB, to read off its IP.

## One-time machine setup

### VS Code + PlatformIO

1. Install [VS Code](https://code.visualstudio.com/).
2. Open the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`) and install
   **PlatformIO IDE**. Opening this folder in VS Code will also prompt you
   to install it (see `.vscode/extensions.json`).
3. Restart VS Code once it finishes installing (first run sets up its own
   toolchains, which takes a few minutes).
4. Open `esp_ota/` itself as the VS Code workspace folder — not the repo
   root, and not `Controls/` — so PlatformIO finds `platformio.ini`.

No MQTT broker, no Python packages — this project is firmware-only.

## Flashing a new ESP32 for the first time (USB, required)

`platformio.ini` in this folder is currently set up for OTA uploads
(`upload_protocol = espota`), which won't work yet on a board that has
never been flashed. For the very first flash on a given board:

1. Open `platformio.ini` and comment out the `upload_protocol` and
   `upload_port` lines:
   ```ini
   ; upload_protocol = espota
   ; upload_port = 10.37.233.91
   ```
   With both commented out, PlatformIO falls back to its default USB/serial
   upload (`esptool`).
2. Plug the ESP32 into your laptop over USB.
3. Build + upload: PlatformIO toolbar (checkmark = Build, right-arrow =
   Upload) or, from a terminal in this folder: `pio run -t upload`. If it
   picks the wrong serial port, set `upload_port` to your board's COM port
   (Windows, e.g. `COM5`) or device path (Mac/Linux, e.g.
   `/dev/cu.usbserial-XXXX` / `/dev/ttyUSB0`).
4. Open the Serial Monitor (plug icon in PlatformIO, or `pio device
   monitor`, 115200 baud), and let it connect to WiFi. Note the **IP
   address** it prints — you'll need it for OTA uploads. You should also
   see the onboard LED start blinking once per second.

## Flashing over WiFi after that (OTA)

Once the board is running this sketch (or any sketch with `ArduinoOTA`
built in) and is connected to WiFi:

1. Edit `platformio.ini`, uncomment/set:
   ```ini
   upload_protocol = espota
   upload_port = <the board's IP address, or esp32-ota-test.local>
   ```
   (The `upload_port` currently checked in has a stray value —
   `10.37.233.91 dcf` — replace the whole line with your board's actual
   address; don't leave trailing text after the IP/hostname.)
2. No cable needed — just make sure your laptop is on the same WiFi network
   as the board. Build + upload as before (PlatformIO toolbar, or
   `pio run -t upload`).
3. You'll see upload progress (`Start updating sketch...` /
   `Progress: NN%` / `End`) in the PlatformIO console instead of the usual
   serial flash output. If it fails to find the device, double check the IP
   (boards get a new one if your router's DHCP lease changes) and that
   nothing is blocking mDNS/UDP on the network.
4. To actually verify the new firmware took effect without a serial cable:
   change `interval` in `src/main.cpp` (e.g. `1000` → `200` for a fast
   blink) before pushing the OTA update, and watch the LED change speed
   after the upload completes.

## Notes for wiring this into the real firmware

This sketch is deliberately standalone so OTA can be tested in isolation.
To get OTA-updatable servo/MQTT firmware, the `ArduinoOTA` setup block and
`ArduinoOTA.handle()` call here need to be merged into
[`../esp_mqtt/src/main.cpp`](../esp_mqtt/src/main.cpp) (call
`ArduinoOTA.handle()` from `networkTask`'s loop, alongside `client.loop()`)
— that merge hasn't happened yet, so today the two are separate projects
and a board runs either the MQTT/servo firmware or this OTA test, not both.
