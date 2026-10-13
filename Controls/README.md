# Controls

Path-planning: fitting the modular snake's rigid segments to a travelling
sine wave so each segment's servo angles can be derived (see the root
`README.md` for the motion concept), plus everything that sends those
angles to the hardware.

There have been two transports for that, in order

## Current: wireless, multi-ESP32 over MQTT — `esp_mqtt/`

One ESP32 per segment, all synced over WiFi/MQTT. See
[`esp_mqtt/README.md`](esp_mqtt/README.md) for the message protocol and how
to run it. The PC-side script, `esp_mqtt/controller_full.py`, is the
actively used control code — it's the same spline/sine-wave path-planner as
`figure_out_snake_v7.2 PySerial.py` below (same algorithm, same servo-angle
correction formula), just with MQTT publishing instead of serial writes,
and later a travelling phase-shift gait layered on top.

## Earlier: tethered, single-Arduino over USB serial

- **`figure_out_snake_v7.2 PySerial.py`** — computes the segment angles and
  streams them over direct USB-serial, one `horizontal/vertical`
  comma-separated pair per segment per line (baud 230400).
- **`Arduino Controller/AngleReceivingCode.ino`** — the receiving end:
  parses that same line format and drives up to 6 servo pairs.
  - `90Deg.ino`, `Snake.ino` — smaller earlier sketches kept for reference
    (fixed test poses / open-loop gait, no serial input).

## `esp_ota/`

Standalone PlatformIO project for testing over-the-air firmware updates on
an ESP32 (WiFi + `ArduinoOTA`, blinks the onboard LED to show it's alive).
Not wired into `esp_mqtt`'s servo control path — a separate utility for
validating OTA flashing, kept here since it's ESP32 firmware tooling too.

## `archive/`

The version history: `figure_out_snake_v1` → `v7.1` (the path-planning
algorithm's evolution, each superseded by the next) and
`controller_full_v1.py` (the first merge of `figure_out_snake` with MQTT,
superseded by `esp_mqtt/controller_full.py`). Kept for reference, not meant
to be run.
