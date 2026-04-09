# IoTEdgeSense_PowerGrid_SDN

**C++ IoT Data Acquisition and Processing firmware for constrained edge devices.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Platform: Linux](https://img.shields.io/badge/Platform-Embedded%20Linux-green.svg)]()

---

## Overview

IoTEdgeSense is a modular firmware framework for edge IoT nodes running embedded Linux (Raspberry Pi, BeagleBone, Yocto-based boards, etc.). It provides a clean abstraction layer for acquiring sensor data over I²C, SPI, UART and GPIO, running it through a configurable processing pipeline, and publishing it upstream via MQTT — all designed to be lightweight enough for constrained devices.

The codebase is structured as a **static library + thin executable**, making it straightforward to embed into larger platforms or extend with new sensor drivers.

### Integration with Power Grid SDN Platform

IoTEdgeSense is designed to act as the **field data acquisition layer** in a Software-Defined Network (SDN) power grid monitoring system:

```
[Field Sensors]
   I²C / SPI / UART / GPIO
         │
  [IoTEdgeSense Edge Node]
   Acquire → Filter → Detect Anomalies → Compress
         │
      MQTT / TLS
         │
  [SDN Grid Monitoring Platform]
   Topology Engine │ Anomaly Detection │ Dashboard
```

Each edge node publishes real-time telemetry (voltage, current, temperature, status) to an MQTT broker. The SDN platform subscribes to these feeds to build network state, detect faults, and reconfigure routing.

---

## Repository Structure

```
IoTEdgeSense/
├── main.cpp                        # Firmware entry point
├── config.h                        # All device/network/power configuration
├── CMakeLists.txt                  # CMake build system
├── LICENSE
├── README.md
│
├── src/
│   ├── system/
│   │   ├── error_handler.h         # Singleton error registry (30 error codes)
│   │   └── error_handler.cpp
│   │
│   ├── sensors/
│   │   ├── sensor_base.h/.cpp      # Abstract SensorBase + SensorReading struct
│   │   ├── i2c_sensor.h/.cpp       # I²C driver (Linux i2c-dev)
│   │   ├── gpio_sensor.h/.cpp      # GPIO driver (Linux sysfs)
│   │   ├── spi_sensor.h/.cpp       # SPI driver (Linux spidev)
│   │   └── uart_sensor.h/.cpp      # UART driver (POSIX termios)
│   │
│   ├── processing/
│   │   ├── data_filter.h/.cpp      # MovingAverage, Threshold, Delta, Median
│   │   └── data_processor.h/.cpp   # Pipeline, aggregation, Z-score anomaly detection
│   │
│   └── comm/
│       ├── mqtt_client.h/.cpp      # MQTT client (Paho C++ or stub)
│       └── comm_manager.h/.cpp     # Orchestrates publish/subscribe + JSON encoding
│
├── tests/
│   └── unit/                       # GoogleTest unit tests (WIP)
│
├── docs/
│   └── architecture.md             # Detailed architecture notes
│
└── scripts/
    └── deploy.sh                   # Deployment helper
```

---

## Architecture

### Sensor Abstraction

All sensor drivers inherit from `Sensors::SensorBase` and must implement six pure virtual methods:

| Method | Purpose |
|---|---|
| `initialize()` | Open hardware interface, run self-test |
| `read()` | Return a `SensorReading` with timestamp and values |
| `calibrate()` | Run sensor-specific calibration sequence |
| `sleep()` | Enter low-power mode |
| `wakeUp()` | Resume from low-power mode |
| `selfTest()` | Run built-in diagnostics |

Every `SensorReading` carries: `timestamp` (ms), `values[]` (floats), `unit`, `sensorId`, and a `valid` flag.

### Processing Pipeline

`Data::DataProcessor` applies a chain of `DataFilter` objects in sequence:

```
Raw Readings
    │
    ▼
MovingAverageFilter  →  smooths noise (configurable window)
    │
    ▼
ThresholdFilter      →  rejects physically impossible values
    │
    ▼
DeltaFilter          →  suppresses redundant unchanged readings
    │
    ▼
(optional) MedianFilter → removes outlier spikes
    │
    ▼
Processed Readings   →  anomaly detection, aggregation, publish
```

### Communication

`Communication::CommManager` owns the `MQTTClient` and:
- Serialises `SensorReading` objects to compact JSON
- Publishes to configurable MQTT topics (telemetry, status, commands)
- Subscribes to the command topic and routes messages to registered handlers
- Supports TLS via Paho's SSL options or config-supplied certificate paths

---

## Building

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt install build-essential cmake git

# For real MQTT (optional)
sudo apt install libpaho-mqtt-dev libpaho-mqttpp-dev
```

### Quick build (stub mode — no hardware required)

```bash
git clone https://github.com/EWC-Compute-Platform/IoTEdgeSense.git
cd IoTEdgeSense
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DIOT_STUB_HARDWARE=ON
cmake --build . --parallel
./bin/iotedgesense
```

### Build with real hardware drivers

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DIOT_STUB_HARDWARE=OFF
cmake --build . --parallel
```

### Build with real MQTT (Paho C++)

```bash
cmake .. -DIOT_STUB_HARDWARE=OFF -DIOT_USE_PAHO_MQTT=ON
cmake --build . --parallel
```

### Build with unit tests

```bash
cmake .. -DIOT_BUILD_TESTS=ON -DIOT_STUB_HARDWARE=ON
cmake --build . --parallel
ctest --output-on-failure
```

### Build options summary

| CMake option | Default | Description |
|---|---|---|
| `IOT_STUB_HARDWARE` | `ON` | Hardware-free stub drivers |
| `IOT_USE_PAHO_MQTT` | `OFF` | Link against Paho MQTT C++ |
| `IOT_BUILD_TESTS` | `OFF` | Compile GoogleTest unit tests |
| `IOT_ENABLE_ASAN` | `OFF` | AddressSanitizer (Debug only) |

---

## Configuration

All tunable parameters live in `config.h`. Key sections:

```cpp
// Device identity
constexpr char DEVICE_ID[]   = "IOT_EDGE_DEVICE_001";
constexpr char FIRMWARE_VERSION[] = "1.0.0";

// MQTT broker
constexpr char MQTT_BROKER[] = "mqtt.example.com";
constexpr uint16_t MQTT_PORT = 8883;           // 8883 = TLS

// Sampling
constexpr uint32_t DEFAULT_SAMPLING_RATE_MS = 1000;

// Power management
constexpr bool ENABLE_LOW_POWER_MODE  = true;
constexpr uint32_t SLEEP_DURATION_MS  = 10000;
```

Secrets (WiFi password, MQTT credentials) are intentionally left empty in `config.h` and should be injected at runtime from a secure store (environment variable, encrypted file, TPM).

---

## Adding a New Sensor

1. Create `src/sensors/my_sensor.h` and `.cpp` inheriting from `SensorBase`.
2. Implement all six virtual methods.
3. Add `src/sensors/my_sensor.cpp` to `SENSOR_SOURCES` in `CMakeLists.txt`.
4. Instantiate and add to the `sensors` vector in `main.cpp`.

```cpp
// Example: custom I²C temperature sensor
class TMP117Sensor : public Sensors::I2CSensor {
public:
    TMP117Sensor(uint8_t id, uint8_t bus)
        : I2CSensor(id, "TMP117", bus, 0x48) {}

    SensorReading read() override {
        // Read 2-byte register 0x00, convert to °C
        uint8_t hi = 0, lo = 0;
        readRegister(0x00, hi);
        readRegister(0x01, lo);
        int16_t raw = (hi << 8) | lo;
        SensorReading r;
        r.values   = { raw * 0.0078125f };  // 7.8125 m°C per LSB
        r.unit     = "degC";
        r.sensorId = mId;
        r.valid    = true;
        // ... set timestamp
        return r;
    }
    // Other virtuals delegate to I2CSensor base
};
```

---

## MQTT Message Format

### Telemetry (single reading)
```json
{
  "ts": 1712345678901,
  "id": 1,
  "unit": "raw",
  "valid": true,
  "values": [0.123456, -0.045678, 0.981234]
}
```

### Batch telemetry
```json
[
  { "ts": 1712345678901, "id": 1, "unit": "V", "valid": true, "values": [3.295] },
  { "ts": 1712345678902, "id": 2, "unit": "digital", "valid": true, "values": [0] }
]
```

### Inbound command (subscribe topic)
```json
{ "cmd": "set_rate", "params": { "rate_ms": 500 } }
```

---

## Error Handling

All subsystem errors funnel through the singleton `System::ErrorHandler`. Register a callback at startup to react to errors in real time:

```cpp
System::ErrorHandler::getInstance().registerCallback(
    [](const System::ErrorEvent& ev) {
        if (ev.severity >= System::ErrorSeverity::ERROR)
            myLogger.write(ev.source + ": " + ev.message);
    }
);
```

Error codes are grouped by subsystem prefix (`0x01xx` sensors, `0x02xx` comm, `0x03xx` data, `0x04xx` system).

---

## Roadmap

### Phase 1 — Foundation (current)
- [x] Abstract sensor base + SensorReading type
- [x] I²C, SPI, GPIO, UART concrete drivers
- [x] Four-filter processing pipeline
- [x] DataProcessor: aggregation, Z-score anomaly detection, compress/decompress
- [x] MQTTClient with Paho C++ and stub backends
- [x] CommManager with JSON serialisation and command routing
- [x] CMake build system with hardware stub mode
- [x] Structured error handling

### Phase 2 — Power Grid Bridge
- [ ] `PowerGridBridge` module: maps sensor readings to grid telemetry schema
- [ ] Voltage / current / frequency sensor specialisations
- [ ] Fault event publishing (line outage, over-voltage, under-frequency)
- [ ] SDN platform REST/gRPC integration adapter

### Phase 3 — Hardening
- [ ] GoogleTest unit test suite
- [ ] GitHub Actions CI pipeline
- [ ] OTA firmware update mechanism
- [ ] Persistent local storage queue (offline buffering)
- [ ] Watchdog and power-management integration
- [ ] Yocto / Buildroot layer recipe

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-sensor`
3. Commit your changes: `git commit -m 'Add TMP117 temperature sensor'`
4. Push: `git push origin feature/my-sensor`
5. Open a Pull Request

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

---

## Related Projects

- **Power Grid SDN Monitor** — Software-defined network monitoring platform that consumes IoTEdgeSense telemetry for grid topology analysis and fault detection.
- [Eclipse Paho MQTT C++](https://github.com/eclipse/paho.mqtt.cpp)
- [Linux i2c-tools](https://i2c.wiki.kernel.org/index.php/I2C_Tools)
