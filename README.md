# Power Grid Edge Project

**C++ IoT edge firmware + Python ML pipelines for software-defined power grid monitoring.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Platform: Embedded Linux](https://img.shields.io/badge/Platform-Embedded%20Linux-green.svg)]()
[![Organisation: EWC Compute Platform](https://img.shields.io/badge/EWC-Compute%20Platform-orange.svg)](https://github.com/EWC-Compute-Platform)

---

## Overview

IoTEdgeSense_PowerGrid_SDN is a monorepo combining two previously separate projects into a single, coherent platform for power grid monitoring at the edge:

- **IoTEdgeSense** — modular C++17 firmware for constrained embedded Linux devices (Raspberry Pi, BeagleBone, Yocto-based gateways). Acquires sensor data over I²C, SPI, UART and GPIO, runs it through a configurable processing pipeline, and publishes structured telemetry over MQTT/TLS.

- **PowerGrid_SDN** — the power grid domain layer. Translates raw sensor readings into electrical domain objects (three-phase measurements, fault events), applies IEC 60038 / EN 50160 threshold checks across 20 fault categories, and provides a bridge to software-defined network (SDN) topology management.

- **ML Predictive Maintenance** — Python training pipelines and C++ inference engine for transformer health monitoring and power quality anomaly detection. Physics-informed feature engineering (IEC 60076-7), PyTorch models exported to ONNX, deployed at the edge via ONNX Runtime.

```
[Field Sensors]                          [SDN Grid Platform]
  I²C / SPI / UART / GPIO                 Topology Engine
         │                                Fault Localisation
  [IoTEdgeSense Edge Node]      MQTT/TLS  Analytics Dashboard
   Acquire → Filter → Bridge  ─────────►  ML Alert Receiver
         │
  [ML Inference Engine]
   Thermal Ageing (IEC 60076-7)
   Power Quality Anomaly Detection
   → MaintenanceAlert published upstream
```

---

## Repository Structure

```
IoTEdgeSense_PowerGrid_SDN/
│
├── config.h                        # All device / network / power configuration
├── main.cpp                        # Firmware entry point
├── CMakeLists.txt                  # CMake build system (C++17, multi-option)
├── LICENSE                         # Apache 2.0
├── README.md
│
├── src/                            # C++ firmware sources
│   ├── system/
│   │   ├── error_handler.h/.cpp    # Singleton error handler (60+ codes, subsystem routing)
│   │   ├── logger.h/.cpp           # Structured logger (4 destinations, ANSI colour, rotation)
│   │   └── watchdog.h/.cpp         # Multi-channel software watchdog (/dev/watchdog integration)
│   │
│   ├── sensors/
│   │   ├── sensor_base.h/.cpp      # Abstract SensorBase + SensorReading struct
│   │   ├── i2c_sensor.h/.cpp       # I²C driver (Linux i2c-dev)
│   │   ├── gpio_sensor.h/.cpp      # GPIO driver (Linux sysfs)
│   │   ├── spi_sensor.h/.cpp       # SPI driver (Linux spidev)
│   │   └── uart_sensor.h/.cpp      # UART driver (POSIX termios)
│   │
│   ├── processing/
│   │   ├── data_filter.h/.cpp      # MovingAverage, Threshold, Delta, Median filters
│   │   └── data_processor.h/.cpp   # Pipeline, aggregation, Z-score anomaly detection
│   │
│   ├── comm/
│   │   ├── mqtt_client.h/.cpp      # MQTT client (Paho C++ or stub backend)
│   │   └── comm_manager.h/.cpp     # JSON serialisation, batch telemetry, command routing
│   │
│   ├── Grid/                       # Power grid domain layer
│   │   ├── grid_types.h            # ThreePhaseMeasurement, GridFaultEvent, GridNodeDescriptor
│   │   ├── grid_config.h           # IEC 60038 / EN 50160 thresholds (50/60 Hz presets)
│   │   ├── grid_sensors.h          # VoltageSensor, CurrentSensor, FrequencySensor, PowerMeter, ThermalSensor
│   │   ├── fault_detector.h/.cpp   # 20 fault types, 4 severity levels, active fault registry
│   │   └── power_grid_bridge.h/.cpp# Sensor→domain decoder, MQTT publisher, callback API
│   │
│   └── ml/                         # C++ ML inference engine
│       ├── onnx_inference.h/.cpp   # ONNX Runtime wrapper (ONNX_STUB for CI)
│       └── predictive_maintenance.h/.cpp  # IEC 60076-7 thermal model, alert dispatch
│
└── ml/                             # Python ML training + service layer
    ├── requirements.txt            # numpy, pandas, torch, onnx, paho-mqtt, sklearn
    │
    ├── features/
    │   └── transformer_features.py # IEC 60076-7 physics model, feature extractor (43 features),
    │                               # synthetic data generator with fault precursors
    ├── models/
    │   ├── thermal_ageing_model.py # Hybrid PyTorch model: physics residual + CNN + multi-task heads
    │   ├── anomaly_detector.py     # Autoencoder for power quality anomaly detection (30 features)
    │   ├── train.py                # Thermal model training pipeline (early stopping, ONNX export)
    │   └── train_anomaly.py        # Autoencoder training + threshold calibration
    │
    ├── export/
    │   └── export_to_onnx.py       # PyTorch → ONNX with numerical validation
    │
    └── service/
        └── prediction_service.py   # MQTT subscriber → inference → MaintenanceAlert publisher
```

---

## Architecture

### Layer 1 — Hardware Acquisition

Physical sensors connect via standard embedded bus interfaces. The `SensorBase` abstraction provides a uniform six-method interface (`initialize`, `read`, `calibrate`, `sleep`, `wakeUp`, `selfTest`) with a full state machine and structured error propagation. Every `SensorReading` carries a millisecond timestamp, typed values array, unit string, and validity flag.

### Layer 2 — Edge Processing

Raw measurements pass through a configurable `DataProcessor` filter chain:

```
SensorReading (raw)
  → MovingAverageFilter   (noise smoothing, configurable window)
  → ThresholdFilter       (reject physically impossible values)
  → DeltaFilter           (suppress redundant readings — bandwidth saving)
  → MedianFilter          (optional — remove outlier spikes)
  → Z-score anomaly check (statistical outlier flagging)
  → aggregation / compression → MQTT publish
```

### Layer 3 — Grid Domain Translation

`PowerGridBridge` decodes sensor readings into typed electrical domain objects using a sensor registration system. Each registered sensor has a declared `SensorRole` (VOLTAGE, CURRENT, FREQUENCY, POWER_METER, THERMAL) that determines how its `values[]` array is interpreted. The `FaultDetector` evaluates every assembled `ThreePhaseMeasurement` against the IEC 60038 / EN 50160 threshold matrix.

**Fault detection covers 20 categories across 4 subsystems:**

| Subsystem | Fault types |
|---|---|
| Voltage | Over-voltage, under-voltage, unbalance, sag, swell, interruption |
| Current | Overcurrent, earth fault, short circuit, unbalance |
| Frequency | Over-frequency, under-frequency, instability (RoCoF) |
| Power quality & thermal | Low power factor, harmonics, flicker, transformer/cable overheat |

### Layer 4 — ML Predictive Maintenance

Two ML models run at the edge and/or in the Python service layer:

**1. TransformerThermalModel** (thermal ageing + RUL prediction)
- Physics-informed feature engineering based on IEC 60076-7 differential thermal model
- 43 features: 15 physics (K, θ_H, V, cumulative ageing…) + 24 rolling window stats + 4 temporal
- Hybrid PyTorch architecture: physics residual MLP + 1D CNN temporal branch + multi-task fusion
- Three outputs: RUL in days, normalised ageing state [0–1], fault probability
- V = 1.0 at 110°C reference (Arrhenius equation, exact per IEC 60076-7 §6.3)

**2. GridAutoencoder** (power quality anomaly detection)
- Unsupervised — trains on normal operating data only, no labelled faults required
- 30 features covering voltage unbalance, current unbalance, frequency deviation, RoCoF, PF, rolling stats
- Detects 6 anomaly types: voltage sag/swell, overcurrent, unbalance, low PF, frequency deviation
- Reconstruction error > calibrated threshold → anomaly alert

Both models are exported to ONNX and loaded by the C++ `OnnxInferenceEngine` for real-time edge inference. The Python `PredictionService` runs alongside the firmware as an MQTT microservice, providing richer batch inference, model reloading, and alert aggregation.

### Layer 5 — Communication and SDN Integration

`CommManager` owns the MQTT client and provides:
- Serialisation of `SensorReading` and `ThreePhaseMeasurement` objects to compact JSON
- Configurable topic routing: `devices/data` (telemetry), `devices/maintenance/{node}` (ML alerts), `devices/status` (heartbeat)
- TLS 1.2/1.3 with mutual certificate authentication
- QoS 1 for fault events and ML alerts; retained status messages for SDN controller reconnection
- Per-command handler registration for inbound control commands (`set_rate`, `set_threshold`…)

---

## Building (C++ firmware)

### Prerequisites

```bash
# Ubuntu / Debian
sudo apt install build-essential cmake git libpthread-stubs0-dev

# Optional: real MQTT broker connection
sudo apt install libpaho-mqtt-dev libpaho-mqttpp-dev

# Optional: ONNX Runtime for real ML inference
# Download from https://github.com/microsoft/onnxruntime/releases
# or: apt install libonnxruntime-dev (if available in your distro)
```

### Quick build — stub mode (no hardware, no broker, no ONNX Runtime needed)

```bash
git clone https://github.com/EWC-Compute-Platform/IoTEdgeSense_PowerGrid_SDN.git
cd IoTEdgeSense_PowerGrid_SDN
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DIOT_STUB_HARDWARE=ON \
  -DIOT_ENABLE_ML=ON
cmake --build . --parallel
./bin/iotedgesense
```

Expected output:
```
[2026-xx-xx xx:xx:xx.xxx] [INFO ] [Logger] Initialized — level=INFO
[2026-xx-xx xx:xx:xx.xxx] [INFO ] [main] === IoT Edge Sensor Node v1.0.0 ===
[2026-xx-xx xx:xx:xx.xxx] [INFO ] [Watchdog] Started with 2 channel(s)
[2026-xx-xx xx:xx:xx.xxx] [INFO ] [Sensor] Created 'IMU_MPU6050' (id=0x01)
...
[MQTT STUB] PUB  devices/data  QoS=1  [{"ts":...,"id":1,...}]
```

### Build options

| CMake option | Default | Description |
|---|---|---|
| `IOT_STUB_HARDWARE` | `ON` | Hardware-free stub drivers for I²C/SPI/GPIO/UART |
| `IOT_USE_PAHO_MQTT` | `OFF` | Link against Eclipse Paho MQTT C++ |
| `IOT_ENABLE_ML` | `ON` | Include ML predictive maintenance module |
| `IOT_USE_ONNX_RUNTIME` | `OFF` | Link against ONNX Runtime (requires install) |
| `IOT_BUILD_TESTS` | `OFF` | Build GoogleTest unit tests |
| `IOT_ENABLE_ASAN` | `OFF` | AddressSanitizer (Debug builds only) |

### Build with real hardware + MQTT + ML inference

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DIOT_STUB_HARDWARE=OFF \
  -DIOT_USE_PAHO_MQTT=ON \
  -DIOT_ENABLE_ML=ON \
  -DIOT_USE_ONNX_RUNTIME=ON
cmake --build . --parallel
```

---

## ML Pipeline (Python)

### Setup

```bash
cd IoTEdgeSense_PowerGrid_SDN
pip install -r ml/requirements.txt
```

### Train the thermal ageing model

```bash
# Generate synthetic data + train + export to ONNX
python ml/models/train.py \
  --days 730 \
  --n-transformers 5 \
  --epochs 50 \
  --output-dir ml/data/checkpoints

# Export trained model to ONNX for C++ deployment
python ml/export/export_to_onnx.py \
  --checkpoint ml/data/checkpoints/best_model.pt \
  --output ml/data/checkpoints/transformer_thermal.onnx
```

### Train the anomaly detector

```bash
python ml/models/train_anomaly.py \
  --n-normal 20000 \
  --epochs 60 \
  --output-dir ml/data/checkpoints
```

### Run the prediction service

```bash
# Connects to MQTT broker, subscribes to devices/data,
# publishes alerts to devices/maintenance/{node_tag}
python ml/service/prediction_service.py \
  --broker mqtt.example.com \
  --port 8883 \
  --tls \
  --model ml/data/checkpoints/transformer_thermal.onnx \
  --scaler ml/data/checkpoints/feature_scaler.pkl

# Or via environment variables
export MQTT_BROKER=mqtt.example.com
export MQTT_PORT=8883
export MQTT_USE_TLS=true
export MODEL_PATH=ml/data/checkpoints/transformer_thermal.onnx
python ml/service/prediction_service.py
```

### ML pipeline end-to-end flow

```
[Synthetic / Real Data]
         │
ml/features/transformer_features.py
  IEC 60076-7 thermal model
  43-feature engineering
         │
ml/models/train.py
  TransformerThermalModel (PyTorch)
  Multi-task: RUL + ageing + fault_prob
         │
ml/export/export_to_onnx.py
  → transformer_thermal.onnx
  → feature_scaler.pkl
  → model_meta.json
         │
    ┌────┴────────────────────────┐
    │                             │
src/ml/onnx_inference.cpp    ml/service/prediction_service.py
  C++ edge inference           Python MQTT service
  (OnnxInferenceEngine)        (subscribes → infers → publishes)
    │                             │
    └────────┬────────────────────┘
             │
    devices/maintenance/{node}   MQTT topic
             │
    SDN Platform / Dashboard
```

---

## Configuration

All device, network, and power parameters are in `config.h`:

```cpp
// Device identity
constexpr char DEVICE_ID[]        = "IOT_EDGE_DEVICE_001";
constexpr char FIRMWARE_VERSION[] = "1.0.0";

// MQTT broker
constexpr char    MQTT_BROKER[] = "mqtt.example.com";
constexpr uint16_t MQTT_PORT   = 8883;   // 8883 = TLS

// Acquisition
constexpr uint32_t DEFAULT_SAMPLING_RATE_MS = 1000;

// Power management
constexpr bool     ENABLE_LOW_POWER_MODE = true;
constexpr uint32_t SLEEP_DURATION_MS     = 10000;
```

Secrets (MQTT credentials, WiFi password) are intentionally left empty and should be injected at runtime via environment variable, encrypted file, or TPM.

---

## MQTT Message Formats

### Sensor telemetry — `devices/data`
```json
[
  {"ts": 1714000000000, "id": 1, "unit": "V",   "valid": true, "values": [229.5, 230.1, 228.8]},
  {"ts": 1714000000000, "id": 2, "unit": "A",   "valid": true, "values": [598.2, 601.1, 597.8]},
  {"ts": 1714000000000, "id": 3, "unit": "Hz",  "valid": true, "values": [50.03, 0.001]}
]
```

### Grid fault event — `devices/maintenance/{node_tag}`
```json
{
  "ts": 1714000000000,
  "node": "BUS_A1",
  "nodeId": 1,
  "severity": "WARNING",
  "rul_days": 1247.3,
  "ageing_state": 0.0914,
  "fault_prob": 0.12,
  "fault_imminent": false,
  "hotspot_c": 78.4,
  "V": 1.34,
  "action": "Increase monitoring frequency. Review at next scheduled maintenance."
}
```

### Inbound command — `devices/commands`
```json
{"cmd": "set_rate", "params": {"rate_ms": 500}}
```

---

## Standards Alignment

| Standard | Status | Description |
|---|---|---|
| IEC 60038 | ✅ Implemented | Voltage nominal values and tolerances (threshold defaults) |
| EN 50160 | ✅ Implemented | Power quality characteristics — detection thresholds |
| IEC 60076-7 | ✅ Implemented | Loading guide for oil-immersed transformers — ML feature engineering |
| IEC 61850 | 🔲 Roadmap | Substation communication — GOOSE/MMS adapter planned |
| IEC 62351 | 🔲 Roadmap | Power system security — cybersecurity layer |
| IEC 61968/61970 | 🔲 Roadmap | Common Information Model — SDN platform integration |

---

## Project Roadmap

### ✅ Phase 1 — Foundation (complete)
- Abstract sensor base + I²C, SPI, GPIO, UART concrete drivers
- Four-stage data filter pipeline
- MQTT client with Paho C++ and stub backends
- Structured error handling, Logger, Watchdog
- CMake build system with hardware stub mode

### ✅ Phase 2 — Power Grid Bridge (complete)
- `GridNodeDescriptor`, `ThreePhaseMeasurement`, `GridFaultEvent` domain types
- `FaultDetector` — 20 fault types, 4 severity levels, IEC 60038/EN 50160 thresholds
- `PowerGridBridge` — sensor → domain translation, MQTT publisher
- Grid-specific sensor classes with defined `values[]` layout contracts

### ✅ Phase 3 — ML Predictive Maintenance (complete)
- IEC 60076-7 physics-informed feature extraction (43 features)
- `TransformerThermalModel` — hybrid PyTorch, multi-task (RUL + ageing + fault)
- `GridAutoencoder` — unsupervised anomaly detection (30 features, 6 anomaly types)
- ONNX export + C++ `OnnxInferenceEngine` + `PredictiveMaintenance` integration
- Python MQTT prediction service with scaler, threshold calibration, alert routing

### 🔲 Phase 4 — Load Forecasting (next)
- 15-minute-ahead feeder load forecasting (LSTM/TCN)
- Feeds SDN platform voltage regulation and DG dispatch decisions
- Historical load patterns + weather features + time-of-day context

### 🔲 Phase 5 — System Hardening
- GoogleTest unit test suite
- GitHub Actions CI pipeline (stub build on every push)
- OTA firmware update mechanism
- Persistent local storage queue (offline MQTT buffering)
- Yocto / Buildroot layer recipe

### 🔲 Phase 6 — Standards and Production
- IEC 61850 GOOSE/MMS adapter (interoperability with conventional substation IEDs)
- IEC 62351 security layer (authentication, encryption, role-based access)
- IEC 61968/61970 CIM adapter (SDN platform enterprise integration)
- Hardware validation on Raspberry Pi 4 with INA219 + ADS1115

---

## Adding a New Sensor

1. Create `src/sensors/my_sensor.h/.cpp` inheriting from `SensorBase`
2. Implement the six virtual methods
3. Add to `SENSOR_SOURCES` in `CMakeLists.txt`
4. Instantiate in `main.cpp` and add to the sensors vector
5. Optionally register with `PowerGridBridge` under the appropriate `SensorRole`

```cpp
// Example: TMP117 precision temperature sensor over I²C
class TMP117Sensor : public Sensors::I2CSensor {
public:
    TMP117Sensor(uint8_t id, uint8_t bus)
        : I2CSensor(id, "TMP117", bus, 0x48) {}

    SensorReading read() override {
        uint8_t hi = 0, lo = 0;
        readRegister(0x00, hi);
        readRegister(0x01, lo);
        int16_t raw = static_cast<int16_t>((hi << 8) | lo);
        SensorReading r;
        r.values   = { raw * 0.0078125f };   // 7.8125 m°C / LSB
        r.unit     = "degC";
        r.sensorId = mId;
        r.valid    = true;
        return r;
    }
};
```

---

## Error Handling

All subsystem errors route through `System::ErrorHandler` (singleton, thread-safe):

```cpp
// Register a callback at startup
System::ErrorHandler::getInstance().registerCallback(
    [](const System::ErrorRecord& r) {
        if (r.severity >= System::ErrorSeverity::ERROR)
            myLogger.write(r.source + ": " + r.message);
    },
    System::ErrorSeverity::WARNING,
    System::ErrorSubsystem::GRID    // filter to grid subsystem only
);
```

Error codes are grouped by subsystem: `0x01xx` sensors, `0x02xx` comm, `0x03xx` power, `0x04xx` data, `0x05xx` storage, `0x06xx` security, `0x07xx` system.

---

## Related Projects

- **EWC-Compute-Platform** — Engineering World Company's broader digital industrial platform
- **IoTEdgeSense-Original** — original flat-header-only prototype (reference archive)
- [Eclipse Paho MQTT C++](https://github.com/eclipse/paho.mqtt.cpp)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [IEC 60076-7 Loading Guide](https://webstore.iec.ch/publication/600)

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

```bash
# Fork and clone
git clone https://github.com/YOUR-FORK/IoTEdgeSense_PowerGrid_SDN.git

# Create a feature branch
git checkout -b feature/load-forecasting

# Build and test in stub mode
mkdir build && cd build
cmake .. -DIOT_STUB_HARDWARE=ON -DIOT_ENABLE_ML=ON -DIOT_BUILD_TESTS=ON
cmake --build . --parallel && ctest --output-on-failure

# Run Python tests
cd .. && python -m pytest ml/tests/ -v

# Commit and push
git commit -m "Add load forecasting model"
git push origin feature/load-forecasting
```

---

## License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

---

*IoT Edge Sense Power Grids SDN Networks Project — part of EWC Compute*
