/**
 * @file main.cpp
 * @brief IoTEdgeSense firmware entry point
 *
 * Initialises all subsystems, runs the main acquisition-process-publish
 * loop, and handles graceful shutdown on SIGINT / SIGTERM.
 *
 * Build flags of interest:
 *   -DI2C_STUB   -DGPIO_STUB   -DSPI_STUB   -DUART_STUB
 *       Compile without real hardware drivers (host testing)
 *   -DMQTT_USE_PAHO
 *       Link against Paho MQTT C++ for a real broker connection
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <sstream>

// Subsystems
#include "config.h"
#include "src/system/error_handler.h"
#include "src/sensors/i2c_sensor.h"
#include "src/sensors/gpio_sensor.h"
#include "src/sensors/spi_sensor.h"
#include "src/sensors/uart_sensor.h"
#include "src/processing/data_processor.h"
#include "src/processing/data_filter.h"
#include "src/comm/comm_manager.h"

// ---------------------------------------------------------------------------
// Global shutdown flag — set by signal handler
// ---------------------------------------------------------------------------
static std::atomic<bool> gShutdown{false};

static void signalHandler(int sig) {
    std::cout << "\n[main] Caught signal " << sig
              << " — initiating graceful shutdown...\n";
    gShutdown = true;
}

// ---------------------------------------------------------------------------
// Helper: build a JSON status heartbeat
// ---------------------------------------------------------------------------
static std::string buildStatusJson(bool sensorsOk, bool commOk,
                                   size_t readingCount) {
    std::ostringstream oss;
    oss << "{"
        << "\"device\":\"" << DeviceConfig::DEVICE_ID << "\","
        << "\"fw\":\""     << DeviceConfig::FIRMWARE_VERSION << "\","
        << "\"sensors_ok\":" << (sensorsOk ? "true" : "false") << ","
        << "\"comm_ok\":"    << (commOk    ? "true" : "false") << ","
        << "\"readings\":"   << readingCount
        << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // ── Signal handling ────────────────────────────────────────────────────
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "===================================================\n"
              << "  " << DeviceConfig::DEVICE_NAME
              << "  v" << DeviceConfig::FIRMWARE_VERSION << "\n"
              << "===================================================\n";

    // ── Error handler setup ────────────────────────────────────────────────
    auto& errHandler = System::ErrorHandler::getInstance();
    errHandler.registerCallback([](const System::ErrorEvent& ev) {
        if (ev.severity >= System::ErrorSeverity::WARNING) {
            std::cerr << "[ERR][" << ev.source << "] "
                      << System::ErrorHandler::errorCodeToString(ev.code)
                      << ": " << ev.message << "\n";
        }
    });

    // ── Sensor initialisation ──────────────────────────────────────────────
    std::vector<std::shared_ptr<Sensors::SensorBase>> sensors;

    // I2C — e.g. MPU-6050 IMU on bus 1, address 0x68
    auto i2cSensor = std::make_shared<Sensors::I2CSensor>(
        0x01, "IMU_MPU6050", /*bus=*/1, /*addr=*/0x68
    );

    // GPIO — digital input, e.g. door / tamper sensor on pin 17
    auto gpioSensor = std::make_shared<Sensors::GPIOSensor>(
        0x02, "TamperSwitch", /*pin=*/17, Sensors::GPIODirection::INPUT
    );

    // SPI — e.g. MAX31855 thermocouple on bus 0, CS 0
    auto spiSensor = std::make_shared<Sensors::SPISensor>(
        0x03, "Thermocouple_MAX31855", /*bus=*/0, /*cs=*/0,
        Sensors::SPIMode::MODE_0, /*speedHz=*/5000000
    );

    // UART — e.g. GPS NMEA module on /dev/ttyUSB0 @ 9600
    auto uartSensor = std::make_shared<Sensors::UARTSensor>(
        0x04, "GPS_UART", "/dev/ttyUSB0", 9600
    );

    sensors = {i2cSensor, gpioSensor, spiSensor, uartSensor};

    bool sensorsOk = true;
    for (auto& s : sensors) {
        if (!s->initialize()) {
            std::cerr << "[main] WARNING: sensor '" << s->getName()
                      << "' failed to initialise — continuing without it.\n";
            sensorsOk = false;
        } else {
            std::cout << "[main] Sensor '" << s->getName() << "' ready.\n";
        }
    }

    // ── Data processing pipeline ───────────────────────────────────────────
    Data::DataProcessor processor;
    processor.initialize();

    // Moving average (5-sample window) to smooth noise
    processor.addFilter(std::make_shared<Data::MovingAverageFilter>("maf1", 5));

    // Threshold guard: reject physically impossible readings
    processor.addFilter(std::make_shared<Data::ThresholdFilter>(
        "thresh1", -1000.0f, 1000.0f
    ));

    // Delta filter: only forward readings that changed by ≥ 0.01
    processor.addFilter(std::make_shared<Data::DeltaFilter>("delta1", 0.01f));

    // ── Communication setup ────────────────────────────────────────────────
    Communication::CommManager comm(
        DeviceConfig::MQTT_CLIENT_ID,
        DeviceConfig::MQTT_BROKER,
        DeviceConfig::MQTT_PORT,
        DeviceConfig::MQTT_USERNAME,
        DeviceConfig::MQTT_PASSWORD,
        DeviceConfig::ENABLE_TLS
    );

    if (DeviceConfig::ENABLE_TLS) {
        comm.setTLSCertificates(
            DeviceConfig::TLS_CA_CERT_PATH,
            DeviceConfig::TLS_CLIENT_CERT_PATH,
            DeviceConfig::TLS_CLIENT_KEY_PATH
        );
    }

    // Register a handler for inbound "set_rate" commands
    comm.registerCommandHandler("set_rate",
        [](const std::string& cmd, const std::string& payload) {
            std::cout << "[main] Command received: " << cmd
                      << "  payload: " << payload << "\n";
            // TODO: parse rate_ms from payload and update sensor sampling rates
        }
    );

    bool commOk = comm.initialize();
    if (!commOk) {
        std::cerr << "[main] WARNING: communication init failed — "
                     "running in offline/log-only mode.\n";
    } else {
        std::cout << "[main] MQTT connected to "
                  << DeviceConfig::MQTT_BROKER << ":"
                  << DeviceConfig::MQTT_PORT << "\n";
    }

    // ── Main acquisition loop ──────────────────────────────────────────────
    size_t totalReadings  = 0;
    size_t loopCount      = 0;
    auto   lastHeartbeat  = std::chrono::steady_clock::now();

    const auto samplingInterval =
        std::chrono::milliseconds(DeviceConfig::DEFAULT_SAMPLING_RATE_MS);
    const auto heartbeatInterval = std::chrono::seconds(30);

    std::cout << "[main] Entering acquisition loop "
              << "(sampling interval: "
              << DeviceConfig::DEFAULT_SAMPLING_RATE_MS << " ms).\n";

    while (!gShutdown) {
        auto loopStart = std::chrono::steady_clock::now();

        // ── 1. Collect readings from all valid sensors ─────────────────────
        std::vector<Sensors::SensorReading> rawReadings;
        for (auto& sensor : sensors) {
            if (!sensor->isValid()) continue;
            auto reading = sensor->read();
            if (reading.valid) {
                rawReadings.push_back(reading);
            }
        }

        // ── 2. Process through the filter pipeline ─────────────────────────
        if (!rawReadings.empty()) {
            auto result = processor.process(rawReadings);
            if (result.success && !result.readings.empty()) {
                totalReadings += result.readings.size();

                // ── 3. Publish (batch if more than one reading) ────────────
                if (commOk && comm.isConnected()) {
                    if (result.readings.size() == 1) {
                        comm.publishTelemetry(result.readings.front());
                    } else {
                        comm.publishBatch(result.readings);
                    }
                }

                // Anomaly check every 10 loops
                if (loopCount % 10 == 0 && result.readings.size() >= 3) {
                    auto anomalies = processor.detectAnomalies(result.readings);
                    if (!anomalies.empty()) {
                        std::cout << "[main] ⚠  " << anomalies.size()
                                  << " anomalous reading(s) detected.\n";
                    }
                }
            }
        }

        // ── 4. Periodic heartbeat ──────────────────────────────────────────
        auto now = std::chrono::steady_clock::now();
        if (now - lastHeartbeat >= heartbeatInterval) {
            std::string status = buildStatusJson(sensorsOk, commOk, totalReadings);
            if (commOk && comm.isConnected()) comm.publishStatus(status);
            std::cout << "[main] Heartbeat: " << status << "\n";
            lastHeartbeat = now;
        }

        ++loopCount;

        // ── 5. Sleep until next sample ─────────────────────────────────────
        auto elapsed = std::chrono::steady_clock::now() - loopStart;
        if (elapsed < samplingInterval) {
            std::this_thread::sleep_for(samplingInterval - elapsed);
        }
    }

    // ── Graceful shutdown ──────────────────────────────────────────────────
    std::cout << "[main] Shutting down...\n";

    // Put sensors to sleep
    for (auto& s : sensors) {
        if (s->isValid()) s->sleep();
    }

    // Publish final offline status
    if (commOk && comm.isConnected()) {
        comm.publishStatus("{\"status\":\"offline\",\"device\":\""
                           + std::string(DeviceConfig::DEVICE_ID) + "\"}");
    }

    comm.shutdown();

    std::cout << "[main] Shutdown complete. "
              << totalReadings << " readings processed in "
              << loopCount << " loop iterations.\n";
    return 0;
}

// ============================================================================
// OPTIONAL: demo_grid_bridge()
// Demonstrates the PowerGridBridge with simulated grid sensors.
// Call from main() in place of the generic loop when testing grid mode.
// ============================================================================
#include "src/grid/grid_types.h"
#include "src/grid/grid_config.h"
#include "src/grid/grid_sensors.h"
#include "src/grid/fault_detector.h"
#include "src/grid/power_grid_bridge.h"

void demo_grid_bridge() {
    std::cout << "\n=== Power Grid Bridge Demo ===\n";

    // ── Shared comm manager (stub mode) ─────────────────────────────────
    auto comm = std::make_shared<Communication::CommManager>(
        "GRID_EDGE_001", "mqtt.example.com", 1883);
    comm->initialize();

    // ── Configure bridge (50 Hz European grid) ───────────────────────────
    Grid::PowerGridBridge bridge(comm, Grid::GridConfig::european50Hz());

    // ── Register grid node ────────────────────────────────────────────────
    Grid::GridNodeDescriptor busA1;
    busA1.nodeId              = 0x10;
    busA1.nodeTag             = "BUS_A1";
    busA1.type                = Grid::NodeType::DISTRIBUTION_BUS;
    busA1.voltageLevel        = Grid::VoltageLevel::LOW;
    busA1.nominalVoltage_V    = 230.0f;
    busA1.nominalFrequency_Hz = 50.0f;
    busA1.ratedCurrent_A      = 200.0f;
    busA1.sdnNodeId           = "sdn-node-BUS_A1";
    bridge.registerNode(busA1);

    // ── Register sensors ──────────────────────────────────────────────────
    bridge.registerSensor(0x20, Grid::SensorRole::VOLTAGE,      0x10);
    bridge.registerSensor(0x21, Grid::SensorRole::CURRENT,      0x10);
    bridge.registerSensor(0x22, Grid::SensorRole::FREQUENCY,    0x10);
    bridge.registerSensor(0x23, Grid::SensorRole::POWER_METER,  0x10);
    bridge.registerSensor(0x24, Grid::SensorRole::THERMAL,      0x10);

    // ── Register callbacks ────────────────────────────────────────────────
    bridge.onMeasurement([](const Grid::ThreePhaseMeasurement& m) {
        std::cout << "[GRID] " << m.nodeTag
                  << "  Va=" << m.voltageA_V << " V"
                  << "  Ia=" << m.currentA_A << " A"
                  << "  Hz=" << m.frequency_Hz
                  << "  PF=" << m.powerFactor << "\n";
    });

    bridge.onFault([](const Grid::GridFaultEvent& ev) {
        std::cout << "[FAULT] " << ev.nodeTag
                  << "  " << ev.description
                  << "  severity=" << static_cast<int>(ev.severity) << "\n";
    });

    // ── Instantiate grid sensors ──────────────────────────────────────────
    Grid::VoltageSensor   vs (0x20, "BUS_A1", 230.0f);
    Grid::CurrentSensor   cs (0x21, "BUS_A1", 200.0f);
    Grid::FrequencySensor fs (0x22, "BUS_A1", 50.0f);
    Grid::PowerMeter      pm (0x23, "BUS_A1", 230.0f, 200.0f, 0.92f);
    Grid::ThermalSensor   ts (0x24, "BUS_A1", 55.0f);

    for (auto* s : std::initializer_list<Sensors::SensorBase*>{
            &vs, &cs, &fs, &pm, &ts}) {
        s->initialize();
    }

    // ── Simulate 5 acquisition cycles ────────────────────────────────────
    for (int cycle = 0; cycle < 5 && !gShutdown; ++cycle) {
        std::cout << "\n--- Cycle " << (cycle + 1) << " ---\n";

        bridge.ingestBatch({
            vs.read(), cs.read(), fs.read(), pm.read(), ts.read()
        });

        // Inject a fault scenario on cycle 3 (over-voltage)
        if (cycle == 2) {
            std::cout << "[TEST] Injecting over-voltage scenario...\n";
            Sensors::SensorReading ovr;
            ovr.sensorId  = 0x20;
            ovr.unit      = "V";
            ovr.valid     = true;
            ovr.timestamp = Grid::nowMs();
            // +15% over nominal → ALARM
            ovr.values = {264.5f, 265.0f, 263.8f, 457.9f, 458.8f, 456.7f};
            bridge.ingest(ovr);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // ── Print summary ─────────────────────────────────────────────────────
    auto faults = bridge.getActiveFaults();
    std::cout << "\nActive faults: " << faults.size() << "\n";
    for (const auto& f : faults) {
        std::cout << "  [" << f.nodeTag << "] " << f.description << "\n";
    }

    auto latest = bridge.getLatestMeasurement(0x10);
    std::cout << "Latest measurement valid: " << (latest.valid ? "YES" : "NO") << "\n";
    std::cout << "=== Grid Bridge Demo Complete ===\n\n";
}
