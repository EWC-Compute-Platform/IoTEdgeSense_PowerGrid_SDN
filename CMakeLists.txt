cmake_minimum_required(VERSION 3.16)
project(IoTEdgeSense_PowerGrid_SDN VERSION 1.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(IOT_STUB_HARDWARE
    "Compile with hardware stub drivers (no real I2C/GPIO/SPI/UART needed)"
    ON)
option(IOT_USE_PAHO_MQTT
    "Link against Eclipse Paho MQTT C++ for a real broker connection"
    OFF)
option(IOT_BUILD_TESTS
    "Build unit tests"
    OFF)
option(IOT_ENABLE_ASAN
    "Enable AddressSanitizer (debug builds only)"
    OFF)
option(IOT_ENABLE_ML
    "Include ML predictive maintenance module"
    ON)
option(IOT_USE_ONNX_RUNTIME
    "Link against ONNX Runtime for real ML inference (requires libonnxruntime)"
    OFF)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
)

if(IOT_ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ---------------------------------------------------------------------------
# Hardware / MQTT / ML stub defines
# ---------------------------------------------------------------------------
if(IOT_STUB_HARDWARE)
    message(STATUS "[IoTEdgeSense] Hardware stubs ENABLED")
    add_compile_definitions(I2C_STUB GPIO_STUB SPI_STUB UART_STUB)
else()
    message(STATUS "[IoTEdgeSense] Building with real hardware drivers")
endif()

add_compile_definitions(WD_HW_STUB)   # always stub on build machines; remove for RPi

if(IOT_USE_PAHO_MQTT)
    message(STATUS "[IoTEdgeSense] Paho MQTT C++ backend ENABLED")
    add_compile_definitions(MQTT_USE_PAHO)
else()
    message(STATUS "[IoTEdgeSense] MQTT stub backend")
endif()

if(IOT_ENABLE_ML)
    message(STATUS "[IoTEdgeSense] ML predictive maintenance module ENABLED")
    if(IOT_USE_ONNX_RUNTIME)
        message(STATUS "[IoTEdgeSense] Using real ONNX Runtime for inference")
    else()
        message(STATUS "[IoTEdgeSense] ONNX stub mode (no runtime needed)")
        add_compile_definitions(ONNX_STUB)
    endif()
else()
    message(STATUS "[IoTEdgeSense] ML module DISABLED")
    add_compile_definitions(ONNX_STUB)
endif()

# ---------------------------------------------------------------------------
# Source file lists
# ---------------------------------------------------------------------------
set(SYSTEM_SOURCES
    src/system/error_handler.cpp
    src/system/logger.cpp
    src/system/watchdog.cpp
)

set(SENSOR_SOURCES
    src/sensors/sensor_base.cpp
    src/sensors/i2c_sensor.cpp
    src/sensors/gpio_sensor.cpp
    src/sensors/spi_sensor.cpp
    src/sensors/uart_sensor.cpp
)

set(PROCESSING_SOURCES
    src/processing/data_filter.cpp
    src/processing/data_processor.cpp
)

set(COMM_SOURCES
    src/comm/mqtt_client.cpp
    src/comm/comm_manager.cpp
)

set(GRID_SOURCES
    src/Grid/fault_detector.cpp
    src/Grid/power_grid_bridge.cpp
)

# ML sources — feature_extractor must come before predictive_maintenance
# (predictive_maintenance depends on feature_extractor)
set(ML_SOURCES "")
if(IOT_ENABLE_ML)
    list(APPEND ML_SOURCES
        src/ml/feature_extractor.cpp
        src/ml/onnx_inference.cpp
        src/ml/predictive_maintenance.cpp
    )
endif()

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
add_library(iotedgesense_lib STATIC
    ${SYSTEM_SOURCES}
    ${SENSOR_SOURCES}
    ${PROCESSING_SOURCES}
    ${COMM_SOURCES}
    ${GRID_SOURCES}
    ${ML_SOURCES}
)

target_include_directories(iotedgesense_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(Threads REQUIRED)
target_link_libraries(iotedgesense_lib PUBLIC Threads::Threads)

# Optional: Paho MQTT C++
if(IOT_USE_PAHO_MQTT)
    find_package(PahoMqttCpp QUIET)
    if(PahoMqttCpp_FOUND)
        target_link_libraries(iotedgesense_lib PUBLIC PahoMqttCpp::paho-mqttpp3)
    else()
        target_link_libraries(iotedgesense_lib PUBLIC paho-mqttpp3 paho-mqtt3as)
        message(WARNING "PahoMqttCpp not found via find_package — linking manually")
    endif()
endif()

# Optional: ONNX Runtime
if(IOT_USE_ONNX_RUNTIME)
    # Install: apt install libonnxruntime-dev
    # Or download from https://github.com/microsoft/onnxruntime/releases
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime/core/session/onnxruntime_cxx_api.h
        PATHS /usr/local/include /usr/include
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS /usr/local/lib /usr/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        target_include_directories(iotedgesense_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(iotedgesense_lib PUBLIC ${ONNXRUNTIME_LIB})
        message(STATUS "[IoTEdgeSense] ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    else()
        message(FATAL_ERROR
            "ONNX Runtime not found. Install libonnxruntime-dev or set "
            "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIB manually.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Firmware executable
# ---------------------------------------------------------------------------
add_executable(iotedgesense main.cpp)
target_link_libraries(iotedgesense PRIVATE iotedgesense_lib)
set_target_properties(iotedgesense PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ---------------------------------------------------------------------------
# Unit tests (optional)
# ---------------------------------------------------------------------------
if(IOT_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        foreach(test_target
            test_data_filter
            test_data_processor
            test_error_handler
            test_thermal_model
            test_feature_extractor
            test_predictive_maintenance)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/${test_target}.cpp")
                add_executable(${test_target}
                    tests/unit/${test_target}.cpp)
                target_link_libraries(${test_target} PRIVATE
                    iotedgesense_lib GTest::gtest_main)
                add_test(NAME ${test_target} COMMAND ${test_target})
            endif()
        endforeach()
    else()
        message(WARNING "GoogleTest not found — skipping unit tests")
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS iotedgesense
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS iotedgesense_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== IoTEdgeSense PowerGrid SDN Build Configuration ===")
message(STATUS "  Version         : ${PROJECT_VERSION}")
message(STATUS "  C++ Standard    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Stub hardware   : ${IOT_STUB_HARDWARE}")
message(STATUS "  Paho MQTT       : ${IOT_USE_PAHO_MQTT}")
message(STATUS "  ML module       : ${IOT_ENABLE_ML}")
message(STATUS "  ONNX Runtime    : ${IOT_USE_ONNX_RUNTIME}")
message(STATUS "  Build tests     : ${IOT_BUILD_TESTS}")
message(STATUS "  ASan            : ${IOT_ENABLE_ASAN}")
message(STATUS "======================================================")
message(STATUS "")
