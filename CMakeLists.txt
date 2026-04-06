cmake_minimum_required(VERSION 3.16)
project(IoTEdgeSense VERSION 1.0.0 LANGUAGES CXX)

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

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter   # common in stub implementations
)

if(IOT_ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ---------------------------------------------------------------------------
# Stub / real hardware defines
# ---------------------------------------------------------------------------
if(IOT_STUB_HARDWARE)
    message(STATUS "[IoTEdgeSense] Hardware stubs ENABLED — no real drivers needed")
    add_compile_definitions(I2C_STUB GPIO_STUB SPI_STUB UART_STUB)
else()
    message(STATUS "[IoTEdgeSense] Building with real hardware drivers")
endif()

if(IOT_USE_PAHO_MQTT)
    message(STATUS "[IoTEdgeSense] Paho MQTT C++ backend ENABLED")
    add_compile_definitions(MQTT_USE_PAHO)
else()
    message(STATUS "[IoTEdgeSense] MQTT stub backend (no broker required)")
endif()

# ---------------------------------------------------------------------------
# Source file lists
# ---------------------------------------------------------------------------
set(SYSTEM_SOURCES
    src/system/error_handler.cpp
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

set(GRID_SOURCES
    src/grid/fault_detector.cpp
    src/grid/power_grid_bridge.cpp
)

set(COMM_SOURCES
    src/comm/mqtt_client.cpp
    src/comm/comm_manager.cpp
)

# ---------------------------------------------------------------------------
# Static library — everything except main
# ---------------------------------------------------------------------------
add_library(iotedgesense_lib STATIC
    ${SYSTEM_SOURCES}
    ${SENSOR_SOURCES}
    ${PROCESSING_SOURCES}
    ${COMM_SOURCES}
    ${GRID_SOURCES}
)

target_include_directories(iotedgesense_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # exposes config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Threads required for std::mutex / std::thread
find_package(Threads REQUIRED)
target_link_libraries(iotedgesense_lib PUBLIC Threads::Threads)

# Optional: Paho MQTT C++ (install via: apt install libpaho-mqtt-dev libpaho-mqttpp-dev)
if(IOT_USE_PAHO_MQTT)
    find_package(PahoMqttCpp QUIET)
    if(PahoMqttCpp_FOUND)
        target_link_libraries(iotedgesense_lib PUBLIC PahoMqttCpp::paho-mqttpp3)
    else()
        # Fallback: manual link if find_package fails
        target_link_libraries(iotedgesense_lib PUBLIC paho-mqttpp3 paho-mqtt3as)
        message(WARNING "PahoMqttCpp not found via find_package — linking manually")
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
# Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS iotedgesense
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS iotedgesense_lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iotedgesense
    FILES_MATCHING PATTERN "*.h"
)

install(FILES config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iotedgesense
)

# ---------------------------------------------------------------------------
# Unit tests (optional)
# ---------------------------------------------------------------------------
if(IOT_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(test_data_filter    tests/unit/test_data_filter.cpp)
        add_executable(test_data_processor tests/unit/test_data_processor.cpp)
        add_executable(test_error_handler  tests/unit/test_error_handler.cpp)

        foreach(test_target
            test_data_filter
            test_data_processor
            test_error_handler)
            target_link_libraries(${test_target} PRIVATE
                iotedgesense_lib GTest::gtest_main)
            add_test(NAME ${test_target} COMMAND ${test_target})
        endforeach()
    else()
        message(WARNING "GoogleTest not found — skipping unit tests")
    endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== IoTEdgeSense Build Configuration ===")
message(STATUS "  Version       : ${PROJECT_VERSION}")
message(STATUS "  C++ Standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Stub hardware : ${IOT_STUB_HARDWARE}")
message(STATUS "  Paho MQTT     : ${IOT_USE_PAHO_MQTT}")
message(STATUS "  Build tests   : ${IOT_BUILD_TESTS}")
message(STATUS "  ASan          : ${IOT_ENABLE_ASAN}")
message(STATUS "=========================================")
message(STATUS "")

