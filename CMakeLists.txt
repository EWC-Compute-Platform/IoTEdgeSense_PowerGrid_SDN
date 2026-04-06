cmake_minimum_required(VERSION 3.14)
project(IoTEdgeSense VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(include)

# Gather all source files (exclude main.cpp stubs)
set(FIRMWARE_SOURCES
    src/sensors/sensor_base.cpp
    src/sensors/i2c_sensor.cpp
    src/sensors/spi_sensor.cpp
    src/sensors/uart_sensor.cpp
    src/sensors/gpio_sensor.cpp
    src/communication/mqtt_client.cpp
    src/communication/coap_client.cpp
    src/communication/comm_manager.cpp
    src/data/data_processor.cpp
    src/data/data_filter.cpp
    src/power/power_manager.cpp
    src/system/error_handler.cpp
    src/system/logger.cpp
    src/system/watchdog.cpp
)

# Main simulator executable
add_executable(iot_firmware_simulator
    ${FIRMWARE_SOURCES}
    src/main_simple.cpp
)

target_include_directories(iot_firmware_simulator PRIVATE include)

# Compiler warnings
target_compile_options(iot_firmware_simulator PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Install target
install(TARGETS iot_firmware_simulator DESTINATION bin)
