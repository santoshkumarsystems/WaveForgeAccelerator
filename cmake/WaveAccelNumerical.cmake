# WaveAccel - AI Accelerator Runtime Prototype
#
# Numerical execution targets.
#
# Author: Santosh Kumar
# Copyright 2026 Santosh Kumar
# SPDX-License-Identifier: Apache-2.0

target_sources(
    waveaccel_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_store.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/numerical_executor.cpp
)

add_executable(
    waveaccel_numerical_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/numerical_demo.cpp
)

target_link_libraries(
    waveaccel_numerical_demo
    PRIVATE
        waveaccel_runtime
)

add_executable(
    numerical_parity_smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/numerical_parity_smoke.cpp
)

target_link_libraries(
    numerical_parity_smoke
    PRIVATE
        waveaccel_runtime
)

add_executable(
    runtime_numerical_integration_smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_numerical_integration_smoke.cpp
)

target_link_libraries(
    runtime_numerical_integration_smoke
    PRIVATE
        waveaccel_runtime
)

set(
    WAVEACCEL_NUMERICAL_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
)

set(
    WAVEACCEL_NUMERICAL_WEIGHTS
    "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.weights"
)

set(
    WAVEACCEL_NUMERICAL_SAMPLE
    "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_sample.bin"
)

if(
    EXISTS "${WAVEACCEL_NUMERICAL_MANIFEST}"
    AND EXISTS "${WAVEACCEL_NUMERICAL_WEIGHTS}"
    AND EXISTS "${WAVEACCEL_NUMERICAL_SAMPLE}"
)
    add_test(
        NAME numerical_parity_smoke
        COMMAND
            numerical_parity_smoke
            "${WAVEACCEL_NUMERICAL_MANIFEST}"
            "${WAVEACCEL_NUMERICAL_WEIGHTS}"
            "${WAVEACCEL_NUMERICAL_SAMPLE}"
    )

    add_test(
        NAME runtime_numerical_integration_smoke
        COMMAND
            runtime_numerical_integration_smoke
            "${WAVEACCEL_NUMERICAL_MANIFEST}"
            "${WAVEACCEL_NUMERICAL_WEIGHTS}"
            "${WAVEACCEL_NUMERICAL_SAMPLE}"
    )
else()
    message(
        STATUS
        "WaveAccel numerical fixture not found; numerical CTest cases "
        "will be built but not registered."
    )
endif()
