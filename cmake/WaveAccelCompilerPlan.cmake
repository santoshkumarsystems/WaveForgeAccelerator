# WaveAccel compiler-plan integration
#
# Author: Santosh Kumar
# Copyright 2026 Santosh Kumar
# SPDX-License-Identifier: Apache-2.0
#
# This module promotes the previously validated compiler-plan bridge into the
# normal CMake/CTest build without changing the existing manual planner tests.

if(NOT TARGET waveaccel_runtime)
    message(FATAL_ERROR
        "WaveAccelCompilerPlan.cmake must be included after waveaccel_runtime "
        "has been created")
endif()

target_sources(
    waveaccel_runtime
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/execution_plan.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_plan_adapter.cpp"
)

if(NOT TARGET compiler_driven_runtime_smoke)
    add_executable(
        compiler_driven_runtime_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/compiler_driven_runtime_smoke.cpp"
    )

    target_link_libraries(
        compiler_driven_runtime_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        compiler_driven_runtime_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

enable_testing()

add_test(
    NAME compiler_driven_runtime_smoke
    COMMAND
        compiler_driven_runtime_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.weights"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_sample.bin"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    compiler_driven_runtime_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

message(STATUS "WaveAccel compiler-plan runtime integration: ON")

# Independent validation:
# compare the legacy/manual MemoryPlanner runtime Task sequence against the
# compiler-produced execution plan, then verify identical numerical output.
if(NOT TARGET compiler_manual_plan_parity)
    add_executable(
        compiler_manual_plan_parity
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/compiler_manual_plan_parity.cpp"
    )

    target_link_libraries(
        compiler_manual_plan_parity
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        compiler_manual_plan_parity
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME compiler_manual_plan_parity
    COMMAND
        compiler_manual_plan_parity
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.weights"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_sample.bin"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    compiler_manual_plan_parity
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Logical accelerator command lowering:
# CompiledRuntimeSchedule -> explicit Device->SRAM DMA + compute commands.
target_sources(
    waveaccel_runtime
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/device_command.cpp"
)

if(NOT TARGET device_command_stream_smoke)
    add_executable(
        device_command_stream_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/device_command_stream_smoke.cpp"
    )

    target_link_libraries(
        device_command_stream_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        device_command_stream_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME device_command_stream_smoke
    COMMAND
        device_command_stream_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    device_command_stream_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Direct backend execution:
# compiler-lowered DeviceCommandStream -> MockDevice without converting back
# into the older generic Task abstraction.
if(NOT TARGET mockdevice_device_command_smoke)
    add_executable(
        mockdevice_device_command_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/mockdevice_device_command_smoke.cpp"
    )

    target_link_libraries(
        mockdevice_device_command_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        mockdevice_device_command_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME mockdevice_device_command_smoke
    COMMAND
        mockdevice_device_command_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    mockdevice_device_command_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Compiler-driven Runtime cutover:
# ExecutionPlan -> DeviceCommandStream -> MockDevice directly, while the
# manual/reference MemoryPlanner -> Task[] path remains independently testable.
if(NOT TARGET compiler_runtime_direct_devicecommand_smoke)
    add_executable(
        compiler_runtime_direct_devicecommand_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/compiler_runtime_direct_devicecommand_smoke.cpp"
    )

    target_link_libraries(
        compiler_runtime_direct_devicecommand_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        compiler_runtime_direct_devicecommand_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME compiler_runtime_direct_devicecommand_smoke
    COMMAND
        compiler_runtime_direct_devicecommand_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.weights"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_sample.bin"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    compiler_runtime_direct_devicecommand_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Logical model-initializer device address space. The compiler-driven runtime
# lowers Device->SRAM staging commands to absolute offsets in this generic
# logical address space; these offsets are not physical hardware addresses.
target_sources(
    waveaccel_runtime
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/device_memory.cpp"
)

if(NOT TARGET device_memory_address_smoke)
    add_executable(
        device_memory_address_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/device_memory_address_smoke.cpp"
    )

    target_link_libraries(
        device_memory_address_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        device_memory_address_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME device_memory_address_smoke
    COMMAND
        device_memory_address_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.weights"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    device_memory_address_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


# MockDevice validates the compiler/runtime logical address contract.
if(NOT TARGET device_address_contract_smoke)
    add_executable(
        device_address_contract_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/device_address_contract_smoke.cpp"
    )

    target_link_libraries(
        device_address_contract_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        device_address_contract_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME device_address_contract_smoke
    COMMAND
        device_address_contract_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    device_address_contract_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


# Transport the compiler-side WaveAccel target contract into C++ and verify
# Runtime accepts supported V1 semantics while rejecting incompatible ones.
if(NOT TARGET target_contract_transport_smoke)
    add_executable(
        target_contract_transport_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/target_contract_transport_smoke.cpp"
    )

    target_link_libraries(
        target_contract_transport_smoke
        PRIVATE
            waveaccel_runtime
    )

    target_include_directories(
        target_contract_transport_smoke
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

add_test(
    NAME target_contract_transport_smoke
    COMMAND
        target_contract_transport_smoke
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory"
        "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/waveaccel_plan.manifest"
)

set_tests_properties(
    target_contract_transport_smoke
    PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
