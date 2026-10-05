cmake_minimum_required(VERSION 3.20)

project(WaveAccel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Upstream dependency: cuWaves
#
# Public/default mode:
#   Fetch the reproducible cuWaves v0.0.1 release from GitHub.
#
# Local development mode:
#   cmake -S . -B build \
#     -DWAVEACCEL_CUWAVES_SOURCE=../cuwaves
# ---------------------------------------------------------------------------

include(FetchContent)

set(
    WAVEACCEL_CUWAVES_SOURCE
    ""
    CACHE PATH
    "Optional local cuWaves source checkout"
)

# WaveAccel V1 requires only the validated cuWaves CPU reference API.
set(CUWAVES_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(CUWAVES_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUWAVES_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(CUWAVES_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

if(WAVEACCEL_CUWAVES_SOURCE)
    get_filename_component(
        WAVEACCEL_CUWAVES_SOURCE_ABS
        "${WAVEACCEL_CUWAVES_SOURCE}"
        ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    message(
        STATUS
        "WaveAccel using local cuWaves: "
        "${WAVEACCEL_CUWAVES_SOURCE_ABS}"
    )

    add_subdirectory(
        "${WAVEACCEL_CUWAVES_SOURCE_ABS}"
        "${CMAKE_BINARY_DIR}/_deps/cuwaves-build"
    )
else()
    message(
        STATUS
        "WaveAccel fetching cuWaves v0.0.1 from GitHub"
    )

    FetchContent_Declare(
        cuwaves
        GIT_REPOSITORY
            https://github.com/santoshkumarsystems/cuwaves.git
        GIT_TAG
            v0.0.1
        GIT_SHALLOW
            TRUE
    )

    FetchContent_MakeAvailable(cuwaves)
endif()

# ---------------------------------------------------------------------------
# WaveAccel mock accelerator runtime
# ---------------------------------------------------------------------------

add_library(waveaccel_runtime
    src/device.cpp
    src/model_memory.cpp
    src/memory_planner.cpp
    src/runtime.cpp
)

target_include_directories(waveaccel_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(
        waveaccel_runtime
        PRIVATE /W4 /WX
    )
else()
    target_compile_options(
        waveaccel_runtime
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()

add_executable(waveaccel_demo
    src/main.cpp
)

target_link_libraries(
    waveaccel_demo
    PRIVATE
        waveaccel_runtime
)

# ---------------------------------------------------------------------------
# cuWaves -> WaveAccel dataset bridge
# ---------------------------------------------------------------------------

add_executable(generate_cuwaves_dataset
    tools/generate_cuwaves_dataset.cpp
)

target_link_libraries(
    generate_cuwaves_dataset
    PRIVATE
        cuWaves::cpu
)

if(MSVC)
    target_compile_options(
        generate_cuwaves_dataset
        PRIVATE /W4 /WX
    )
else()
    target_compile_options(
        generate_cuwaves_dataset
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

add_executable(runtime_smoke
    tests/runtime_smoke.cpp
)

target_link_libraries(
    runtime_smoke
    PRIVATE
        waveaccel_runtime
)

add_test(
    NAME runtime_smoke
    COMMAND runtime_smoke
)

# ---------------------------------------------------------------------------
# ONNX-derived model memory metadata smoke test
# ---------------------------------------------------------------------------

add_executable(model_memory_smoke
    tests/model_memory_smoke.cpp
)

target_link_libraries(
    model_memory_smoke
    PRIVATE
        waveaccel_runtime
)

add_test(
    NAME model_memory_smoke
    COMMAND
        model_memory_smoke
        ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory
)

# ---------------------------------------------------------------------------
# Accelerator SRAM memory-planning smoke test
# ---------------------------------------------------------------------------

add_executable(memory_planner_smoke
    tests/memory_planner_smoke.cpp
)

target_link_libraries(
    memory_planner_smoke
    PRIVATE
        waveaccel_runtime
)

add_test(
    NAME memory_planner_smoke
    COMMAND
        memory_planner_smoke
        ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/wave_model.memory
)

# ---------------------------------------------------------------------------
# Real float32 numerical execution using trained ONNX initializer values
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/WaveAccelNumerical.cmake)

# Compiler-produced execution-plan -> C++ runtime integration.
include(cmake/WaveAccelCompilerPlan.cmake)
