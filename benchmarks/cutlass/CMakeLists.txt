cmake_minimum_required(VERSION 3.23)

project(CutlassLaunchProbe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CUTLASS_ROOT OR CUTLASS_ROOT STREQUAL "")
  message(FATAL_ERROR "CUTLASS_ROOT_UNAVAILABLE: pass -DCUTLASS_ROOT=/home/zm/project/cutlass")
endif()

if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS_ROOT_UNAVAILABLE: ${CUTLASS_ROOT}/include/cutlass/cutlass.h not found")
endif()

if(NOT EXISTS "${CUTLASS_ROOT}/tools/util/include")
  message(FATAL_ERROR "CUTLASS_ROOT_UNAVAILABLE: ${CUTLASS_ROOT}/tools/util/include not found")
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120-real;120-virtual")
  message(FATAL_ERROR "This CUTLASS probe only supports CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual")
endif()

find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND git -C "${CUTLASS_ROOT}" rev-parse --short HEAD
  OUTPUT_VARIABLE CUTLASS_GIT_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(CUTLASS_GIT_REVISION STREQUAL "")
  set(CUTLASS_GIT_REVISION "unknown")
endif()

function(configure_uxsched_cutlass_target target)
  target_include_directories(${target} PRIVATE
    "${CUTLASS_ROOT}/include"
    "${CUTLASS_ROOT}/tools/util/include"
  )

  target_compile_definitions(${target} PRIVATE
    CUTLASS_GIT_REVISION="${CUTLASS_GIT_REVISION}"
    UXSCHED_CUTLASS_NATIVE_SM120=1
    CUTLASS_ENABLE_CUDA_HOST_ADAPTER=0
  )

  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=none>
  )

  target_link_libraries(${target} PRIVATE CUDA::cudart)

  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "120-real;120-virtual"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endfunction()

add_executable(cutlass_launch_probe
  cutlass_launch_probe.cu
)
configure_uxsched_cutlass_target(cutlass_launch_probe)

add_executable(cutlass_realtime_worker
  cutlass_realtime_worker.cu
)
configure_uxsched_cutlass_target(cutlass_realtime_worker)
