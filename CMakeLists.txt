cmake_minimum_required(VERSION 3.18)

project(IonMD VERSION 2.0 LANGUAGES CXX)

option(IONMD_USE_OPENMP "Enable OpenMP multithreading" ON)
option(IONMD_USE_CUDA "Build the CUDA (GPU) Coulomb backend" OFF)
option(BUILD_TESTS "Build C++ tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()

# Optional: -DIONMD_NATIVE=ON adds -march=native (build on the compute node!)
option(IONMD_NATIVE "Optimise for the build machine's CPU (-march=native)" OFF)
if(IONMD_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(IONMD_USE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(IONMD_USE_CUDA)
  # Tesla V100 = sm_70. Override with -DCMAKE_CUDA_ARCHITECTURES=...
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "IonMD: CUDA backend enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")
endif()

add_subdirectory(lib)
add_subdirectory(app)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
