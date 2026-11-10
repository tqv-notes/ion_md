set(IONMD_SOURCES
  config.cpp
  coulomb_cpu.cpp
  data.cpp
  simulation.cpp
)

if(IONMD_USE_CUDA)
  list(APPEND IONMD_SOURCES coulomb_cuda.cu)
endif()

add_library(lib STATIC ${IONMD_SOURCES})
set_target_properties(lib PROPERTIES OUTPUT_NAME ionmd)

target_include_directories(lib PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(lib PUBLIC Threads::Threads)

if(IONMD_USE_OPENMP)
  target_link_libraries(lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(IONMD_USE_CUDA)
  target_compile_definitions(lib PUBLIC IONMD_HAVE_CUDA)
  target_link_libraries(lib PUBLIC CUDA::cudart)
  set_target_properties(lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_compile_options(lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Wno-deprecated-gpu-targets>)
endif()

# The bundled nlohmann::json 2.1.1 triggers a false-positive -Wuninitialized
# in GCC >= 13. It is reported inside <bits/move.h>, out of reach of the
# pragmas in ionmd/json.hpp, so silence it for the one file that parses JSON.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(config.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-uninitialized;-Wno-maybe-uninitialized")
endif()

# std::filesystem needs an extra library with old GCC (< 9)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(lib PUBLIC stdc++fs)
endif()
