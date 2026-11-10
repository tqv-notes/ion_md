add_executable(ionmd ionmd_main.cpp)
target_link_libraries(ionmd PRIVATE lib)
install(TARGETS ionmd RUNTIME DESTINATION bin)
