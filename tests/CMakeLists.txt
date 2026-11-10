add_executable(tests test_main.cpp test_core.cpp test_dynamics.cpp)
target_link_libraries(tests PRIVATE lib)

add_test(NAME ionmd_unit_tests COMMAND tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
