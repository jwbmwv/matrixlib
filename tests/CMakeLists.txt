# Tests

# Placeholder for unit tests
# You can add Google Test, Catch2, or custom test framework here

# Example with custom test:
# add_executable(test_vec test_vec.cpp)
# target_link_libraries(test_vec PRIVATE matrixlib)
# add_test(NAME test_vec COMMAND test_vec)

# Example with Google Test:
# find_package(GTest REQUIRED)
# add_executable(test_matrixlib test_vec.cpp test_mat.cpp test_quat.cpp)
# target_link_libraries(test_matrixlib PRIVATE matrixlib GTest::GTest GTest::Main)
# gtest_discover_tests(test_matrixlib)

message(STATUS "MatrixLib: Tests directory configured (add test files here)")
