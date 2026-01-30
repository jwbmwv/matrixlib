# Tests

# Find Google Test
find_package(GTest QUIET)

if(GTest_FOUND)
    include(GoogleTest)
    message(STATUS "MatrixLib: Building Google Test suite")

    # Create test executable
    add_executable(matrixlib_gtests
        google/test_constants.cpp
        google/test_vec.cpp
        google/test_mat.cpp
        google/test_quat.cpp
        google/test_edge_cases.cpp
        google/test_constexpr.cpp
    )

    # Link libraries
    target_link_libraries(matrixlib_gtests
        PRIVATE
            matrixlib
            GTest::GTest
            GTest::Main
    )

    # Discover tests
    gtest_discover_tests(matrixlib_gtests)
else()
    message(WARNING "MatrixLib: Google Test not found, tests will not be built")
endif()
