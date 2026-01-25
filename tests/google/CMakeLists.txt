# SPDX-License-Identifier: MIT
# MatrixLib Google Test Suite

cmake_minimum_required(VERSION 3.13.1)
project(matrixlib_tests CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MatrixLib (assuming it's in the parent directory)
if(NOT TARGET matrixlib)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/matrixlib)
endif()

# Find Google Test
find_package(GTest REQUIRED)
include(GoogleTest)

# Enable testing
enable_testing()

# Create test executable
add_executable(matrixlib_gtests
    test_constants.cpp
    test_vec.cpp
    test_mat.cpp
    test_quat.cpp
    test_edge_cases.cpp
    test_constexpr.cpp
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
