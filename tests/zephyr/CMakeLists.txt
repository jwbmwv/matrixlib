# SPDX-License-Identifier: MIT
# MatrixLib Zephyr Test Suite

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(matrixlib_tests)

# Add test sources
target_sources(app PRIVATE
    src/test_constants.cpp
    src/test_vec.cpp
    src/test_mat.cpp
    src/test_quat.cpp
    src/test_main.cpp
)
