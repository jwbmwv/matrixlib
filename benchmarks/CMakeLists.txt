# SPDX-License-Identifier: MIT
# MatrixLib Performance Benchmarks

cmake_minimum_required(VERSION 3.13.1)
project(matrixlib_benchmarks CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MatrixLib
if(NOT TARGET matrixlib)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_BINARY_DIR}/matrixlib)
endif()

# Find Google Benchmark
find_package(benchmark QUIET)

if(benchmark_FOUND)
    # Matrix multiplication benchmark
    add_executable(bench_matrix_multiply bench_matrix_multiply.cpp)
    target_link_libraries(bench_matrix_multiply
        PRIVATE
            matrixlib
            benchmark::benchmark
            benchmark::benchmark_main
    )

    # SIMD vs non-SIMD benchmark
    add_executable(bench_simd bench_simd.cpp)
    target_link_libraries(bench_simd
        PRIVATE
            matrixlib
            benchmark::benchmark
            benchmark::benchmark_main
    )

    # Constexpr vs runtime benchmark
    add_executable(bench_constexpr bench_constexpr.cpp)
    target_link_libraries(bench_constexpr
        PRIVATE
            matrixlib
            benchmark::benchmark
            benchmark::benchmark_main
    )

    # Vector operations benchmark
    add_executable(bench_vector_ops bench_vector_ops.cpp)
    target_link_libraries(bench_vector_ops
        PRIVATE
            matrixlib
            benchmark::benchmark
            benchmark::benchmark_main
    )

    message(STATUS "MatrixLib benchmarks enabled")
else()
    message(STATUS "Google Benchmark not found. Benchmarks will not be built.")
    message(STATUS "Install with: sudo apt install libbenchmark-dev (Ubuntu)")
    message(STATUS "           or: brew install google-benchmark (macOS)")
    message(STATUS "           or: vcpkg install benchmark (Windows)")
endif()
