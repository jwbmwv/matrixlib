cmake_minimum_required(VERSION 3.13.1)

project(matrixlib VERSION 1.0.0 LANGUAGES CXX)

# Interface library (header-only)
add_library(matrixlib INTERFACE)
add_library(matrixlib::matrixlib ALIAS matrixlib)

# Include directories
target_include_directories(matrixlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Require C++11
target_compile_features(matrixlib INTERFACE cxx_std_11)

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "IAR")
    # IAR Embedded Workbench for ARM
    # Enable C++11 support
    target_compile_options(matrixlib INTERFACE --c++ --cpp_init_routine=_Z41__static_initialization_and_destruction_0ii)
    # Suppress specific warnings if needed
    # target_compile_options(matrixlib INTERFACE --diag_suppress=Pe111,Pe177)
endif()

# Optional NEON support (ARM Cortex-A, Apple Silicon, ARM64)
option(MATRIXLIB_ENABLE_NEON "Enable ARM NEON optimizations" OFF)
if(MATRIXLIB_ENABLE_NEON)
    target_compile_definitions(matrixlib INTERFACE CONFIG_MATRIXLIB_NEON)
    # Enable NEON on ARM platforms
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
        if(CMAKE_CXX_COMPILER_ID MATCHES "IAR")
            # IAR: NEON is typically enabled via project settings or --fpu=VFPv4_sp
            target_compile_options(matrixlib INTERFACE --fpu=VFPv4_sp)
        else()
            target_compile_options(matrixlib INTERFACE -mfpu=neon)
        endif()
    endif()
endif()

# Optional CMSIS-DSP support (ARM Cortex-M)
option(MATRIXLIB_ENABLE_CMSIS "Enable CMSIS-DSP optimizations" OFF)
if(MATRIXLIB_ENABLE_CMSIS)
    target_compile_definitions(matrixlib INTERFACE CONFIG_MATRIXLIB_CMSIS)
    # User must link CMSIS::DSP in their project
endif()

# Optional: Build examples
option(MATRIX_LINEAR_BUILD_EXAMPLES "Build example programs" OFF)
if(MATRIX_LINEAR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional: Build tests
option(MATRIX_LINEAR_BUILD_TESTS "Build test programs" OFF)
if(MATRIX_LINEAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: Build benchmarks
option(MATRIX_LINEAR_BUILD_BENCHMARKS "Build benchmark programs" OFF)
if(MATRIX_LINEAR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Optional: Build documentation
option(MATRIX_LINEAR_BUILD_DOCS "Build Doxygen documentation" OFF)
if(MATRIX_LINEAR_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        message(STATUS "Doxygen found, documentation will be built")
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
    else()
        message(WARNING "Doxygen not found, documentation will not be built")
    endif()
endif()

# Installation
include(GNUInstallDirs)
install(DIRECTORY include/matrixlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS matrixlib
    EXPORT matrixlib-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT matrixlib-targets
    FILE matrixlib-targets.cmake
    NAMESPACE matrixlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrixlib
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/matrixlib-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/matrixlib-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrixlib
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/matrixlib-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/matrixlib-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/matrixlib-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrixlib
)
