# Zephyr-specific CMake configuration for MatrixLib

if(CONFIG_MATRIXLIB)
    zephyr_interface_library_named(matrixlib)

    # Add include directory
    zephyr_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

    # Add compile definitions if NEON is enabled
    if(CONFIG_MATRIXLIB_NEON)
        zephyr_compile_definitions(CONFIG_MATRIXLIB_NEON)
        # Compatibility alias for non-Zephyr build switches
        zephyr_compile_definitions(MATRIXLIB_ENABLE_NEON)
    endif()

    # Add compile definitions if CMSIS is enabled
    if(CONFIG_MATRIXLIB_CMSIS)
        zephyr_compile_definitions(CONFIG_MATRIXLIB_CMSIS)
        # Compatibility alias for non-Zephyr build switches
        zephyr_compile_definitions(MATRIXLIB_ENABLE_CMSIS)
        # Link CMSIS-DSP if available in Zephyr
        if(CONFIG_CMSIS_DSP)
            zephyr_link_libraries(CMSIS)
        endif()
    endif()

    # Add compile definitions if MVE is enabled
    if(CONFIG_MATRIXLIB_MVE)
        zephyr_compile_definitions(CONFIG_MATRIXLIB_MVE)
        # Compatibility alias for non-Zephyr build switches
        zephyr_compile_definitions(MATRIXLIB_ENABLE_MVE)
    endif()

    # Link interface library to app
    target_link_libraries(matrixlib INTERFACE zephyr_interface)
endif()
