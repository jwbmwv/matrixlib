// SPDX-License-Identifier: MIT
/// @file test_main.cpp
/// @brief MatrixLib Zephyr Test Suite - Main Entry Point
/// @details Test suite setup and teardown for Zephyr ztest framework.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <zephyr/ztest.h>

// Test suite setup and teardown
static void* matrixlib_setup(void)
{
    printk("MatrixLib Test Suite Starting...\n");
    return NULL;
}

static void matrixlib_before(void* f)
{
    ARG_UNUSED(f);
}

static void matrixlib_after(void* f)
{
    ARG_UNUSED(f);
}

static void matrixlib_teardown(void* f)
{
    ARG_UNUSED(f);
    printk("MatrixLib Test Suite Complete\n");
}

// Register test suites (defined in other files)
extern void register_vec_tests(void);
extern void register_mat_tests(void);
extern void register_quat_tests(void);

ZTEST_SUITE(matrixlib_constants, NULL, matrixlib_setup, matrixlib_before, matrixlib_after, matrixlib_teardown);
ZTEST_SUITE(matrixlib_vec, NULL, matrixlib_setup, matrixlib_before, matrixlib_after, matrixlib_teardown);
ZTEST_SUITE(matrixlib_mat, NULL, matrixlib_setup, matrixlib_before, matrixlib_after, matrixlib_teardown);
ZTEST_SUITE(matrixlib_quat, NULL, matrixlib_setup, matrixlib_before, matrixlib_after, matrixlib_teardown);
