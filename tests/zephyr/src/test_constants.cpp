// SPDX-License-Identifier: MIT
/// @file test_constants.cpp
/// @brief MatrixLib Zephyr Test Suite - Constants Tests
/// @details Unit tests for mathematical constants in the constants namespace.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <zephyr/ztest.h>
#include <matrixlib/constants.hpp>
#include <math.h>

using namespace matrixlib;

// Precision for floating point comparisons
#define FLOAT_EPSILON 1e-6f
#define DOUBLE_EPSILON 1e-12

// ==================== Fundamental Constants Tests ====================

ZTEST(matrixlib_constants, test_pi_precision)
{
    zassert_within(constants::pi<float>, 3.14159265f, FLOAT_EPSILON, "Pi (float) precision");
    zassert_within(constants::pi<double>, 3.14159265358979323846, DOUBLE_EPSILON, "Pi (double) precision");
}

ZTEST(matrixlib_constants, test_pi_derivatives)
{
    // Two pi
    zassert_within(constants::two_pi<float>, 2.0f * constants::pi<float>, FLOAT_EPSILON, "Two pi");

    // Half pi
    zassert_within(constants::half_pi<float>, constants::pi<float> / 2.0f, FLOAT_EPSILON, "Half pi");

    // Quarter pi
    zassert_within(constants::quarter_pi<float>, constants::pi<float> / 4.0f, FLOAT_EPSILON, "Quarter pi");
}

ZTEST(matrixlib_constants, test_euler_number)
{
    zassert_within(constants::e<float>, 2.71828182f, FLOAT_EPSILON, "Euler's number");
}

ZTEST(matrixlib_constants, test_golden_ratio)
{
    // Golden ratio = (1 + sqrt(5)) / 2
    float phi_calculated = (1.0f + sqrtf(5.0f)) / 2.0f;
    zassert_within(constants::golden_ratio<float>, phi_calculated, 1e-5f, "Golden ratio");
}

ZTEST(matrixlib_constants, test_square_roots)
{
    zassert_within(constants::sqrt2<float>, sqrtf(2.0f), FLOAT_EPSILON, "Square root of 2");
    zassert_within(constants::sqrt3<float>, sqrtf(3.0f), FLOAT_EPSILON, "Square root of 3");
}

ZTEST(matrixlib_constants, test_natural_logarithms)
{
    zassert_within(constants::ln2<float>, logf(2.0f), FLOAT_EPSILON, "Natural log of 2");
    zassert_within(constants::ln10<float>, logf(10.0f), FLOAT_EPSILON, "Natural log of 10");
}

// ==================== Conversion Factor Tests ====================

ZTEST(matrixlib_constants, test_degree_radian_conversion)
{
    // deg_to_rad = pi / 180
    float deg_to_rad_calculated = constants::pi<float> / 180.0f;
    zassert_within(constants::deg_to_rad<float>, deg_to_rad_calculated, FLOAT_EPSILON, "Degrees to radians");

    // rad_to_deg = 180 / pi
    float rad_to_deg_calculated = 180.0f / constants::pi<float>;
    zassert_within(constants::rad_to_deg<float>, rad_to_deg_calculated, FLOAT_EPSILON, "Radians to degrees");

    // Round-trip conversion
    float degrees = 90.0f;
    float radians = degrees * constants::deg_to_rad<float>;
    float back_to_degrees = radians * constants::rad_to_deg<float>;
    zassert_within(back_to_degrees, degrees, 1e-5f, "Round-trip conversion");
}

ZTEST(matrixlib_constants, test_conversion_examples)
{
    // 90 degrees = pi/2 radians
    zassert_within(90.0f * constants::deg_to_rad<float>, constants::half_pi<float>, 1e-5f, "90 degrees");

    // 180 degrees = pi radians
    zassert_within(180.0f * constants::deg_to_rad<float>, constants::pi<float>, 1e-5f, "180 degrees");

    // 360 degrees = 2*pi radians
    zassert_within(360.0f * constants::deg_to_rad<float>, constants::two_pi<float>, 1e-5f, "360 degrees");
}

// ==================== Epsilon Tests ====================

ZTEST(matrixlib_constants, test_epsilon_values)
{
    zassert_equal(constants::epsilon_f, 1e-6f, "Single precision epsilon");
    zassert_equal(constants::epsilon_d, 1e-12, "Double precision epsilon");
}

// ==================== Physical Constants Tests ====================

ZTEST(matrixlib_constants, test_physical_constants)
{
    // Standard gravity
    zassert_within(constants::gravity<float>, 9.80665f, 1e-5f, "Standard gravity");

    // Speed of light (exact value)
    zassert_equal(constants::speed_of_light<float>, 299792458.0f, "Speed of light");
}

// ==================== Constexpr Tests ====================

ZTEST(matrixlib_constants, test_constexpr_evaluation)
{
    // These should compile successfully - verifies constexpr works
    constexpr float pi_constexpr = constants::pi<float>;
    constexpr float half_pi_constexpr = constants::half_pi<float>;
    constexpr float deg_to_rad_constexpr = constants::deg_to_rad<float>;

    zassert_within(pi_constexpr, 3.14159265f, FLOAT_EPSILON, "Constexpr pi");
    zassert_within(half_pi_constexpr, 1.5707963f, FLOAT_EPSILON, "Constexpr half pi");
    zassert_within(deg_to_rad_constexpr, 0.01745329f, FLOAT_EPSILON, "Constexpr deg_to_rad");
}

// ==================== Real-World Usage Tests ====================

ZTEST(matrixlib_constants, test_real_world_usage)
{
    // Circle circumference: C = 2 * pi * r
    float radius = 5.0f;
    float circumference = constants::two_pi<float> * radius;
    zassert_within(circumference, 31.415926f, 1e-5f, "Circle circumference");

    // Right triangle with 45-degree angle
    float angle_45_rad = 45.0f * constants::deg_to_rad<float>;
    zassert_within(angle_45_rad, constants::quarter_pi<float>, 1e-5f, "45 degree angle");
}

ZTEST(matrixlib_constants, test_independent_usage)
{
    // Verify constants.hpp can be used independently
    float area_of_circle = constants::pi<float> * 10.0f * 10.0f;
    zassert_within(area_of_circle, 314.159265f, 1e-4f, "Circle area");
}
