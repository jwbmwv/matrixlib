// SPDX-License-Identifier: MIT
/// @file test_constants.cpp
/// @brief MatrixLib Google Test Suite - Constants Tests
/// @details Unit tests for mathematical constants in the constants namespace.
///          Verifies precision, type safety, and accessibility.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <gtest/gtest.h>
#include <matrixlib/constants.hpp>
#include <cmath>

using namespace matrixlib;

class ConstantsTest : public ::testing::Test
{
protected:
    static constexpr double precision_epsilon = 1e-15;

    bool double_eq(double a, double b, double eps = precision_epsilon) const { return std::fabs(a - b) < eps; }
};

// ==================== Fundamental Constants Tests ====================

TEST_F(ConstantsTest, PiPrecision)
{
    // Test pi with high precision
    EXPECT_NEAR(constants::pi<double>, 3.14159265358979323846, 1e-15);
    EXPECT_NEAR(constants::pi<float>, 3.14159265f, 1e-6f);

    // Test type deduction (default should be double)
    EXPECT_TRUE((std::is_same<decltype(constants::pi<>), const double>::value));
}

TEST_F(ConstantsTest, PiDerivatives)
{
    // Two pi
    EXPECT_NEAR(constants::two_pi<double>, 2.0 * constants::pi<double>, 1e-15);
    EXPECT_NEAR(constants::two_pi<float>, 2.0f * constants::pi<float>, 1e-6f);

    // Half pi
    EXPECT_NEAR(constants::half_pi<double>, constants::pi<double> / 2.0, 1e-15);
    EXPECT_NEAR(constants::half_pi<float>, constants::pi<float> / 2.0f, 1e-6f);

    // Quarter pi
    EXPECT_NEAR(constants::quarter_pi<double>, constants::pi<double> / 4.0, 1e-15);
    EXPECT_NEAR(constants::quarter_pi<float>, constants::pi<float> / 4.0f, 1e-6f);
}

TEST_F(ConstantsTest, EulerNumber)
{
    EXPECT_NEAR(constants::e<double>, 2.71828182845904523536, 1e-15);
    EXPECT_NEAR(constants::e<float>, 2.71828182f, 1e-6f);
}

TEST_F(ConstantsTest, GoldenRatio)
{
    // Golden ratio = (1 + sqrt(5)) / 2
    double phi_calculated = (1.0 + std::sqrt(5.0)) / 2.0;
    EXPECT_NEAR(constants::golden_ratio<double>, phi_calculated, 1e-10);
    EXPECT_NEAR(constants::golden_ratio<double>, 1.61803398874989484820, 1e-15);
}

TEST_F(ConstantsTest, SquareRoots)
{
    EXPECT_NEAR(constants::sqrt2<double>, std::sqrt(2.0), 1e-15);
    EXPECT_NEAR(constants::sqrt3<double>, std::sqrt(3.0), 1e-15);

    EXPECT_NEAR(constants::sqrt2<float>, std::sqrt(2.0f), 1e-6f);
    EXPECT_NEAR(constants::sqrt3<float>, std::sqrt(3.0f), 1e-6f);
}

TEST_F(ConstantsTest, NaturalLogarithms)
{
    EXPECT_NEAR(constants::ln2<double>, std::log(2.0), 1e-15);
    EXPECT_NEAR(constants::ln10<double>, std::log(10.0), 1e-15);

    EXPECT_NEAR(constants::ln2<float>, std::log(2.0f), 1e-6f);
    EXPECT_NEAR(constants::ln10<float>, std::log(10.0f), 1e-6f);
}

// ==================== Conversion Factor Tests ====================

TEST_F(ConstantsTest, DegreeRadianConversion)
{
    // deg_to_rad = pi / 180
    double deg_to_rad_calculated = constants::pi<double> / 180.0;
    EXPECT_NEAR(constants::deg_to_rad<double>, deg_to_rad_calculated, 1e-15);

    // rad_to_deg = 180 / pi
    double rad_to_deg_calculated = 180.0 / constants::pi<double>;
    EXPECT_NEAR(constants::rad_to_deg<double>, rad_to_deg_calculated, 1e-15);

    // Test round-trip conversion
    double degrees = 90.0;
    double radians = degrees * constants::deg_to_rad<double>;
    double back_to_degrees = radians * constants::rad_to_deg<double>;
    EXPECT_NEAR(back_to_degrees, degrees, 1e-12);
}

TEST_F(ConstantsTest, ConversionExamples)
{
    // 90 degrees = pi/2 radians
    EXPECT_NEAR(90.0 * constants::deg_to_rad<double>, constants::half_pi<double>, 1e-12);

    // 180 degrees = pi radians
    EXPECT_NEAR(180.0 * constants::deg_to_rad<double>, constants::pi<double>, 1e-12);

    // 360 degrees = 2*pi radians
    EXPECT_NEAR(360.0 * constants::deg_to_rad<double>, constants::two_pi<double>, 1e-12);
}

// ==================== Epsilon Tests ====================

TEST_F(ConstantsTest, EpsilonValues)
{
    // Epsilon values are now based on std::numeric_limits<T>::epsilon() with scaling
    // epsilon_f ≈ 8x machine epsilon for float ≈ 9.5e-7
    // epsilon_d ≈ 4.5e6x machine epsilon for double ≈ 1e-9
    EXPECT_NEAR(constants::epsilon_f, std::numeric_limits<float>::epsilon() * 8.0f, 1e-8f);
    EXPECT_NEAR(constants::epsilon_d, std::numeric_limits<double>::epsilon() * 4.5e6, 1e-10);

    // Generic template: 100x machine epsilon
    EXPECT_NEAR(constants::epsilon<float>, std::numeric_limits<float>::epsilon() * 100.0f, 1e-8f);
    EXPECT_NEAR(constants::epsilon<double>, std::numeric_limits<double>::epsilon() * 100.0, 1e-12);
}

TEST_F(ConstantsTest, EpsilonTypeSafety)
{
    // Verify types are correct
    EXPECT_TRUE((std::is_same<decltype(constants::epsilon_f), const float>::value));
    EXPECT_TRUE((std::is_same<decltype(constants::epsilon_d), const double>::value));
}

// ==================== Physical Constants Tests ====================

TEST_F(ConstantsTest, PhysicalConstants)
{
    // Standard gravity
    EXPECT_NEAR(constants::gravity<double>, 9.80665, 1e-10);

    // Speed of light (exact value)
    EXPECT_EQ(constants::speed_of_light<double>, 299792458.0);
}

// ==================== Type Safety Tests ====================

TEST_F(ConstantsTest, TypeFlexibility)
{
    // Test that constants work with different types
    float pi_f = constants::pi<float>;
    double pi_d = constants::pi<double>;
    long double pi_ld = constants::pi<long double>;

    EXPECT_GT(pi_f, 3.14f);
    EXPECT_GT(pi_d, 3.14);
    EXPECT_GT(pi_ld, 3.14L);
}

// ==================== Compile-Time Tests ====================

TEST_F(ConstantsTest, ConstexprEvaluation)
{
    // These should compile successfully - verifies constexpr works
    constexpr double pi_constexpr = constants::pi<double>;
    constexpr float half_pi_constexpr = constants::half_pi<float>;
    constexpr double deg_to_rad_constexpr = constants::deg_to_rad<double>;

    EXPECT_NEAR(pi_constexpr, 3.14159265358979323846, 1e-15);
    EXPECT_NEAR(half_pi_constexpr, 1.5707963f, 1e-6f);
    EXPECT_NEAR(deg_to_rad_constexpr, 0.01745329251994329577, 1e-15);
}

// ==================== Usage Pattern Tests ====================

TEST_F(ConstantsTest, RealWorldUsage)
{
    // Test constants in actual calculations

    // Circle circumference: C = 2 * pi * r
    double radius = 5.0;
    double circumference = constants::two_pi<double> * radius;
    EXPECT_NEAR(circumference, 31.41592653589793, 1e-12);

    // Right triangle with 45-degree angle
    double angle_45_rad = 45.0 * constants::deg_to_rad<double>;
    EXPECT_NEAR(angle_45_rad, constants::quarter_pi<double>, 1e-12);

    // Pythagorean diagonal of unit square
    double diagonal = constants::sqrt2<double>;
    EXPECT_NEAR(diagonal, std::sqrt(1.0 * 1.0 + 1.0 * 1.0), 1e-15);
}

TEST_F(ConstantsTest, IndependentUsage)
{
    // Verify constants.hpp can be used independently
    // (This test file only includes constants.hpp, not the full matrixlib.hpp)

    double area_of_circle = constants::pi<double> * 10.0 * 10.0;
    EXPECT_NEAR(area_of_circle, 314.159265, 1e-6);
}
