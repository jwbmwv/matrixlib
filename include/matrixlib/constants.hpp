// SPDX-License-Identifier: MIT
/// \file constants.hpp
/// \brief Mathematical constants with C++11-C++26 optimizations
/// \details Provides compile-time mathematical constants for various numeric types.
///          This header is independent and can be used by any component.
/// \copyright Copyright (c) 2026 James Baldwin
/// \author James Baldwin

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "compiler_features.hpp"

namespace matrixlib
{

/// \namespace constants
/// \brief Mathematical constants for common calculations
/// \details All constants are provided as template variables for type flexibility.
///          Default type is double for maximum precision.
namespace constants
{
/// \brief Pi (π) - Ratio of circle's circumference to diameter
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T pi = T(3.14159265358979323846);

/// \brief Two times Pi (2π)
/// \tparam T Numeric type (default: double)
/// \details Used for full circle calculations (360 degrees = 2π radians)
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T two_pi = T(6.28318530717958647692);

/// \brief Half Pi (π/2)
/// \tparam T Numeric type (default: double)
/// \details 90 degrees in radians
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T half_pi = T(1.57079632679489661923);

/// \brief Quarter Pi (π/4)
/// \tparam T Numeric type (default: double)
/// \details 45 degrees in radians
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T quarter_pi = T(0.78539816339744830962);

/// \brief Euler's number (e) - Base of natural logarithms
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T e = T(2.71828182845904523536);

/// \brief Golden ratio (φ) - (1 + √5) / 2
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T golden_ratio = T(1.61803398874989484820);

/// \brief Square root of 2 (√2)
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T sqrt2 = T(1.41421356237309504880);

/// \brief Square root of 3 (√3)
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T sqrt3 = T(1.73205080756887729352);

/// \brief Natural logarithm of 2 (ln(2))
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T ln2 = T(0.69314718055994530942);

/// \brief Natural logarithm of 10 (ln(10))
/// \tparam T Numeric type (default: double)
/// \details Precision: 20 decimal places
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T ln10 = T(2.30258509299404568402);

/// \brief Default epsilon for floating-point comparisons (float)
/// \details Scaled machine epsilon (~8x) for practical comparison tolerance.
///          Use for approximate equality: abs(a - b) < epsilon_f()
inline MATRIX_CONSTEXPR float epsilon_f() noexcept
{
    return std::numeric_limits<float>::epsilon() * 8.0f;  // ≈ 9.5e-7
}

/// \brief Default epsilon for floating-point comparisons (double)
/// \details Scaled machine epsilon (~4.5e6x) for practical comparison tolerance.
///          Use for approximate equality: abs(a - b) < epsilon_d()
inline MATRIX_CONSTEXPR double epsilon_d() noexcept
{
    return std::numeric_limits<double>::epsilon() * 4.5e6;  // ≈ 1e-9 (relaxed from 1e-12)
}

/// \brief Default epsilon for floating-point comparisons (generic)
/// \tparam T Numeric type (default: double)
/// \details Scaled machine epsilon for practical comparison tolerance.
///          For float: ~8x machine epsilon, for double: ~100x machine epsilon
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T epsilon = std::numeric_limits<T>::epsilon() * T(100);

// ==================== Conversion Factors ====================

/// \brief Degrees to radians multiplier (π/180)
/// \tparam T Numeric type (default: double)
/// \details Multiply degrees by this to get radians
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T deg_to_rad = T(0.01745329251994329577);

/// \brief Radians to degrees multiplier (180/π)
/// \tparam T Numeric type (default: double)
/// \details Multiply radians by this to get degrees
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T rad_to_deg = T(57.29577951308232087680);

// ==================== Physical Constants (Optional - Commonly Used in Embedded) ====================

/// \brief Gravity acceleration at Earth's surface (m/s²)
/// \tparam T Numeric type (default: double)
/// \details Standard gravity: 9.80665 m/s²
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T gravity = T(9.80665);

/// \brief Speed of light in vacuum (m/s)
/// \tparam T Numeric type (default: double)
/// \details Exact value: 299,792,458 m/s
template<typename T = double>
MATRIX_INLINE_VAR MATRIX_CONSTEXPR T speed_of_light = T(299792458.0);

}  // namespace constants

}  // namespace matrixlib
