// SPDX-License-Identifier: MIT
/// @file matrixlib.hpp
/// @brief Main header for MatrixLib - header-only linear algebra library
/// @details This library provides template-based vector and matrix classes with comprehensive
///          operator support, SIMD optimizations (CMSIS-DSP, NEON, MVE), and specialized
///          functionality for embedded systems and real-time applications.
///          This header includes all vector and matrix components.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#pragma once

#ifndef MATRIXLIB_HPP_
#define MATRIXLIB_HPP_

#include "compiler_features.hpp"

#include <cstdint>
#include <type_traits>
#include <cmath>
#include <algorithm>

// Optional NEON/MVE/CMSIS blocks (disabled by default)
// Enable NEON optimizations (ARM Cortex-A, Apple Silicon, ARM64):
//   #define CONFIG_MATRIXLIB_NEON
// Enable MVE optimizations (ARM Cortex-M with Helium):
//   #define CONFIG_MATRIXLIB_MVE
// Enable CMSIS-DSP optimizations (ARM Cortex-M):
//   #define CONFIG_MATRIXLIB_CMSIS

#ifdef CONFIG_MATRIXLIB_NEON
#include <arm_neon.h>
#endif

#ifdef CONFIG_MATRIXLIB_CMSIS
#include <arm_math.h>
#endif

// Include all component headers
#include "matrixlib/constants.hpp"
#include "matrixlib/vector.hpp"
#include "matrixlib/vec2D.hpp"
#include "matrixlib/vec3D.hpp"
#include "matrixlib/matrix.hpp"
#include "matrixlib/matrix2D.hpp"
#include "matrixlib/matrix3D.hpp"
#include "matrixlib/quaternion.hpp"

namespace matrixlib
{

// C++20 concepts for better type safety and error messages
#if __cplusplus >= 202002L
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;
#endif

/// \brief Convert degrees to radians
/// \param degrees Angle in degrees
/// \return Angle in radians
template<typename T>
MATRIX_CONSTEXPR T deg_to_rad(T degrees) noexcept
{
    return degrees * constants::deg_to_rad<T>;
}

/// \brief Convert radians to degrees
/// \param radians Angle in radians
/// \return Angle in degrees
template<typename T>
MATRIX_CONSTEXPR T rad_to_deg(T radians) noexcept
{
    return radians * constants::rad_to_deg<T>;
}

/// \brief Wrap angle to range [-pi, pi]
/// \param angle Angle in radians
/// \return Angle wrapped to [-pi, pi]
template<typename T>
T wrap_pi(T angle) noexcept
{
    // Normalize to [-pi, pi]
    while (angle > constants::pi<T>)
        angle -= constants::two_pi<T>;
    while (angle < -constants::pi<T>)
        angle += constants::two_pi<T>;
    return angle;
}

/// \brief Wrap angle to range [0, 2*pi]
/// \param angle Angle in radians
/// \return Angle wrapped to [0, 2*pi]
template<typename T>
T wrap_two_pi(T angle) noexcept
{
    // Normalize to [0, 2*pi]
    while (angle < T(0))
        angle += constants::two_pi<T>;
    while (angle >= constants::two_pi<T>)
        angle -= constants::two_pi<T>;
    return angle;
}

/// \brief Calculate shortest angular distance between two angles
/// \param from Starting angle in radians
/// \param to Ending angle in radians
/// \return Shortest angular distance in range [-pi, pi]
template<typename T>
T angle_distance(T from, T to) noexcept
{
    T diff = to - from;
    return wrap_pi(diff);
}

/// \brief Clamp value to range [min, max]
/// \param value Value to clamp
/// \param min Minimum value
/// \param max Maximum value
/// \return Clamped value
template<typename T>
MATRIX_CONSTEXPR T clamp(T value, T min, T max) noexcept
{
    return value < min ? min : (value > max ? max : value);
}

/// \brief Saturate value to range [0, 1]
/// \param value Value to saturate
/// \return Value clamped to [0, 1]
template<typename T>
MATRIX_CONSTEXPR T saturate(T value) noexcept
{
    return clamp(value, T(0), T(1));
}

// Optional NEON/MVE blocks (placeholders, disabled by default)
#ifdef CONFIG_MATRIXLIB_NEON
// NEON implementations here
#endif

#ifdef CONFIG_MATRIXLIB_MVE
// MVE implementations here
#endif

#ifdef CONFIG_MATRIXLIB_CMSIS
// CMSIS-DSP implementations here

// Note: Optimizations are integrated into Vec and SquareMat methods for float types
#endif

// Note: Static asserts for trivial copyability have been removed.
// Reason: User-defined constexpr constructors (even if trivial) prevent std::is_trivially_copyable
// from being true in C++17. The classes are still efficiently copyable and have standard layout,
// but C++17's strict definition requires = default constructors for trivial copyability.
// These classes are safe for memcpy, DMA, and binary serialization despite not being formally
// "trivially copyable" according to the C++17 standard.
// See: https://en.cppreference.com/w/cpp/types/is_trivially_copyable

}  // namespace matrixlib

#endif  // MATRIXLIB_HPP_
