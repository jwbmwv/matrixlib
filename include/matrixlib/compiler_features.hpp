// SPDX-License-Identifier: MIT
/// @file compiler_features.hpp
/// @brief Compiler feature detection and cross-version C++ compatibility macros
/// @details Provides MATRIX_* macros that adapt to different C++ standard versions (C++11-C++26).
///          This header centralizes all compiler feature detection to avoid duplication.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#pragma once

#ifndef MATRIXLIB_COMPILER_FEATURES_HPP_
#define MATRIXLIB_COMPILER_FEATURES_HPP_

// ==================== C++14 Features ====================
#if __cplusplus >= 201402L
#define MATRIX_CONSTEXPR constexpr
#define MATRIX_CONSTEXPR14 constexpr
#else
#define MATRIX_CONSTEXPR
#define MATRIX_CONSTEXPR14
#endif

// ==================== C++17 Features ====================
// Note: MATRIX_NODISCARD must be placed BEFORE the return type in function declarations
// Correct: MATRIX_NODISCARD MATRIX_CONSTEXPR ReturnType func()
// Incorrect: MATRIX_CONSTEXPR MATRIX_NODISCARD ReturnType func()
#if __cplusplus >= 201703L
#define MATRIX_NODISCARD [[nodiscard]]
#define MATRIX_CONSTEXPR17 constexpr
#define MATRIX_INLINE_VAR inline
#define MATRIX_IF_CONSTEXPR if constexpr
#else
#define MATRIX_NODISCARD
#define MATRIX_CONSTEXPR17
#define MATRIX_INLINE_VAR static
#define MATRIX_IF_CONSTEXPR if
#endif

// ==================== C++20 Features ====================
#if __cplusplus >= 202002L
#define MATRIX_CONSTEXPR20 constexpr
#define MATRIX_CONSTEVAL consteval
#define MATRIX_CONSTINIT constinit
#define MATRIX_LIKELY [[likely]]
#define MATRIX_UNLIKELY [[unlikely]]
#else
#define MATRIX_CONSTEXPR20
#define MATRIX_CONSTEVAL constexpr
#define MATRIX_CONSTINIT
#define MATRIX_LIKELY
#define MATRIX_UNLIKELY
#endif

// ==================== C++23 Features ====================
#if __cplusplus >= 202302L
#define MATRIX_CONSTEXPR23 constexpr
#define MATRIX_IF_CONSTEVAL if consteval
#include <utility>
#define MATRIX_UNREACHABLE() std::unreachable()
#else
#define MATRIX_CONSTEXPR23
#define MATRIX_IF_CONSTEVAL if (false)
// Compiler-specific unreachable hints for older standards
#if defined(__GNUC__) || defined(__clang__)
#define MATRIX_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define MATRIX_UNREACHABLE() __assume(0)
#else
#define MATRIX_UNREACHABLE() \
    do                       \
    {                        \
    } while (0)
#endif
#endif

// ==================== C++26 Features ====================
#if __cplusplus >= 202600L
#define MATRIX_CONSTEXPR26 constexpr
// C++26 makes std::sin, std::cos, std::sqrt, etc. constexpr
#define MATRIX_CONSTEXPR_TRIG constexpr
#else
#define MATRIX_CONSTEXPR26
#define MATRIX_CONSTEXPR_TRIG
#endif

// ==================== Type Conversion ====================
#if __cplusplus >= 202002L
#include <bit>
#define MATRIX_BIT_CAST(T, val) std::bit_cast<T>(val)
#else
// C++11 fallback: Use memcpy-based type conversion (safe, well-defined)
// This avoids strict aliasing violations that reinterpret_cast can cause
#include <cstring>
#include <type_traits>

namespace matrixlib
{
namespace detail
{
template<typename To, typename From>
inline To bit_cast_memcpy(const From& src) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "bit_cast requires same size types");
    static_assert(std::is_trivially_copyable<To>::value, "To must be trivially copyable");
    static_assert(std::is_trivially_copyable<From>::value, "From must be trivially copyable");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}
}  // namespace detail
}  // namespace matrixlib

#define MATRIX_BIT_CAST(T, val) (::matrixlib::detail::bit_cast_memcpy<T>(val))
#endif

// ==================== Compiler Hints & Optimizations ====================
// MATRIX_ASSUME: Optimization hint that condition is always true
#if defined(__clang__)
#define MATRIX_ASSUME(cond) __builtin_assume(cond)
#elif defined(_MSC_VER)
#define MATRIX_ASSUME(cond) __assume(cond)
#elif defined(__GNUC__) && __GNUC__ >= 13
#define MATRIX_ASSUME(cond) __attribute__((assume(cond)))
#else
#define MATRIX_ASSUME(cond)       \
    do                            \
    {                             \
        if (!(cond))              \
            MATRIX_UNREACHABLE(); \
    } while (0)
#endif

// MATRIX_FORCEINLINE: Strong inline hint for performance-critical code
#if defined(_MSC_VER)
#define MATRIX_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MATRIX_FORCEINLINE __attribute__((always_inline)) inline
#else
#define MATRIX_FORCEINLINE inline
#endif

#endif  // MATRIXLIB_COMPILER_FEATURES_HPP_
