// SPDX-License-Identifier: MIT
/// @file vector.hpp
/// @brief Generic vector class Vec<T,N> with full operator support
/// @details This header provides the template-based vector class with comprehensive
///          operator support, SIMD optimizations (CMSIS-DSP, NEON, MVE), and specialized
///          functionality for embedded systems and real-time applications.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#pragma once

#ifndef _MATRIXLIB_VECTOR_HPP_
#define _MATRIXLIB_VECTOR_HPP_

#include <cstdint>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <limits>

#include "compiler_features.hpp"
#include "constants.hpp"

// Optional NEON/MVE/CMSIS blocks (disabled by default)
#ifdef CONFIG_MATRIXLIB_NEON
#include <arm_neon.h>
#endif

#ifdef CONFIG_MATRIXLIB_CMSIS
#include <arm_math.h>
#endif

namespace matrixlib
{

/// \class Vec<T, N>
/// \brief A vector class templated on type T and dimension N.
/// \tparam T The storage type.
/// \tparam N The dimension.
template<typename T, std::uint32_t N>
class Vec
{
public:
    // Common component indices (shared with Quaternion)
    enum
    {
        X = 0,
        Y = 1,
        Z = 2,
        W = 3
    };

    // Single-dimensional array ensures contiguous memory layout, POD compatibility,
    // and efficient SIMD operations. Row-major storage: data[i] = element i
    alignas(16) T data[N];  // Aligned for SIMD performance

    /// \brief Default constructor (zero-initialized for safety).
    MATRIX_CONSTEXPR Vec()
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = T(0);
        }
    }

    /// \brief Constructor from array.
    /// \param arr The array to copy from.
    MATRIX_CONSTEXPR explicit Vec(const T* arr)
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = arr[i];
        }
    }

    /// \brief Copy constructor.
    /// \param other The vector to copy.
    MATRIX_CONSTEXPR Vec(const Vec& other)
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = other.data[i];
        }
    }

    /// \brief Variadic constructor for direct initialization.
    /// \tparam Args The types of the arguments.
    /// \param args The values to initialize with.
    template<typename... Args>
    MATRIX_CONSTEXPR Vec(typename std::enable_if<sizeof...(Args) == N && sizeof...(Args) >= 2, T>::type first,
                         Args... args)
    {
        T temp[] = {first, static_cast<T>(args)...};
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = temp[i];
        }
    }

    /// \brief Subscript operator.
    /// \param index The index.
    /// \return Reference to the element.
    T& operator[](std::uint32_t index) { return data[index]; }

    /// \brief Subscript operator (const).
    /// \param index The index.
    /// \return Const reference to the element.
    const T& operator[](std::uint32_t index) const { return data[index]; }

    /// \brief Bounds-checked element access.
    /// \param index The index.
    /// \return Reference to the element.
    /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if index >= N.
    T& at(std::uint32_t index)
    {
#ifdef MATRIXLIB_DEBUG
        assert(index < N && "Vec::at: index out of range");
#endif
        return data[index];
    }

    /// \brief Bounds-checked element access (const).
    /// \param index The index.
    /// \return Const reference to the element.
    /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if index >= N.
    const T& at(std::uint32_t index) const
    {
#ifdef MATRIXLIB_DEBUG
        assert(index < N && "Vec::at: index out of range");
#endif
        return data[index];
    }

    /// \brief Get the number of elements in the vector.
    /// \return The size of the vector (N).
    static constexpr std::size_t size() noexcept { return N; }

    /// \brief Addition operator.
    /// \param other The vector to add.
    /// \return The result vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec operator+(const Vec& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        if (std::is_same<T, float>::value && N == 2)
        {
            Vec result;
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t b = vld1_f32(reinterpret_cast<const float*>(other.data));
            float32x2_t r = vadd_f32(a, b);
            vst1_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
        if (std::is_same<T, float>::value && N == 3)
        {
            Vec result;
            // Load 4 floats (3 + padding) for alignment
            float temp_a[4] = {data[0], data[1], data[2], 0.0f};
            float temp_b[4] = {other.data[0], other.data[1], other.data[2], 0.0f};
            float32x4_t a = vld1q_f32(temp_a);
            float32x4_t b = vld1q_f32(temp_b);
            float32x4_t r = vaddq_f32(a, b);
            float temp_r[4];
            vst1q_f32(temp_r, r);
            result.data[0] = temp_r[0];
            result.data[1] = temp_r[1];
            result.data[2] = temp_r[2];
            return result;
        }
        if (std::is_same<T, float>::value && N == 4)
        {
            Vec result;
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t b = vld1q_f32(reinterpret_cast<const float*>(other.data));
            float32x4_t r = vaddq_f32(a, b);
            vst1q_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Vec result;
            arm_add_f32(reinterpret_cast<const float*>(data), reinterpret_cast<const float*>(other.data),
                        reinterpret_cast<float*>(result.data), N);
            return result;
        }
#endif
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    /// \brief Subtraction operator.
    /// \param other The vector to subtract.
    /// \return The result vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec operator-(const Vec& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        if (std::is_same<T, float>::value && N == 2)
        {
            Vec result;
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t b = vld1_f32(reinterpret_cast<const float*>(other.data));
            float32x2_t r = vsub_f32(a, b);
            vst1_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
        if (std::is_same<T, float>::value && N == 4)
        {
            Vec result;
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t b = vld1q_f32(reinterpret_cast<const float*>(other.data));
            float32x4_t r = vsubq_f32(a, b);
            vst1q_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Vec result;
            arm_sub_f32(reinterpret_cast<const float*>(data), reinterpret_cast<const float*>(other.data),
                        reinterpret_cast<float*>(result.data), N);
            return result;
        }
#endif
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    /// \brief Unary minus operator.
    /// \return The negated vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec operator-() const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = -data[i];
        }
        return result;
    }

    /// \brief Scalar multiplication operator.
    /// \param scalar The scalar to multiply by.
    /// \return The result vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec operator*(T scalar) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        if (std::is_same<T, float>::value && N == 2)
        {
            Vec result;
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t s = vdup_n_f32(static_cast<float>(scalar));
            float32x2_t r = vmul_f32(a, s);
            vst1_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
        if (std::is_same<T, float>::value && N == 4)
        {
            Vec result;
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t s = vdupq_n_f32(static_cast<float>(scalar));
            float32x4_t r = vmulq_f32(a, s);
            vst1q_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Vec result;
            arm_scale_f32(reinterpret_cast<const float*>(data), static_cast<float>(scalar),
                          reinterpret_cast<float*>(result.data), N);
            return result;
        }
#endif
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    /// \brief Scalar division operator.
    /// \param scalar The scalar to divide by.
    /// \return The result vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec operator/(T scalar) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        // Fast reciprocal for NEON float vectors
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 2)
        {
            MATRIX_LIKELY
            Vec result;
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t s = vdup_n_f32(static_cast<float>(scalar));
            // Fast reciprocal with Newton-Raphson refinement
            float32x2_t recip = vrecpe_f32(s);
            recip = vmul_f32(recip, vrecps_f32(s, recip));
            recip = vmul_f32(recip, vrecps_f32(s, recip));
            float32x2_t r = vmul_f32(a, recip);
            vst1_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 3)
        {
            MATRIX_LIKELY
            Vec result;
            float32x4_t a = vld1q_dup_f32(&data[0]);
            a = vld1q_lane_f32(&data[0], a, 0);
            a = vld1q_lane_f32(&data[1], a, 1);
            a = vld1q_lane_f32(&data[2], a, 2);
            float32x4_t s = vdupq_n_f32(static_cast<float>(scalar));
            float32x4_t recip = vrecpeq_f32(s);
            recip = vmulq_f32(recip, vrecpsq_f32(s, recip));
            recip = vmulq_f32(recip, vrecpsq_f32(s, recip));
            float32x4_t r = vmulq_f32(a, recip);
            vst1q_lane_f32(&result.data[0], r, 0);
            vst1q_lane_f32(&result.data[1], r, 1);
            vst1q_lane_f32(&result.data[2], r, 2);
            return result;
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 4)
        {
            MATRIX_LIKELY
            Vec result;
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t s = vdupq_n_f32(static_cast<float>(scalar));
            float32x4_t recip = vrecpeq_f32(s);
            recip = vmulq_f32(recip, vrecpsq_f32(s, recip));
            recip = vmulq_f32(recip, vrecpsq_f32(s, recip));
            float32x4_t r = vmulq_f32(a, recip);
            vst1q_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
#endif
        Vec result;
        // Check for zero to avoid undefined behavior (minimal overhead - single comparison)
        if (scalar == T(0))
        {
            // Return zero vector for safety
            for (std::uint32_t i = 0; i < N; ++i)
            {
                result.data[i] = T(0);
            }
            return result;
        }
        const T inv_scalar = T(1) / scalar;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] * inv_scalar;
        }
        return result;
    }

    /// \brief Equality operator.
    /// \param other The vector to compare.
    /// \return True if equal (uses epsilon comparison for floating point types).
    MATRIX_CONSTEXPR bool operator==(const Vec& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        // SIMD path for float equality comparison with epsilon
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 4)
        {
            MATRIX_LIKELY
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t b = vld1q_f32(reinterpret_cast<const float*>(other.data));
            float32x4_t diff = vabdq_f32(a, b);
            float32x4_t eps = vdupq_n_f32(std::numeric_limits<float>::epsilon());
            uint32x4_t cmp = vcleq_f32(diff, eps);
            // All lanes must be true
            uint64x2_t cmp64 = vreinterpretq_u64_u32(cmp);
            return vgetq_lane_u64(cmp64, 0) == ~0ULL && vgetq_lane_u64(cmp64, 1) == ~0ULL;
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 3)
        {
            MATRIX_LIKELY
            // Use approximate comparison for floats
            for (std::uint32_t i = 0; i < 3; ++i)
            {
                if (std::abs(data[i] - other.data[i]) > std::numeric_limits<float>::epsilon())
                    return false;
            }
            return true;
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 2)
        {
            MATRIX_LIKELY
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t b = vld1_f32(reinterpret_cast<const float*>(other.data));
            float32x2_t diff = vabd_f32(a, b);
            float32x2_t eps = vdup_n_f32(std::numeric_limits<float>::epsilon());
            uint32x2_t cmp = vcle_f32(diff, eps);
            // Both lanes must be true
            return vget_lane_u32(cmp, 0) == ~0U && vget_lane_u32(cmp, 1) == ~0U;
        }
#endif
        // For floating point types, use epsilon comparison
        MATRIX_IF_CONSTEXPR(std::is_floating_point<T>::value)
        {
            MATRIX_LIKELY
            for (std::uint32_t i = 0; i < N; ++i)
            {
                if (std::abs(data[i] - other.data[i]) > std::numeric_limits<T>::epsilon())
                {
                    return false;
                }
            }
            return true;
        }
        // For integral types, use exact comparison
        for (std::uint32_t i = 0; i < N; ++i)
        {
            if (data[i] != other.data[i])
            {
                return false;
            }
        }
        return true;
    }

    /// \brief Inequality operator.
    /// \param other The vector to compare.
    /// \return True if not equal.
    bool operator!=(const Vec& other) const noexcept { return !(*this == other); }

    /// \brief Addition assignment operator.
    /// \param other The vector to add.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Vec& operator+=(const Vec& other) noexcept
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = data[i] + other.data[i];
        }
        return *this;
    }

    /// \brief Subtraction assignment operator.
    /// \param other The vector to subtract.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Vec& operator-=(const Vec& other) noexcept
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = data[i] - other.data[i];
        }
        return *this;
    }

    /// \brief Scalar multiplication assignment operator.
    /// \param scalar The scalar to multiply by.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Vec& operator*=(T scalar) noexcept
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = data[i] * scalar;
        }
        return *this;
    }

    /// \brief Scalar division assignment operator.
    /// \param scalar The scalar to divide by.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Vec& operator/=(T scalar) noexcept
    {
        // Check for zero to avoid undefined behavior (minimal overhead)
        if (scalar == T(0))
        {
            // Set to zero for safety
            for (std::uint32_t i = 0; i < N; ++i)
            {
                data[i] = T(0);
            }
            return *this;
        }
        const T inv_scalar = T(1) / scalar;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            data[i] = data[i] * inv_scalar;
        }
        return *this;
    }

    /// \brief Dot product.
    /// \param other The other vector.
    /// \return The dot product.
    MATRIX_CONSTEXPR14 T dot(const Vec& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 2)
        {
            MATRIX_LIKELY
            float32x2_t a = vld1_f32(reinterpret_cast<const float*>(data));
            float32x2_t b = vld1_f32(reinterpret_cast<const float*>(other.data));
            float32x2_t mul = vmul_f32(a, b);
            float32x2_t sum = vpadd_f32(mul, mul);  // Horizontal add
            return static_cast<T>(vget_lane_f32(sum, 0));
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 3)
        {
            MATRIX_LIKELY
            // Load using lane operations to avoid temp arrays
            float32x4_t a = vld1q_dup_f32(&data[0]);
            a = vld1q_lane_f32(&data[0], a, 0);
            a = vld1q_lane_f32(&data[1], a, 1);
            a = vld1q_lane_f32(&data[2], a, 2);
            float32x4_t b = vld1q_dup_f32(&other.data[0]);
            b = vld1q_lane_f32(&other.data[0], b, 0);
            b = vld1q_lane_f32(&other.data[1], b, 1);
            b = vld1q_lane_f32(&other.data[2], b, 2);
            float32x4_t mul = vmulq_f32(a, b);
            float32x2_t sum = vadd_f32(vget_low_f32(mul), vget_high_f32(mul));
            sum = vpadd_f32(sum, sum);
            return static_cast<T>(vget_lane_f32(sum, 0));
        }
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && N == 4)
        {
            MATRIX_LIKELY
            float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
            float32x4_t b = vld1q_f32(reinterpret_cast<const float*>(other.data));
            float32x4_t mul = vmulq_f32(a, b);
            float32x2_t sum = vadd_f32(vget_low_f32(mul), vget_high_f32(mul));
            sum = vpadd_f32(sum, sum);
            return static_cast<T>(vget_lane_f32(sum, 0));
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value)
        {
            MATRIX_LIKELY
            float result;
            arm_dot_prod_f32(reinterpret_cast<const float*>(data), reinterpret_cast<const float*>(other.data), N,
                             &result);
            return static_cast<T>(result);
        }
#endif
        T sum = T(0);
        for (std::uint32_t i = 0; i < N; ++i)
        {
            sum += data[i] * other.data[i];
        }
        return sum;
    }

    /// \brief Cross product (only for 3D vectors).
    /// \param other The other vector.
    /// \return The cross product.
    template<std::uint32_t NN = N>
    MATRIX_CONSTEXPR MATRIX_NODISCARD typename std::enable_if<NN == 3, Vec>::type cross(const Vec& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value)
        {
            MATRIX_LIKELY
            Vec result;
            // Load vectors: [x, y, z, 0]
            float32x4_t a = vld1q_dup_f32(&data[0]);
            a = vld1q_lane_f32(&data[0], a, 0);
            a = vld1q_lane_f32(&data[1], a, 1);
            a = vld1q_lane_f32(&data[2], a, 2);
            float32x4_t b = vld1q_dup_f32(&other.data[0]);
            b = vld1q_lane_f32(&other.data[0], b, 0);
            b = vld1q_lane_f32(&other.data[1], b, 1);
            b = vld1q_lane_f32(&other.data[2], b, 2);
            // Shuffle to [y, z, x, 0] using vext (shift by 1)
            float32x4_t a_yzx = vextq_f32(a, a, 1);
            float32x4_t b_yzx = vextq_f32(b, b, 1);
            // Shuffle to [z, x, y, 0] using vext (shift by 2)
            float32x4_t a_zxy = vextq_f32(a, a, 2);
            float32x4_t b_zxy = vextq_f32(b, b, 2);
            // result = a_yzx * b_zxy - a_zxy * b_yzx
            float32x4_t r = vmulq_f32(a_yzx, b_zxy);
            r = vfmsq_f32(r, a_zxy, b_yzx);  // FMA: r = r - (a_zxy * b_yzx)
            vst1q_lane_f32(&result.data[0], r, 0);
            vst1q_lane_f32(&result.data[1], r, 1);
            vst1q_lane_f32(&result.data[2], r, 2);
            return result;
        }
#endif
        Vec result;
        result.data[X] = data[Y] * other.data[Z] - data[Z] * other.data[Y];
        result.data[Y] = data[Z] * other.data[X] - data[X] * other.data[Z];
        result.data[Z] = data[X] * other.data[Y] - data[Y] * other.data[X];
        return result;
    }

    /// \brief Dot product operator.
    /// \param other The other vector.
    /// \return The dot product.
    MATRIX_CONSTEXPR T operator|(const Vec& other) const noexcept { return dot(other); }

    /// \brief Cross product operator (only for 3D).
    /// \param other The other vector.
    /// \return The cross product.
    template<std::uint32_t NN = N>
    MATRIX_CONSTEXPR MATRIX_NODISCARD typename std::enable_if<NN == 3, Vec>::type
    operator^(const Vec& other) const noexcept
    {
        return cross(other);
    }

    /// \brief Length of the vector.
    /// \return The Euclidean length.
    T length() const noexcept { return std::sqrt(dot(*this)); }

    /// \brief Normalized vector.
    /// \return The unit vector in the same direction.
    Vec normalized() const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && (N == 2 || N == 3 || N == 4))
        {
            MATRIX_LIKELY
            const T len_sq = dot(*this);
            if (len_sq == T(0))
                return *this;

            // Fast inverse square root with Newton-Raphson refinement
            float32x2_t len_sq_v = vdup_n_f32(static_cast<float>(len_sq));
            float32x2_t rsqrt = vrsqrte_f32(len_sq_v);  // Initial estimate
            // Newton-Raphson: x' = x * (3 - x^2 * a) / 2
            rsqrt = vmul_f32(rsqrt, vrsqrts_f32(vmul_f32(len_sq_v, rsqrt), rsqrt));
            rsqrt = vmul_f32(rsqrt, vrsqrts_f32(vmul_f32(len_sq_v, rsqrt), rsqrt));
            const T inv_len = static_cast<T>(vget_lane_f32(rsqrt, 0));
            return *this * inv_len;
        }
#endif
        const T len = length();
        if (len == T(0))
            return *this;
        return *this / len;
    }

    /// \brief Angle between two vectors (in radians).
    /// \param other The other vector.
    /// \return The angle in radians.
    T angle(const Vec& other) const noexcept
    {
        const T len1 = length();
        const T len2 = other.length();
        const T denom = len1 * len2;
        if (denom == T(0))
            return T(0);
        T cos_theta = dot(other) / denom;
        // Clamp to [-1, 1] to avoid domain errors
        if (cos_theta > T(1))
            cos_theta = T(1);
        if (cos_theta < T(-1))
            cos_theta = T(-1);
        return std::acos(cos_theta);
    }

    /// \brief Get x component (index 0).
    /// \return Reference to x component.
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 1), T&>::type x() noexcept
    {
        return data[X];
    }
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 1), const T&>::type x() const noexcept
    {
        return data[X];
    }

    /// \brief Get y component (index 1).
    /// \return Reference to y component.
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 2), T&>::type y() noexcept
    {
        return data[Y];
    }
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 2), const T&>::type y() const noexcept
    {
        return data[Y];
    }

    /// \brief Get z component (index 2).
    /// \return Reference to z component.
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 3), T&>::type z() noexcept
    {
        return data[Z];
    }
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 3), const T&>::type z() const noexcept
    {
        return data[Z];
    }

    /// \brief Get w component (index 3).
    /// \return Reference to w component.
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 4), T&>::type w() noexcept
    {
        return data[W];
    }
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 4), const T&>::type w() const noexcept
    {
        return data[W];
    }

    /// \brief Approximate equality comparison.
    /// \param other The vector to compare.
    /// \param epsilon The tolerance (defaults to machine epsilon for type T).
    /// \return True if approximately equal.
    bool approx_equal(const Vec& other, T epsilon = std::numeric_limits<T>::epsilon()) const noexcept
    {
        for (std::uint32_t i = 0; i < N; ++i)
        {
            if (std::abs(data[i] - other.data[i]) > epsilon)
            {
                return false;
            }
        }
        return true;
    }

    /// \brief Linear interpolation between two vectors.
    /// \param other The target vector.
    /// \param t The interpolation parameter [0, 1].
    /// \return The interpolated vector.
    Vec lerp(const Vec& other, T t) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] + t * (other.data[i] - data[i]);
        }
        return result;
    }

    /// \brief Cubic Hermite interpolation between two vectors with tangents
    /// \param other The target vector
    /// \param tangent1 The tangent at this vector
    /// \param tangent2 The tangent at the other vector
    /// \param t The interpolation parameter [0, 1]
    /// \return The interpolated vector
    Vec cubic_hermite(const Vec& other, const Vec& tangent1, const Vec& tangent2, T t) const noexcept
    {
        const T t2 = t * t;
        const T t3 = t2 * t;
        const T h00 = T(2) * t3 - T(3) * t2 + T(1);
        const T h10 = t3 - T(2) * t2 + t;
        const T h01 = T(-2) * t3 + T(3) * t2;
        const T h11 = t3 - t2;

        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = h00 * data[i] + h10 * tangent1.data[i] + h01 * other.data[i] + h11 * tangent2.data[i];
        }
        return result;
    }

    /// \brief Catmull-Rom spline interpolation (passes through control points)
    /// \param p0 Point before this vector
    /// \param p2 Point after other vector
    /// \param other The target vector (p1)
    /// \param t The interpolation parameter [0, 1]
    /// \return The interpolated vector
    Vec catmull_rom(const Vec& other, const Vec& p0, const Vec& p2, T t) const noexcept
    {
        const T t2 = t * t;
        const T t3 = t2 * t;

        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = T(0.5) * ((T(2) * data[i]) + (-p0.data[i] + other.data[i]) * t +
                                       (T(2) * p0.data[i] - T(5) * data[i] + T(4) * other.data[i] - p2.data[i]) * t2 +
                                       (-p0.data[i] + T(3) * data[i] - T(3) * other.data[i] + p2.data[i]) * t3);
        }
        return result;
    }

    /// \brief Element-wise (Hadamard) product.
    /// \param other The other vector.
    /// \return The element-wise product.
    Vec hadamard(const Vec& other) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    /// \brief Get minimum element.
    /// \return The minimum element value.
    T min_element() const noexcept
    {
        T min_val = data[X];
        for (std::uint32_t i = 1; i < N; ++i)
        {
            if (data[i] < min_val)
                min_val = data[i];
        }
        return min_val;
    }

    /// \brief Get maximum element.
    /// \return The maximum element value.
    T max_element() const noexcept
    {
        T max_val = data[X];
        for (std::uint32_t i = 1; i < N; ++i)
        {
            if (data[i] > max_val)
                max_val = data[i];
        }
        return max_val;
    }

    /// \brief Element-wise absolute value.
    /// \return Vector with absolute values.
    Vec abs() const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = std::abs(data[i]);
        }
        return result;
    }

    /// \brief Clamp vector elements between min and max.
    /// \param min_val The minimum value.
    /// \param max_val The maximum value.
    /// \return The clamped vector.
    Vec clamp(T min_val, T max_val) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = std::min(std::max(data[i], min_val), max_val);
        }
        return result;
    }

    /// \brief Element-wise minimum with another vector.
    /// \param other The other vector.
    /// \return Vector with element-wise minimums.
    Vec min(const Vec& other) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = std::min(data[i], other.data[i]);
        }
        return result;
    }

    /// \brief Element-wise maximum with another vector.
    /// \param other The other vector.
    /// \return Vector with element-wise maximums.
    Vec max(const Vec& other) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = std::max(data[i], other.data[i]);
        }
        return result;
    }

    /// \brief Project this vector onto another vector.
    /// \param other The vector to project onto.
    /// \return The projection of this vector onto other.
    Vec project(const Vec& other) const noexcept
    {
        const T dot_prod = dot(other);
        const T other_len_sq = other.dot(other);
        if (other_len_sq == T(0))
            return Vec();
        return other * (dot_prod / other_len_sq);
    }

    /// \brief Reject this vector from another (perpendicular component).
    /// \param other The vector to reject from.
    /// \return The component of this vector perpendicular to other.
    Vec reject(const Vec& other) const noexcept { return *this - project(other); }

    /// \brief Signed angle between two 3D vectors relative to a normal axis.
    /// \param other The other vector.
    /// \param normal The normal vector defining the plane (should be normalized).
    /// \return The signed angle in radians [-π, π].
    template<std::uint32_t NN = N>
    typename std::enable_if<NN == 3, T>::type signed_angle(const Vec& other, const Vec& normal) const noexcept
    {
        T cos_theta = dot(other) / (length() * other.length());
        // Clamp to [-1, 1] to avoid domain errors
        if (cos_theta > T(1))
            cos_theta = T(1);
        if (cos_theta < T(-1))
            cos_theta = T(-1);
        T angle = std::acos(cos_theta);
        // Determine sign using the normal
        Vec cross_prod = cross(other);
        if (cross_prod.dot(normal) < T(0))
        {
            angle = -angle;
        }
        return angle;
    }

    /// \brief Rotate this vector around an axis by an angle (Rodrigues' rotation formula).
    /// \param axis The rotation axis (should be normalized).
    /// \param angle The rotation angle in radians.
    /// \return The rotated vector.
    template<std::uint32_t NN = N>
    typename std::enable_if<NN == 3, Vec>::type rotate(const Vec& axis, T angle) const noexcept
    {
        const T cos_a = std::cos(angle);
        const T sin_a = std::sin(angle);
        // Rodrigues' formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        const Vec k_cross_v = axis.cross(*this);
        const T k_dot_v = axis.dot(*this);
        Vec result;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            result.data[i] = data[i] * cos_a + k_cross_v.data[i] * sin_a + axis.data[i] * k_dot_v * (T(1) - cos_a);
        }
        return result;
    }

    /// \brief Static factory: create a zero vector
    /// \return Zero vector with all components set to 0
    static MATRIX_CONSTEXPR Vec zero() noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = T(0);
        }
        return result;
    }

    /// \brief Static factory: create a vector with all components set to 1
    /// \return Vector with all components set to 1
    static MATRIX_CONSTEXPR Vec one() noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = T(1);
        }
        return result;
    }

    /// \brief Static factory: create unit vector along X axis (1, 0, 0, ...)
    /// \return Unit X vector
    static Vec unit_x() noexcept
    {
        Vec result = zero();
        result.data[X] = T(1);
        return result;
    }

    /// \brief Static factory: create unit vector along Y axis (0, 1, 0, ...)
    /// \return Unit Y vector
    template<std::uint32_t NN = N>
    static typename std::enable_if<(NN >= 2), Vec>::type unit_y() noexcept
    {
        Vec result = zero();
        result.data[Y] = T(1);
        return result;
    }

    /// \brief Static factory: create unit vector along Z axis (0, 0, 1, ...)
    /// \return Unit Z vector
    template<std::uint32_t NN = N>
    static typename std::enable_if<(NN >= 3), Vec>::type unit_z() noexcept
    {
        Vec result = zero();
        result.data[Z] = T(1);
        return result;
    }

    /// \brief Static factory: create unit vector along W axis (0, 0, 0, 1)
    /// \return Unit W vector
    template<std::uint32_t NN = N>
    static typename std::enable_if<(NN >= 4), Vec>::type unit_w() noexcept
    {
        Vec result = zero();
        result.data[W] = T(1);
        return result;
    }

    /// \brief Safe normalization (returns zero vector if magnitude is too small)
    /// \param epsilon Threshold below which to return zero vector (defaults to machine epsilon).
    /// \return Normalized vector, or zero vector if length < epsilon
    Vec safe_normalized(T epsilon = std::numeric_limits<T>::epsilon()) const noexcept
    {
        const T len = length();
        if (len < epsilon)
            return zero();
        return *this / len;
    }

    /// \brief Clamp all components to range [min, max]
    /// \param min Minimum value
    /// \param max Maximum value
    /// \return Vector with clamped components
    Vec clamped(T min, T max) const noexcept
    {
        Vec result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            if (data[i] < min)
                result.data[i] = min;
            else if (data[i] > max)
                result.data[i] = max;
            else
                result.data[i] = data[i];
        }
        return result;
    }

    /// \brief Saturate all components to range [0, 1]
    /// \return Vector with saturated components
    Vec saturated() const noexcept { return clamped(T(0), T(1)); }

    /// \brief Get 2D swizzle (only for N >= 2)
    /// \return Vec<T,2> with x,y components
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 2), Vec<T, 2>>::type xy() const noexcept
    {
        return Vec<T, 2>(data[X], data[Y]);
    }

    /// \brief Get 3D swizzle (only for N >= 3)
    /// \return Vec<T,3> with x,y,z components
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 3), Vec<T, 3>>::type xyz() const noexcept
    {
        return Vec<T, 3>(data[X], data[Y], data[Z]);
    }

    /// \brief Get xz swizzle (only for N >= 3)
    /// \return Vec<T,2> with x,z components
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 3), Vec<T, 2>>::type xz() const noexcept
    {
        return Vec<T, 2>(data[X], data[Z]);
    }

    /// \brief Get yz swizzle (only for N >= 3)
    /// \return Vec<T,2> with y,z components
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 3), Vec<T, 2>>::type yz() const noexcept
    {
        return Vec<T, 2>(data[Y], data[Z]);
    }

    /// \brief Convert to homogeneous coordinates (add w=1 component)
    /// \return Vec<T,N+1> with w=1 appended
    Vec<T, N + 1> to_homogeneous() const noexcept
    {
        Vec<T, N + 1> result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            result.data[i] = data[i];
        }
        result.data[N] = T(1);
        return result;
    }

    /// \brief Convert from homogeneous coordinates (divide by w, drop w component)
    /// \return Vec<T,N-1> with perspective division applied
    template<std::uint32_t NN = N>
    typename std::enable_if<(NN >= 2), Vec<T, N - 1>>::type from_homogeneous() const noexcept
    {
        Vec<T, N - 1> result;
        const T w = data[N - 1];
        if (std::abs(w) > std::numeric_limits<T>::epsilon())
        {
            const T inv_w = T(1) / w;
            for (std::uint32_t i = 0; i < N - 1; ++i)
            {
                result.data[i] = data[i] * inv_w;
            }
        }
        else
        {
            // w is too small, return as-is without division
            for (std::uint32_t i = 0; i < N - 1; ++i)
            {
                result.data[i] = data[i];
            }
        }
        return result;
    }
};

/// \brief Scalar multiplication (commutative).
/// \tparam T The type.
/// \tparam N The dimension.
/// \param scalar The scalar.
/// \param v The vector.
/// \return The result vector.
template<typename T, std::uint32_t N>
MATRIX_CONSTEXPR MATRIX_NODISCARD Vec<T, N> operator*(T scalar, const Vec<T, N>& v) noexcept
{
    return v * scalar;
}

}  // namespace matrixlib

#endif  // _MATRIXLIB_VECTOR_HPP_
