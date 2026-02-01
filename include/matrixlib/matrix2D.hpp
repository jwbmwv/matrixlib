// SPDX-License-Identifier: MIT
/// @file matrix2D.hpp
/// @brief 2D matrix specializations and extensions
/// @details This header provides 2D-specific matrix functionality including
///          rotation matrices for 2D transformations.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#pragma once

#ifndef _MATRIXLIB_MATRIX2D_HPP_
#define _MATRIXLIB_MATRIX2D_HPP_

#include "compiler_features.hpp"
#include "matrix.hpp"

namespace matrixlib
{

// 2D matrix type aliases for common types
template<typename T>
using Mat2 = SquareMat<T, 2>;

using Mat2f = SquareMat<float, 2>;
using Mat2d = SquareMat<double, 2>;

/// \brief 2D matrix extensions
/// \details These methods are only available for 2x2 matrices (N=2)
template<typename T>
class Mat2DExtensions
{
public:
    /// \brief Create a 2D rotation matrix.
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static MATRIX_CONSTEXPR_TRIG SquareMat<T, 2> rotation(T angle) noexcept
    {
        SquareMat<T, 2> result;
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        result.data[(0 * 2) + 0] = c;
        result.data[(0 * 2) + 1] = -s;
        result.data[(1 * 2) + 0] = s;
        result.data[(1 * 2) + 1] = c;
        return result;
    }

    /// \brief Create compile-time 2D rotation matrix for special angles (C++11-C++23)
    /// \tparam angle_deg Angle in degrees (must be 0, 90, 180, or 270)
    /// \return The rotation matrix (constexpr)
    /// \note For C++26+, use rotation() directly with constexpr context
    template<int angle_deg>
    static MATRIX_CONSTEXPR SquareMat<T, 2> rotation_deg() noexcept
    {
        static_assert(angle_deg == 0 || angle_deg == 90 || angle_deg == 180 || angle_deg == 270,
                      "Compile-time rotation only supports 0, 90, 180, 270 degrees");
        SquareMat<T, 2> result = SquareMat<T, 2>::identity();
        if (angle_deg == 90)
        {
            result.data[0] = T(0);
            result.data[1] = T(-1);
            result.data[2] = T(1);
            result.data[3] = T(0);
        }
        else if (angle_deg == 180)
        {
            result.data[0] = T(-1);
            result.data[1] = T(0);
            result.data[2] = T(0);
            result.data[3] = T(-1);
        }
        else if (angle_deg == 270)
        {
            result.data[0] = T(0);
            result.data[1] = T(1);
            result.data[2] = T(-1);
            result.data[3] = T(0);
        }
        return result;
    }
};

// Add 2D-specific methods to SquareMat<T,2> via template specialization
template<typename T>
class SquareMat<T, 2> : public Mat<T, 2, 2>
{
public:
    using Mat<T, 2, 2>::Mat;

    /// \brief Trace of the matrix (sum of diagonal elements).
    /// \return The trace.
    MATRIX_CONSTEXPR T trace() const noexcept { return this->data[0] + this->data[3]; }

    /// \brief Create an identity matrix.
    /// \return The identity matrix.
    static MATRIX_CONSTEXPR SquareMat<T, 2> identity()
    {
        SquareMat<T, 2> result;
        result.data[0] = T(1);
        result.data[1] = T(0);
        result.data[2] = T(0);
        result.data[3] = T(1);
        return result;
    }

    /// \brief Rank of the matrix (number of linearly independent rows/columns).
    /// \return The rank.
    std::uint32_t rank() const
    {
        Mat<T, 2, 2> temp = *this;
        std::uint32_t rank = 0;
        // Use epsilon for floating point, zero for integers
        const T epsilon = std::is_floating_point<T>::value ? std::numeric_limits<T>::epsilon() * T(100) : T(0);
        for (std::uint32_t i = 0; i < 2; ++i)
        {
            // Find pivot
            std::uint32_t max_row = i;
            for (std::uint32_t k = i; k < 2; ++k)
            {
                if (std::abs(temp.data[(k * 2) + i]) > std::abs(temp.data[(max_row * 2) + i]))
                {
                    max_row = k;
                }
            }
            if (std::abs(temp.data[(max_row * 2) + i]) <= epsilon)
            {
                continue;
            }
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < 2; ++k)
                {
                    T swap_temp = temp.data[(i * 2) + k];
                    temp.data[(i * 2) + k] = temp.data[(max_row * 2) + k];
                    temp.data[(max_row * 2) + k] = swap_temp;
                }
            }
            ++rank;
            for (std::uint32_t k = i + 1; k < 2; ++k)
            {
                T factor = temp.data[(k * 2) + i] / temp.data[(i * 2) + i];
                for (std::uint32_t j = i; j < 2; ++j)
                {
                    temp.data[(k * 2) + j] -= factor * temp.data[(i * 2) + j];
                }
            }
        }
        return rank;
    }

    /// \brief Determinant of the matrix.
    /// \return The determinant.
    T determinant() const { return this->data[0] * this->data[3] - this->data[1] * this->data[2]; }

    /// \brief Inverse of the matrix.
    /// \return The inverse matrix.
    /// \note Returns identity matrix if determinant is near zero (singular matrix).
    SquareMat inverse() const
    {
        const T det = determinant();
        // Check for near-zero determinant to avoid division by zero
        const T epsilon = std::is_floating_point<T>::value ? std::numeric_limits<T>::epsilon() * T(100) : T(0);
        if (std::abs(det) <= epsilon)
        {
            return SquareMat::identity();  // Return identity for singular matrix
        }
        SquareMat result;
        result.data[0] = this->data[3] / det;
        result.data[1] = -this->data[1] / det;
        result.data[2] = -this->data[2] / det;
        result.data[3] = this->data[0] / det;
        return result;
    }

    /// \brief Create a 2D rotation matrix.
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static MATRIX_CONSTEXPR_TRIG SquareMat rotation(T angle) noexcept { return Mat2DExtensions<T>::rotation(angle); }

    /// \brief Create compile-time 2D rotation matrix for special angles (C++11-C++23)
    /// \tparam angle_deg Angle in degrees (must be 0, 90, 180, or 270)
    /// \return The rotation matrix (constexpr)
    /// \note For C++26+, use rotation() directly with constexpr context
    template<int angle_deg>
    static MATRIX_CONSTEXPR SquareMat rotation_deg() noexcept
    {
        return Mat2DExtensions<T>::template rotation_deg<angle_deg>();
    }

    /// \brief Create a scale matrix.
    /// \param scale The scale factors (one per axis).
    /// \return The scale matrix.
    static SquareMat scale(const Vec<T, 2>& scale_vec) noexcept
    {
        SquareMat result;
        result.data[0] = scale_vec.data[0];
        result.data[1] = T(0);
        result.data[2] = T(0);
        result.data[3] = scale_vec.data[1];
        return result;
    }

    /// \brief Create a uniform scale matrix.
    /// \param s The uniform scale factor.
    /// \return The scale matrix.
    static SquareMat scale(T s) noexcept
    {
        SquareMat result;
        result.data[0] = s;
        result.data[1] = T(0);
        result.data[2] = T(0);
        result.data[3] = s;
        return result;
    }
};

}  // namespace matrixlib

#endif  // _MATRIXLIB_MATRIX2D_HPP_
