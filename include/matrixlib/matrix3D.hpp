// SPDX-License-Identifier: MIT
/// @file matrix3D.hpp
/// @brief 3D matrix specializations and extensions
/// @details This header provides 3D-specific matrix functionality including
///          rotation matrices, Euler angles, and 4x4 transformation matrices.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#pragma once

#ifndef _MATRIXLIB_MATRIX3D_HPP_
#define _MATRIXLIB_MATRIX3D_HPP_

#include "compiler_features.hpp"
#include "matrix.hpp"
#include "vec3D.hpp"

namespace matrixlib
{

// 3D matrix type aliases for common types
template<typename T>
using Mat3 = SquareMat<T, 3>;

template<typename T>
using Mat4 = SquareMat<T, 4>;

using Mat3f = SquareMat<float, 3>;
using Mat3d = SquareMat<double, 3>;
using Mat4f = SquareMat<float, 4>;
using Mat4d = SquareMat<double, 4>;

/// \brief 3D matrix extensions for SquareMat<T,3>
template<typename T>
class SquareMat<T, 3> : public Mat<T, 3, 3>
{
public:
    using Mat<T, 3, 3>::Mat;

    /// \brief Trace of the matrix (sum of diagonal elements).
    /// \return The trace.
    MATRIX_CONSTEXPR T trace() const noexcept { return this->data[0] + this->data[4] + this->data[8]; }

    /// \brief Create an identity matrix.
    /// \return The identity matrix.
    static MATRIX_CONSTEXPR SquareMat<T, 3> identity()
    {
        SquareMat<T, 3> result;
        result.data[0] = T(1);
        result.data[1] = T(0);
        result.data[2] = T(0);
        result.data[3] = T(0);
        result.data[4] = T(1);
        result.data[5] = T(0);
        result.data[6] = T(0);
        result.data[7] = T(0);
        result.data[8] = T(1);
        return result;
    }

    /// \brief Rank of the matrix (number of linearly independent rows/columns).
    /// \return The rank.
    std::uint32_t rank() const
    {
        Mat<T, 3, 3> temp = *this;
        std::uint32_t rank = 0;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            std::uint32_t max_row = i;
            for (std::uint32_t k = i; k < 3; ++k)
            {
                if (std::abs(temp.data[(k * 3) + i]) > std::abs(temp.data[(max_row * 3) + i]))
                {
                    max_row = k;
                }
            }
            if (temp.data[(max_row * 3) + i] == T(0))
            {
                continue;
            }
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < 3; ++k)
                {
                    T swap_temp = temp.data[(i * 3) + k];
                    temp.data[(i * 3) + k] = temp.data[(max_row * 3) + k];
                    temp.data[(max_row * 3) + k] = swap_temp;
                }
            }
            ++rank;
            for (std::uint32_t k = i + 1; k < 3; ++k)
            {
                T factor = temp.data[(k * 3) + i] / temp.data[(i * 3) + i];
                for (std::uint32_t j = i; j < 3; ++j)
                {
                    temp.data[(k * 3) + j] -= factor * temp.data[(i * 3) + j];
                }
            }
        }
        return rank;
    }

    /// \brief Determinant of the matrix.
    /// \return The determinant.
    T determinant() const
    {
        return this->data[0] * (this->data[4] * this->data[8] - this->data[5] * this->data[7]) -
               this->data[1] * (this->data[3] * this->data[8] - this->data[5] * this->data[6]) +
               this->data[2] * (this->data[3] * this->data[7] - this->data[4] * this->data[6]);
    }

    /// \brief Inverse of the matrix.
    /// \return The inverse matrix.
    /// \note Returns an undefined matrix if determinant is zero (singular matrix).
    ///       Check determinant() before calling for safety.
    SquareMat inverse() const
    {
        const T det = determinant();
        SquareMat result;

        result.data[0] = (this->data[4] * this->data[8] - this->data[5] * this->data[7]) / det;
        result.data[1] = (this->data[2] * this->data[7] - this->data[1] * this->data[8]) / det;
        result.data[2] = (this->data[1] * this->data[5] - this->data[2] * this->data[4]) / det;

        result.data[3] = (this->data[5] * this->data[6] - this->data[3] * this->data[8]) / det;
        result.data[4] = (this->data[0] * this->data[8] - this->data[2] * this->data[6]) / det;
        result.data[5] = (this->data[2] * this->data[3] - this->data[0] * this->data[5]) / det;

        result.data[6] = (this->data[3] * this->data[7] - this->data[4] * this->data[6]) / det;
        result.data[7] = (this->data[1] * this->data[6] - this->data[0] * this->data[7]) / det;
        result.data[8] = (this->data[0] * this->data[4] - this->data[1] * this->data[3]) / det;

        return result;
    }

    /// \brief Create a 3D rotation matrix around X axis.
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static MATRIX_CONSTEXPR_TRIG SquareMat rotation_x(T angle) noexcept
    {
        SquareMat result = identity();
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        result.data[(1 * 3) + 1] = c;
        result.data[(1 * 3) + 2] = -s;
        result.data[(2 * 3) + 1] = s;
        result.data[(2 * 3) + 2] = c;
        return result;
    }

    /// \brief Create a 3D rotation matrix around Y axis.
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static MATRIX_CONSTEXPR_TRIG SquareMat rotation_y(T angle) noexcept
    {
        SquareMat result = identity();
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        result.data[(0 * 3) + 0] = c;
        result.data[(0 * 3) + 2] = s;
        result.data[(2 * 3) + 0] = -s;
        result.data[(2 * 3) + 2] = c;
        return result;
    }

    /// \brief Create a 3D rotation matrix around Z axis.
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static MATRIX_CONSTEXPR_TRIG SquareMat rotation_z(T angle) noexcept
    {
        SquareMat result = identity();
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        result.data[(0 * 3) + 0] = c;
        result.data[(0 * 3) + 1] = -s;
        result.data[(1 * 3) + 0] = s;
        result.data[(1 * 3) + 1] = c;
        return result;
    }

    /// \brief Create compile-time 3D rotation matrix around X axis for special angles (C++11-C++23)
    /// \tparam angle_deg Angle in degrees (must be 0, 90, 180, or 270)
    /// \return The rotation matrix (constexpr)
    /// \note For C++26+, use rotation_x() directly with constexpr context
    template<int angle_deg>
    static MATRIX_CONSTEXPR SquareMat rotation_x_deg() noexcept
    {
        static_assert(angle_deg == 0 || angle_deg == 90 || angle_deg == 180 || angle_deg == 270,
                      "Compile-time rotation only supports 0, 90, 180, 270 degrees");
        SquareMat result = identity();
        if (angle_deg == 90)
        {
            result.data[(1 * 3) + 1] = T(0);
            result.data[(1 * 3) + 2] = T(-1);
            result.data[(2 * 3) + 1] = T(1);
            result.data[(2 * 3) + 2] = T(0);
        }
        else if (angle_deg == 180)
        {
            result.data[(1 * 3) + 1] = T(-1);
            result.data[(1 * 3) + 2] = T(0);
            result.data[(2 * 3) + 1] = T(0);
            result.data[(2 * 3) + 2] = T(-1);
        }
        else if (angle_deg == 270)
        {
            result.data[(1 * 3) + 1] = T(0);
            result.data[(1 * 3) + 2] = T(1);
            result.data[(2 * 3) + 1] = T(-1);
            result.data[(2 * 3) + 2] = T(0);
        }
        return result;
    }

    /// \brief Create compile-time 3D rotation matrix around Y axis for special angles (C++11-C++23)
    /// \tparam angle_deg Angle in degrees (must be 0, 90, 180, or 270)
    /// \return The rotation matrix (constexpr)
    /// \note For C++26+, use rotation_y() directly with constexpr context
    template<int angle_deg>
    static MATRIX_CONSTEXPR SquareMat rotation_y_deg() noexcept
    {
        static_assert(angle_deg == 0 || angle_deg == 90 || angle_deg == 180 || angle_deg == 270,
                      "Compile-time rotation only supports 0, 90, 180, 270 degrees");
        SquareMat result = identity();
        if (angle_deg == 90)
        {
            result.data[(0 * 3) + 0] = T(0);
            result.data[(0 * 3) + 2] = T(1);
            result.data[(2 * 3) + 0] = T(-1);
            result.data[(2 * 3) + 2] = T(0);
        }
        else if (angle_deg == 180)
        {
            result.data[(0 * 3) + 0] = T(-1);
            result.data[(0 * 3) + 2] = T(0);
            result.data[(2 * 3) + 0] = T(0);
            result.data[(2 * 3) + 2] = T(-1);
        }
        else if (angle_deg == 270)
        {
            result.data[(0 * 3) + 0] = T(0);
            result.data[(0 * 3) + 2] = T(-1);
            result.data[(2 * 3) + 0] = T(1);
            result.data[(2 * 3) + 2] = T(0);
        }
        return result;
    }

    /// \brief Create compile-time 3D rotation matrix around Z axis for special angles (C++11-C++23)
    /// \tparam angle_deg Angle in degrees (must be 0, 90, 180, or 270)
    /// \return The rotation matrix (constexpr)
    /// \note For C++26+, use rotation_z() directly with constexpr context
    template<int angle_deg>
    static MATRIX_CONSTEXPR SquareMat rotation_z_deg() noexcept
    {
        static_assert(angle_deg == 0 || angle_deg == 90 || angle_deg == 180 || angle_deg == 270,
                      "Compile-time rotation only supports 0, 90, 180, 270 degrees");
        SquareMat result = identity();
        if (angle_deg == 90)
        {
            result.data[(0 * 3) + 0] = T(0);
            result.data[(0 * 3) + 1] = T(-1);
            result.data[(1 * 3) + 0] = T(1);
            result.data[(1 * 3) + 1] = T(0);
        }
        else if (angle_deg == 180)
        {
            result.data[(0 * 3) + 0] = T(-1);
            result.data[(0 * 3) + 1] = T(0);
            result.data[(1 * 3) + 0] = T(0);
            result.data[(1 * 3) + 1] = T(-1);
        }
        else if (angle_deg == 270)
        {
            result.data[(0 * 3) + 0] = T(0);
            result.data[(0 * 3) + 1] = T(1);
            result.data[(1 * 3) + 0] = T(-1);
            result.data[(1 * 3) + 1] = T(0);
        }
        return result;
    }

    /// \brief Create a 3x3 rotation matrix that rotates from one vector to another.
    /// \param from The source vector (will be normalized).
    /// \param to The target vector (will be normalized).
    /// \return The rotation matrix.
    static SquareMat rotation_from_to(const Vec<T, 3>& from, const Vec<T, 3>& to) noexcept
    {
        const Vec<T, 3> v1 = from.normalized();
        const Vec<T, 3> v2 = to.normalized();
        const T dot_prod = v1.dot(v2);

        // Check if vectors are parallel
        if (dot_prod > T(0.999999))
        {
            return identity();  // Already aligned
        }
        else if (dot_prod < T(-0.999999))
        {
            // Opposite directions - rotate 180° around any perpendicular axis
            Vec<T, 3> axis;
            if (std::abs(v1.data[0]) < T(0.9))
            {
                axis = Vec<T, 3>(T(1), T(0), T(0)).cross(v1).normalized();
            }
            else
            {
                axis = Vec<T, 3>(T(0), T(1), T(0)).cross(v1).normalized();
            }
            return rotation_axis_angle(axis, constants::pi<T>);
        }

        const Vec<T, 3> axis = v1.cross(v2);
        const T s = axis.length();
        const T c = dot_prod;

        // Skew-symmetric cross-product matrix
        SquareMat K;
        K.data[(0 * 3) + 0] = T(0);
        K.data[(0 * 3) + 1] = -axis.data[2];
        K.data[(0 * 3) + 2] = axis.data[1];
        K.data[(1 * 3) + 0] = axis.data[2];
        K.data[(1 * 3) + 1] = T(0);
        K.data[(1 * 3) + 2] = -axis.data[0];
        K.data[(2 * 3) + 0] = -axis.data[1];
        K.data[(2 * 3) + 1] = axis.data[0];
        K.data[(2 * 3) + 2] = T(0);

        // Rodrigues formula: R = I + K + K^2 * (1-c)/s^2
        SquareMat K2 = K;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            for (std::uint32_t j = 0; j < 3; ++j)
            {
                T sum = T(0);
                for (std::uint32_t k = 0; k < 3; ++k)
                {
                    sum += K.data[(i * 3) + k] * K.data[(k * 3) + j];
                }
                K2.data[(i * 3) + j] = sum;
            }
        }

        SquareMat result = identity();
        const T factor = (T(1) - c) / (s * s);
        for (std::uint32_t i = 0; i < 9; ++i)
        {
            result.data[i] += K.data[i] + K2.data[i] * factor;
        }
        return result;
    }

    /// \brief Create a 3x3 rotation matrix from axis and angle (Rodrigues formula).
    /// \param axis The rotation axis (should be normalized).
    /// \param angle The rotation angle in radians.
    /// \return The rotation matrix.
    static SquareMat rotation_axis_angle(const Vec<T, 3>& axis, T angle) noexcept
    {
        const T c = std::cos(angle);
        const T s = std::sin(angle);
        const T t = T(1) - c;

        const T x = axis.data[0];
        const T y = axis.data[1];
        const T z = axis.data[2];

        SquareMat result;
        result.data[(0 * 3) + 0] = t * x * x + c;
        result.data[(0 * 3) + 1] = t * x * y - s * z;
        result.data[(0 * 3) + 2] = t * x * z + s * y;
        result.data[(1 * 3) + 0] = t * x * y + s * z;
        result.data[(1 * 3) + 1] = t * y * y + c;
        result.data[(1 * 3) + 2] = t * y * z - s * x;
        result.data[(2 * 3) + 0] = t * x * z - s * y;
        result.data[(2 * 3) + 1] = t * y * z + s * x;
        result.data[(2 * 3) + 2] = t * z * z + c;

        return result;
    }

    /// \brief Create a 3x3 look-at rotation matrix aligned to a direction.
    /// \param direction The forward direction (will be normalized).
    /// \param up The up direction hint (will be normalized, default is +Z).
    /// \return The rotation matrix with Z-axis pointing in direction.
    static SquareMat look_at(const Vec<T, 3>& direction, const Vec<T, 3>& up = Vec<T, 3>(T(0), T(0), T(1))) noexcept
    {
        const Vec<T, 3> z_axis = direction.normalized();
        const Vec<T, 3> x_axis = up.cross(z_axis).normalized();
        const Vec<T, 3> y_axis = z_axis.cross(x_axis);

        SquareMat result;
        result.data[(0 * 3) + 0] = x_axis.data[0];
        result.data[(0 * 3) + 1] = x_axis.data[1];
        result.data[(0 * 3) + 2] = x_axis.data[2];
        result.data[(1 * 3) + 0] = y_axis.data[0];
        result.data[(1 * 3) + 1] = y_axis.data[1];
        result.data[(1 * 3) + 2] = y_axis.data[2];
        result.data[(2 * 3) + 0] = z_axis.data[0];
        result.data[(2 * 3) + 1] = z_axis.data[1];
        result.data[(2 * 3) + 2] = z_axis.data[2];

        return result;
    }

    /// \brief Extract Euler angles (roll, pitch, yaw) from a 3x3 rotation matrix.
    /// \return Vec3 containing (roll, pitch, yaw) in radians.
    /// Roll (X-axis rotation), Pitch (Y-axis rotation), Yaw (Z-axis rotation).
    Vec<T, 3> euler_angles() const noexcept
    {
        Vec<T, 3> result;

        // Extract pitch (rotation around Y-axis)
        const T sin_pitch = -this->data[(2 * 3) + 0];
        if (sin_pitch >= T(1))
        {
            result.data[1] = T(1.5707963267948966);  // π/2
            result.data[0] = T(0);
            result.data[2] = std::atan2(-this->data[(0 * 3) + 1], this->data[(1 * 3) + 1]);
        }
        else if (sin_pitch <= T(-1))
        {
            result.data[1] = T(-1.5707963267948966);  // -π/2
            result.data[0] = T(0);
            result.data[2] = std::atan2(-this->data[(0 * 3) + 1], this->data[(1 * 3) + 1]);
        }
        else
        {
            result.data[1] = std::asin(sin_pitch);
            result.data[0] = std::atan2(this->data[(2 * 3) + 1], this->data[(2 * 3) + 2]);  // Roll
            result.data[2] = std::atan2(this->data[(1 * 3) + 0], this->data[(0 * 3) + 0]);  // Yaw
        }

        return result;
    }

    /// \brief Create a scale matrix.
    /// \param scale The scale factors (one per axis).
    /// \return The scale matrix.
    static SquareMat scale(const Vec<T, 3>& scale_vec) noexcept
    {
        SquareMat result;
        result.data[0] = scale_vec.data[0];
        result.data[1] = T(0);
        result.data[2] = T(0);
        result.data[3] = T(0);
        result.data[4] = scale_vec.data[1];
        result.data[5] = T(0);
        result.data[6] = T(0);
        result.data[7] = T(0);
        result.data[8] = scale_vec.data[2];
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
        result.data[3] = T(0);
        result.data[4] = s;
        result.data[5] = T(0);
        result.data[6] = T(0);
        result.data[7] = T(0);
        result.data[8] = s;
        return result;
    }
};

/// \brief 4D matrix extensions for SquareMat<T,4>
template<typename T>
class SquareMat<T, 4> : public Mat<T, 4, 4>
{
public:
    using Mat<T, 4, 4>::Mat;

    /// \brief Trace of the matrix (sum of diagonal elements).
    /// \return The trace.
    MATRIX_CONSTEXPR T trace() const noexcept
    {
        return this->data[0] + this->data[5] + this->data[10] + this->data[15];
    }

    /// \brief Create an identity matrix.
    /// \return The identity matrix.
    static MATRIX_CONSTEXPR SquareMat<T, 4> identity()
    {
        SquareMat<T, 4> result;
        for (std::uint32_t i = 0; i < 16; ++i)
        {
            result.data[i] = T(0);
        }
        result.data[0] = T(1);
        result.data[5] = T(1);
        result.data[10] = T(1);
        result.data[15] = T(1);
        return result;
    }

    /// \brief Rank of the matrix (number of linearly independent rows/columns).
    /// \return The rank.
    std::uint32_t rank() const
    {
        Mat<T, 4, 4> temp = *this;
        std::uint32_t rank = 0;
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            std::uint32_t max_row = i;
            for (std::uint32_t k = i; k < 4; ++k)
            {
                if (std::abs(temp.data[(k * 4) + i]) > std::abs(temp.data[(max_row * 4) + i]))
                {
                    max_row = k;
                }
            }
            if (temp.data[(max_row * 4) + i] == T(0))
            {
                continue;
            }
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < 4; ++k)
                {
                    T swap_temp = temp.data[(i * 4) + k];
                    temp.data[(i * 4) + k] = temp.data[(max_row * 4) + k];
                    temp.data[(max_row * 4) + k] = swap_temp;
                }
            }
            ++rank;
            for (std::uint32_t k = i + 1; k < 4; ++k)
            {
                T factor = temp.data[(k * 4) + i] / temp.data[(i * 4) + i];
                for (std::uint32_t j = i; j < 4; ++j)
                {
                    temp.data[(k * 4) + j] -= factor * temp.data[(i * 4) + j];
                }
            }
        }
        return rank;
    }

    /// \brief Determinant of the matrix.
    /// \return The determinant.
    T determinant() const
    {
        Mat<T, 4, 4> temp = *this;
        T det = T(1);
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            std::uint32_t max_row = i;
            for (std::uint32_t k = i + 1; k < 4; ++k)
            {
                if (std::abs(temp.data[(k * 4) + i]) > std::abs(temp.data[(max_row * 4) + i]))
                {
                    max_row = k;
                }
            }
            if (temp.data[(max_row * 4) + i] == T(0))
            {
                return T(0);
            }
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < 4; ++k)
                {
                    T swap_temp = temp.data[(i * 4) + k];
                    temp.data[(i * 4) + k] = temp.data[(max_row * 4) + k];
                    temp.data[(max_row * 4) + k] = swap_temp;
                }
                det = -det;
            }
            det *= temp.data[(i * 4) + i];
            for (std::uint32_t k = i + 1; k < 4; ++k)
            {
                T factor = temp.data[(k * 4) + i] / temp.data[(i * 4) + i];
                for (std::uint32_t j = i; j < 4; ++j)
                {
                    temp.data[(k * 4) + j] -= factor * temp.data[(i * 4) + j];
                }
            }
        }
        return det;
    }

    /// \brief Inverse of the matrix.
    /// \return The inverse matrix.
    /// \note Returns an undefined matrix if the matrix is singular (determinant near zero).
    ///       For numerical stability, consider checking determinant() before calling.
    ///       Uses Gauss-Jordan elimination with partial pivoting.
    SquareMat inverse() const
    {
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            arm_matrix_instance_f32 src, dst;
            float inv_data[16];
            arm_mat_init_f32(&src, 4, 4, const_cast<float*>(reinterpret_cast<const float*>(this->data)));
            arm_mat_init_f32(&dst, 4, 4, inv_data);
            arm_status status = arm_mat_inverse_f32(&src, &dst);
            if (status == ARM_MATRIX_SUCCESS)
            {
                SquareMat result;
                for (std::uint32_t i = 0; i < 16; ++i)
                {
                    result.data[i] = static_cast<T>(inv_data[i]);
                }
                return result;
            }
            // Fall through to Gauss-Jordan if CMSIS fails
        }
#endif
        // Gauss-Jordan elimination with partial pivoting
        Mat<T, 4, 4> temp = *this;
        SquareMat result;

        for (std::uint32_t i = 0; i < 4; ++i)
        {
            for (std::uint32_t j = 0; j < 4; ++j)
            {
                result.data[(i * 4) + j] = (i == j) ? T(1) : T(0);
            }
        }

        for (std::uint32_t i = 0; i < 4; ++i)
        {
            // Partial pivoting: find row with largest pivot
            std::uint32_t max_row = i;
            for (std::uint32_t k = i + 1; k < 4; ++k)
            {
                if (std::abs(temp.data[(k * 4) + i]) > std::abs(temp.data[(max_row * 4) + i]))
                {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < 4; ++k)
                {
                    T swap_temp = temp.data[(i * 4) + k];
                    temp.data[(i * 4) + k] = temp.data[(max_row * 4) + k];
                    temp.data[(max_row * 4) + k] = swap_temp;

                    swap_temp = result.data[(i * 4) + k];
                    result.data[(i * 4) + k] = result.data[(max_row * 4) + k];
                    result.data[(max_row * 4) + k] = swap_temp;
                }
            }

            // Scale pivot row (note: no check for zero pivot - caller must ensure non-singular matrix)
            const T pivot = temp.data[(i * 4) + i];
            for (std::uint32_t k = 0; k < 4; ++k)
            {
                temp.data[(i * 4) + k] /= pivot;
                result.data[(i * 4) + k] /= pivot;
            }

            // Eliminate column
            for (std::uint32_t k = 0; k < 4; ++k)
            {
                if (k != i)
                {
                    const T factor = temp.data[(k * 4) + i];
                    for (std::uint32_t j = 0; j < 4; ++j)
                    {
                        temp.data[(k * 4) + j] -= factor * temp.data[(i * 4) + j];
                        result.data[(k * 4) + j] -= factor * result.data[(i * 4) + j];
                    }
                }
            }
        }

        return result;
    }

    /// \brief Create a 4x4 translation matrix.
    /// \param translation The translation vector (x, y, z).
    /// \return The 4x4 translation matrix.
    static SquareMat translation(const Vec<T, 3>& translation) noexcept
    {
        SquareMat result = identity();
        result.data[(0 * 4) + 3] = translation.data[0];  // x translation
        result.data[(1 * 4) + 3] = translation.data[1];  // y translation
        result.data[(2 * 4) + 3] = translation.data[2];  // z translation
        return result;
    }

    /// \brief Create a scale matrix.
    /// \param scale The scale factors (one per axis).
    /// \return The scale matrix.
    static SquareMat scale(const Vec<T, 4>& scale_vec) noexcept
    {
        SquareMat result;
        for (std::uint32_t i = 0; i < 16; ++i)
        {
            result.data[i] = T(0);
        }
        result.data[0] = scale_vec.data[0];
        result.data[5] = scale_vec.data[1];
        result.data[10] = scale_vec.data[2];
        result.data[15] = scale_vec.data[3];
        return result;
    }

    /// \brief Create a uniform scale matrix.
    /// \param s The uniform scale factor.
    /// \return The scale matrix.
    static SquareMat scale(T s) noexcept
    {
        SquareMat result;
        for (std::uint32_t i = 0; i < 16; ++i)
        {
            result.data[i] = T(0);
        }
        result.data[0] = s;
        result.data[5] = s;
        result.data[10] = s;
        result.data[15] = T(1);
        return result;
    }
};

/// \brief Transform a 3D vector by a 4x4 matrix.
/// \tparam T The type (float, double, etc.).
/// \param mat The 4x4 transformation matrix.
/// \param vec The 3D vector to transform.
/// \param w The homogeneous coordinate (0 for directions, 1 for points).
/// \return The transformed 3D vector.
/// \note This extends the Vec3 to Vec4 with the specified w, multiplies by the matrix,
///       and returns the result as Vec3 (discarding the w component).
///       Use w=0 for transforming directions (ignores translation).
///       Use w=1 for transforming points (includes translation).
template<typename T>
MATRIX_NODISCARD Vec<T, 3> transform_vector(const Mat4<T>& mat, const Vec<T, 3>& vec, T w = T(0)) noexcept
{
    using V3 = Vec<T, 3>;
    // Treat vec as Vec4(vec.x, vec.y, vec.z, w)
    const T x = mat.data[(0 * 4) + 0] * vec.data[V3::X] + mat.data[(0 * 4) + 1] * vec.data[V3::Y] +
                mat.data[(0 * 4) + 2] * vec.data[V3::Z] + mat.data[(0 * 4) + 3] * w;
    const T y = mat.data[(1 * 4) + 0] * vec.data[V3::X] + mat.data[(1 * 4) + 1] * vec.data[V3::Y] +
                mat.data[(1 * 4) + 2] * vec.data[V3::Z] + mat.data[(1 * 4) + 3] * w;
    const T z = mat.data[(2 * 4) + 0] * vec.data[V3::X] + mat.data[(2 * 4) + 1] * vec.data[V3::Y] +
                mat.data[(2 * 4) + 2] * vec.data[V3::Z] + mat.data[(2 * 4) + 3] * w;

    return Vec<T, 3>(x, y, z);
}

/// \brief Transform a 3D point by a 4x4 matrix (includes translation).
/// \tparam T The type (float, double, etc.).
/// \param mat The 4x4 transformation matrix.
/// \param point The 3D point to transform.
/// \return The transformed 3D point.
/// \note Convenience wrapper for transform_vector with w=1.
template<typename T>
MATRIX_NODISCARD Vec<T, 3> transform_point(const Mat4<T>& mat, const Vec<T, 3>& point) noexcept
{
    return transform_vector(mat, point, T(1));
}

/// \brief Transform a 3D direction by a 4x4 matrix (ignores translation).
/// \tparam T The type (float, double, etc.).
/// \param mat The 4x4 transformation matrix.
/// \param direction The 3D direction to transform.
/// \return The transformed 3D direction.
/// \note Convenience wrapper for transform_vector with w=0.
template<typename T>
MATRIX_NODISCARD Vec<T, 3> transform_direction(const Mat4<T>& mat, const Vec<T, 3>& direction) noexcept
{
    return transform_vector(mat, direction, T(0));
}

}  // namespace matrixlib

#endif  // _MATRIXLIB_MATRIX3D_HPP_
