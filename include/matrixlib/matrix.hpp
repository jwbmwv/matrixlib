// SPDX-License-Identifier: MIT
/// @file matrix.hpp
/// @brief Generic matrix classes Mat<T,R,C> and SquareMat<T,N> with full operator support
/// @details This header provides the template-based matrix classes with comprehensive
///          operator support, SIMD optimizations (CMSIS-DSP, NEON, MVE), and specialized
///          functionality for embedded systems and real-time applications.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#pragma once

#ifndef _MATRIXLIB_MATRIX_HPP_
#define _MATRIXLIB_MATRIX_HPP_

#include "vector.hpp"

namespace matrixlib
{

/// \class Mat<T, R, C>
/// \brief A matrix class templated on type T, rows R, columns C.
/// \tparam T The storage type.
/// \tparam R The number of rows.
/// \tparam C The number of columns.
template<typename T, std::uint32_t R, std::uint32_t C>
class Mat
{
public:
    // Single-dimensional array ensures contiguous memory layout, POD compatibility,
    // and efficient SIMD operations. Row-major storage: data[(row * C) + col]
    alignas(16) T data[R * C];  // Aligned for SIMD performance

    /// \brief Default constructor (zero-initialized for safety).
    MATRIX_CONSTEXPR Mat()
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = T(0);
        }
    }

    /// \brief Constructor from array.
    /// \param arr The array to copy from.
    MATRIX_CONSTEXPR explicit Mat(const T* arr)
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = arr[i];
        }
    }

    /// \brief Copy constructor.
    /// \param other The matrix to copy.
    MATRIX_CONSTEXPR Mat(const Mat& other)
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = other.data[i];
        }
    }

    /// \brief Proxy class for row access.
    class RowProxy
    {
        T* row;

    public:
        RowProxy(T* r) : row(r) {}
        T& operator[](std::uint32_t col) { return row[col]; }
        const T& operator[](std::uint32_t col) const { return row[col]; }

        /// \brief Bounds-checked column access.
        /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if col >= C.
        T& at(std::uint32_t col)
        {
#ifdef MATRIXLIB_DEBUG
            assert(col < C && "Mat::RowProxy::at: column out of range");
#endif
            return row[col];
        }

        const T& at(std::uint32_t col) const
        {
#ifdef MATRIXLIB_DEBUG
            assert(col < C && "Mat::RowProxy::at: column out of range");
#endif
            return row[col];
        }
    };

    /// \brief Const proxy class for row access.
    class ConstRowProxy
    {
        const T* row;

    public:
        ConstRowProxy(const T* r) : row(r) {}
        const T& operator[](std::uint32_t col) const { return row[col]; }

        /// \brief Bounds-checked column access.
        /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if col >= C.
        const T& at(std::uint32_t col) const
        {
#ifdef MATRIXLIB_DEBUG
            assert(col < C && "Mat::ConstRowProxy::at: column out of range");
#endif
            return row[col];
        }
    };

    /// \brief Subscript operator (returns row proxy for mat[row][col]).
    /// \param row The row index.
    /// \return Row proxy.
    RowProxy operator[](std::uint32_t row) { return RowProxy(&data[row * C]); }

    /// \brief Subscript operator (const, returns const row proxy for mat[row][col]).
    /// \param row The row index.
    /// \return Const row proxy.
    ConstRowProxy operator[](std::uint32_t row) const { return ConstRowProxy(&data[row * C]); }

    /// \brief Bounds-checked row access.
    /// \param row The row index.
    /// \return Row proxy.
    /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if row >= R.
    RowProxy at(std::uint32_t row)
    {
#ifdef MATRIXLIB_DEBUG
        assert(row < R && "Mat::at: row out of range");
#endif
        return RowProxy(&data[row * C]);
    }

    /// \brief Bounds-checked row access (const).
    /// \param row The row index.
    /// \return Const row proxy.
    /// \note In debug builds (MATRIXLIB_DEBUG defined), triggers assertion if row >= R.
    ConstRowProxy at(std::uint32_t row) const
    {
#ifdef MATRIXLIB_DEBUG
        assert(row < R && "Mat::at: row out of range");
#endif
        return ConstRowProxy(&data[row * C]);
    }

    /// \brief Addition operator.
    /// \param other The matrix to add.
    /// \return The result matrix.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat operator+(const Mat& other) const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    /// \brief Subtraction operator.
    /// \param other The matrix to subtract.
    /// \return The result matrix.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat operator-(const Mat& other) const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    /// \brief Scalar multiplication operator.
    /// \param scalar The scalar to multiply by.
    /// \return The result matrix.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat operator*(T scalar) const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    /// \brief Scalar division operator.
    /// \param scalar The scalar to divide by.
    /// \return The result matrix.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat operator/(T scalar) const noexcept
    {
        Mat result;
        // Check for zero to avoid undefined behavior (minimal overhead)
        if (scalar == T(0))
        {
            // Return zero matrix for safety
            for (std::uint32_t i = 0; i < R * C; ++i)
            {
                result.data[i] = T(0);
            }
            return result;
        }
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    /// \brief Matrix multiplication operator.
    /// \tparam S The columns of other matrix.
    /// \param other The matrix to multiply.
    /// \return The result matrix.
    template<std::uint32_t S>
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat<T, R, S> operator*(const Mat<T, C, S>& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        // Optimized 4x4 matrix multiplication
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && R == 4 && C == 4 && S == 4)
        {
            MATRIX_LIKELY
            Mat<T, R, S> result;
            float32x4_t row0 = vld1q_f32(&data[0]);
            float32x4_t row1 = vld1q_f32(&data[4]);
            float32x4_t row2 = vld1q_f32(&data[8]);
            float32x4_t row3 = vld1q_f32(&data[12]);

            for (std::uint32_t j = 0; j < 4; ++j)
            {
                float32x4_t col = {other.data[j], other.data[4 + j], other.data[8 + j], other.data[12 + j]};
                float32x4_t col_xxxx = vdupq_laneq_f32(col, 0);
                float32x4_t col_yyyy = vdupq_laneq_f32(col, 1);
                float32x4_t col_zzzz = vdupq_laneq_f32(col, 2);
                float32x4_t col_wwww = vdupq_laneq_f32(col, 3);

                // result[i][j] = row[i] dot col
                float32x4_t r = vmulq_f32(row0, col_xxxx);
                r = vfmaq_f32(r, row1, col_yyyy);
                r = vfmaq_f32(r, row2, col_zzzz);
                r = vfmaq_f32(r, row3, col_wwww);

                result.data[j] = vgetq_lane_f32(r, 0);
                result.data[4 + j] = vgetq_lane_f32(r, 1);
                result.data[8 + j] = vgetq_lane_f32(r, 2);
                result.data[12 + j] = vgetq_lane_f32(r, 3);
            }
            return result;
        }
        // Optimized 3x3 matrix multiplication
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && R == 3 && C == 3 && S == 3)
        {
            MATRIX_LIKELY
            Mat<T, R, S> result;
            // Load rows with padding
            float32x4_t row0 = vld1q_f32(&data[0]);
            float32x4_t row1 = vld1q_f32(&data[3]);
            float32x4_t row2 = vld1q_f32(&data[6]);

            for (std::uint32_t j = 0; j < 3; ++j)
            {
                float32x4_t col = {other.data[j], other.data[3 + j], other.data[6 + j], 0.0f};
                float32x4_t col_xxxx = vdupq_laneq_f32(col, 0);
                float32x4_t col_yyyy = vdupq_laneq_f32(col, 1);
                float32x4_t col_zzzz = vdupq_laneq_f32(col, 2);

                float32x4_t r = vmulq_f32(row0, col_xxxx);
                r = vfmaq_f32(r, row1, col_yyyy);
                r = vfmaq_f32(r, row2, col_zzzz);

                result.data[j] = vgetq_lane_f32(r, 0);
                result.data[3 + j] = vgetq_lane_f32(r, 1);
                result.data[6 + j] = vgetq_lane_f32(r, 2);
            }
            return result;
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Mat<T, R, S> result;
            arm_matrix_instance_f32 A, B, Res;
            arm_mat_init_f32(&A, R, C, const_cast<float*>(reinterpret_cast<const float*>(this->data)));
            arm_mat_init_f32(&B, C, S, const_cast<float*>(reinterpret_cast<const float*>(other.data)));
            arm_mat_init_f32(&Res, R, S, reinterpret_cast<float*>(result.data));
            arm_mat_mult_f32(&A, &B, &Res);
            return result;
        }
#endif
        Mat<T, R, S> result;
        for (std::uint32_t i = 0; i < R; ++i)
        {
            for (std::uint32_t j = 0; j < S; ++j)
            {
                T sum = T(0);
                for (std::uint32_t k = 0; k < C; ++k)
                {
                    sum += data[(i * C) + k] * other.data[(k * S) + j];
                }
                result.data[(i * S) + j] = sum;
            }
        }
        return result;
    }

    /// \brief Matrix-vector multiplication operator.
    /// \param v The vector to multiply.
    /// \return The result vector.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Vec<T, R> operator*(const Vec<T, C>& v) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        // Optimized 4x4 matrix-vector multiplication
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && R == 4 && C == 4)
        {
            MATRIX_LIKELY
            Vec<T, R> result;
            float32x4_t vec = vld1q_f32(reinterpret_cast<const float*>(v.data));
            float32x4_t v_xxxx = vdupq_laneq_f32(vec, 0);
            float32x4_t v_yyyy = vdupq_laneq_f32(vec, 1);
            float32x4_t v_zzzz = vdupq_laneq_f32(vec, 2);
            float32x4_t v_wwww = vdupq_laneq_f32(vec, 3);

            float32x4_t row0 = vld1q_f32(&data[0]);
            float32x4_t row1 = vld1q_f32(&data[4]);
            float32x4_t row2 = vld1q_f32(&data[8]);
            float32x4_t row3 = vld1q_f32(&data[12]);

            float32x4_t r = vmulq_f32(row0, v_xxxx);
            r = vfmaq_f32(r, row1, v_yyyy);
            r = vfmaq_f32(r, row2, v_zzzz);
            r = vfmaq_f32(r, row3, v_wwww);

            vst1q_f32(reinterpret_cast<float*>(result.data), r);
            return result;
        }
        // Optimized 3x3 matrix-vector multiplication
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && R == 3 && C == 3)
        {
            MATRIX_LIKELY
            Vec<T, R> result;
            float32x4_t vec = {v.data[0], v.data[1], v.data[2], 0.0f};
            float32x4_t v_xxxx = vdupq_laneq_f32(vec, 0);
            float32x4_t v_yyyy = vdupq_laneq_f32(vec, 1);
            float32x4_t v_zzzz = vdupq_laneq_f32(vec, 2);

            float32x4_t row0 = vld1q_f32(&data[0]);
            float32x4_t row1 = vld1q_f32(&data[3]);
            float32x4_t row2 = vld1q_f32(&data[6]);

            float32x4_t r = vmulq_f32(row0, v_xxxx);
            r = vfmaq_f32(r, row1, v_yyyy);
            r = vfmaq_f32(r, row2, v_zzzz);

            result.data[0] = vgetq_lane_f32(r, 0);
            result.data[1] = vgetq_lane_f32(r, 1);
            result.data[2] = vgetq_lane_f32(r, 2);
            return result;
        }
#endif
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Vec<T, R> result;
            arm_matrix_instance_f32 A;
            arm_mat_init_f32(&A, R, C, const_cast<float*>(reinterpret_cast<const float*>(this->data)));
            arm_mat_vec_mult_f32(&A, reinterpret_cast<const float*>(v.data), reinterpret_cast<float*>(result.data));
            return result;
        }
#endif
        Vec<T, R> result;
        for (std::uint32_t i = 0; i < R; ++i)
        {
            T sum = T(0);
            for (std::uint32_t j = 0; j < C; ++j)
            {
                sum += data[(i * C) + j] * v.data[j];
            }
            result.data[i] = sum;
        }
        return result;
    }

    /// \brief Equality operator.
    /// \param other The matrix to compare.
    /// \return True if equal (uses epsilon comparison for floating point types).
    MATRIX_CONSTEXPR bool operator==(const Mat& other) const noexcept
    {
#ifdef CONFIG_MATRIXLIB_NEON
        // SIMD path for 4x4 float matrix equality comparison
        MATRIX_IF_CONSTEXPR(std::is_same<T, float>::value && R == 4 && C == 4)
        {
            MATRIX_LIKELY
            float32x4_t eps = vdupq_n_f32(std::numeric_limits<float>::epsilon());
            for (std::uint32_t i = 0; i < 4; ++i)
            {
                float32x4_t a = vld1q_f32(&data[i * 4]);
                float32x4_t b = vld1q_f32(&other.data[i * 4]);
                float32x4_t diff = vabdq_f32(a, b);
                uint32x4_t cmp = vcleq_f32(diff, eps);
                uint64x2_t cmp64 = vreinterpretq_u64_u32(cmp);
                if (!(vgetq_lane_u64(cmp64, 0) == ~0ULL && vgetq_lane_u64(cmp64, 1) == ~0ULL))
                    return false;
            }
            return true;
        }
#endif
        // For floating point types, use epsilon comparison
        MATRIX_IF_CONSTEXPR(std::is_floating_point<T>::value)
        {
            MATRIX_LIKELY
            for (std::uint32_t i = 0; i < R * C; ++i)
            {
                if (std::abs(data[i] - other.data[i]) > std::numeric_limits<T>::epsilon())
                {
                    return false;
                }
            }
            return true;
        }
        // For integral types, use exact comparison
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            if (data[i] != other.data[i])
            {
                return false;
            }
        }
        return true;
    }

    /// \brief Inequality operator.
    /// \param other The matrix to compare.
    /// \return True if not equal.
    bool operator!=(const Mat& other) const noexcept { return !(*this == other); }

    /// \brief Addition assignment operator.
    /// \param other The matrix to add.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Mat& operator+=(const Mat& other) noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = data[i] + other.data[i];
        }
        return *this;
    }

    /// \brief Subtraction assignment operator.
    /// \param other The matrix to subtract.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Mat& operator-=(const Mat& other) noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = data[i] - other.data[i];
        }
        return *this;
    }

    /// \brief Scalar multiplication assignment operator.
    /// \param scalar The scalar to multiply by.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Mat& operator*=(T scalar) noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = data[i] * scalar;
        }
        return *this;
    }

    /// \brief Scalar division assignment operator.
    /// \param scalar The scalar to divide by.
    /// \return Reference to this.
    MATRIX_CONSTEXPR Mat& operator/=(T scalar) noexcept
    {
        // Check for zero to avoid undefined behavior (minimal overhead)
        if (scalar == T(0))
        {
            // Set to zero for safety
            for (std::uint32_t i = 0; i < R * C; ++i)
            {
                data[i] = T(0);
            }
            return *this;
        }
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = data[i] / scalar;
        }
        return *this;
    }

    /// \brief Unary negation operator.
    /// \return The negated matrix.
    MATRIX_CONSTEXPR MATRIX_NODISCARD Mat operator-() const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = -data[i];
        }
        return result;
    }

    /// \brief Transpose of the matrix.
    /// \return The transposed matrix.
    Mat<T, C, R> transpose() const
    {
#ifdef CONFIG_MATRIXLIB_CMSIS
        if (std::is_same<T, float>::value)
        {
            Mat<T, C, R> result;
            arm_matrix_instance_f32 A, Res;
            arm_mat_init_f32(&A, R, C, const_cast<float*>(reinterpret_cast<const float*>(this->data)));
            arm_mat_init_f32(&Res, C, R, reinterpret_cast<float*>(result.data));
            arm_mat_trans_f32(&A, &Res);
            return result;
        }
#endif
        Mat<T, C, R> result;
        for (std::uint32_t i = 0; i < R; ++i)
        {
            for (std::uint32_t j = 0; j < C; ++j)
            {
                result.data[(j * R) + i] = data[(i * C) + j];
            }
        }
        return result;
    }

    /// \brief Frobenius norm of the matrix.
    /// \return The Frobenius norm.
    T frobenius_norm() const noexcept
    {
        T sum = T(0);
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }

    /// \brief Normalized matrix (divided by its Frobenius norm).
    /// \return The normalized matrix.
    Mat normalized() const noexcept
    {
        const T norm = frobenius_norm();
        if (norm == T(0))
            return *this;
        return *this / norm;
    }

    /// \brief Get the total number of elements in the matrix.
    /// \return The size of the matrix (R * C).
    MATRIX_CONSTEXPR std::uint32_t size() const noexcept { return R * C; }

    /// \brief Get the number of rows.
    /// \return The number of rows.
    MATRIX_CONSTEXPR std::uint32_t rows() const noexcept { return R; }

    /// \brief Get the number of columns.
    /// \return The number of columns.
    MATRIX_CONSTEXPR std::uint32_t cols() const noexcept { return C; }

    /// \brief Bounds-checked element access.
    /// \param row The row index.
    /// \param col The column index.
    /// \return Reference to the element.
    T& at(std::uint32_t row, std::uint32_t col) { return data[(row * C) + col]; }

    /// \brief Bounds-checked element access (const).
    /// \param row The row index.
    /// \param col The column index.
    /// \return Const reference to the element.
    const T& at(std::uint32_t row, std::uint32_t col) const { return data[(row * C) + col]; }

    /// \brief Fill the matrix with a value.
    /// \param value The value to fill with.
    MATRIX_CONSTEXPR void fill(T value) noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            data[i] = value;
        }
    }

    /// \brief Create a zero matrix.
    /// \return The zero matrix.
    static MATRIX_CONSTEXPR Mat zero()
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = T(0);
        }
        return result;
    }

    /// \brief Create a matrix filled with ones.
    /// \return The ones matrix.
    static Mat ones()
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = T(1);
        }
        return result;
    }

    /// \brief Create an identity matrix (only for square matrices).
    /// \return The identity matrix.
    template<std::uint32_t RR = R, std::uint32_t CC = C>
    static MATRIX_CONSTEXPR typename std::enable_if<RR == CC, Mat>::type identity()
    {
        Mat result;
        for (std::uint32_t i = 0; i < R; ++i)
        {
            for (std::uint32_t j = 0; j < C; ++j)
            {
                result.data[(i * C) + j] = (i == j) ? T(1) : T(0);
            }
        }
        return result;
    }

    /// \brief Get raw pointer to data.
    /// \return Pointer to the underlying data array.
    T* ptr() noexcept { return data; }

    /// \brief Get raw pointer to data (const).
    /// \return Const pointer to the underlying data array.
    const T* ptr() const noexcept { return data; }

    /// \brief Get total number of elements.
    /// \return The total number of elements (R * C).
    MATRIX_CONSTEXPR std::uint32_t size() const noexcept { return R * C; }

    /// \brief Swap two matrices.
    /// \param other The matrix to swap with.
    void swap(Mat& other) noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            T temp = data[i];
            data[i] = other.data[i];
            other.data[i] = temp;
        }
    }

    /// \brief Approximate equality comparison.
    /// \param other The matrix to compare.
    /// \param epsilon The tolerance (defaults to machine epsilon for type T).
    /// \return True if approximately equal.
    bool approx_equal(const Mat& other, T epsilon = std::numeric_limits<T>::epsilon()) const noexcept
    {
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            if (std::abs(data[i] - other.data[i]) > epsilon)
            {
                return false;
            }
        }
        return true;
    }

    /// \brief Linear interpolation between two matrices.
    /// \param other The target matrix.
    /// \param t The interpolation parameter [0, 1].
    /// \return The interpolated matrix.
    Mat lerp(const Mat& other, T t) const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] + t * (other.data[i] - data[i]);
        }
        return result;
    }

    /// \brief Element-wise (Hadamard) product.
    /// \param other The other matrix.
    /// \return The element-wise product.
    Mat hadamard(const Mat& other) const noexcept
    {
        Mat result;
        for (std::uint32_t i = 0; i < R * C; ++i)
        {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    /// \brief Extract a row as a vector.
    /// \param row_idx The row index.
    /// \return The row as a vector.
    Vec<T, C> row(std::uint32_t row_idx) const noexcept
    {
        Vec<T, C> result;
        for (std::uint32_t j = 0; j < C; ++j)
        {
            result.data[j] = data[(row_idx * C) + j];
        }
        return result;
    }

    /// \brief Extract a column as a vector.
    /// \param col_idx The column index.
    /// \return The column as a vector.
    Vec<T, R> col(std::uint32_t col_idx) const noexcept
    {
        Vec<T, R> result;
        for (std::uint32_t i = 0; i < R; ++i)
        {
            result.data[i] = data[(i * C) + col_idx];
        }
        return result;
    }

    /// \brief Get minimum element.
    /// \return The minimum element value.
    T min_element() const noexcept
    {
        T min_val = data[0];
        for (std::uint32_t i = 1; i < R * C; ++i)
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
        T max_val = data[0];
        for (std::uint32_t i = 1; i < R * C; ++i)
        {
            if (data[i] > max_val)
                max_val = data[i];
        }
        return max_val;
    }
};

/// \brief Scalar multiplication (commutative).
/// \tparam T The type.
/// \tparam R The rows.
/// \tparam C The columns.
/// \param scalar The scalar.
/// \param m The matrix.
/// \return The result matrix.
template<typename T, std::uint32_t R, std::uint32_t C>
MATRIX_CONSTEXPR MATRIX_NODISCARD Mat<T, R, C> operator*(T scalar, const Mat<T, R, C>& m) noexcept
{
    return m * scalar;
}

/// \brief Matrix multiplication free function.
/// \tparam T The type.
/// \tparam R The rows of A.
/// \tparam C The columns of A / rows of B.
/// \tparam S The columns of B.
/// \param A The first matrix.
/// \param B The second matrix.
/// \return The result matrix.
template<typename T, std::uint32_t R, std::uint32_t C, std::uint32_t S>
MATRIX_CONSTEXPR MATRIX_NODISCARD Mat<T, R, S> mul(const Mat<T, R, C>& A, const Mat<T, C, S>& B) noexcept
{
    return A * B;
}

/// \brief Matrix-vector multiplication free function.
/// \tparam T The type.
/// \tparam R The rows of A.
/// \tparam C The columns of A.
/// \param A The matrix.
/// \param x The vector.
/// \return The result vector.
template<typename T, std::uint32_t R, std::uint32_t C>
MATRIX_CONSTEXPR MATRIX_NODISCARD Vec<T, R> mul(const Mat<T, R, C>& A, const Vec<T, C>& x) noexcept
{
    return A * x;
}

/// \class SquareMat<T, N>
/// \brief A square matrix class templated on type T and size N.
/// \tparam T The storage type.
/// \tparam N The size (N x N).
template<typename T, std::uint32_t N>
class SquareMat : public Mat<T, N, N>
{
public:
    /// \brief Inherit constructors.
    using Mat<T, N, N>::Mat;

    /// \brief Trace of the matrix (sum of diagonal elements).
    /// \return The trace.
    MATRIX_CONSTEXPR T trace() const noexcept
    {
        T sum = T(0);
        for (std::uint32_t i = 0; i < N; ++i)
        {
            sum += this->data[(i * N) + i];
        }
        return sum;
    }

    /// \brief Create an identity matrix.
    /// \return The identity matrix.
    static MATRIX_CONSTEXPR SquareMat<T, N> identity()
    {
        SquareMat<T, N> result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            for (std::uint32_t j = 0; j < N; ++j)
            {
                result.data[(i * N) + j] = (i == j) ? T(1) : T(0);
            }
        }
        return result;
    }

    /// \brief Rank of the matrix (number of linearly independent rows/columns).
    /// \return The rank.
    std::uint32_t rank() const
    {
        Mat<T, N, N> temp = *this;
        std::uint32_t rank = 0;
        // Use epsilon for floating point, zero for integers
        const T epsilon = std::is_floating_point<T>::value ? std::numeric_limits<T>::epsilon() * T(100) : T(0);
        for (std::uint32_t i = 0; i < N; ++i)
        {
            // Find pivot
            std::uint32_t max_row = i;
            for (std::uint32_t k = i; k < N; ++k)
            {
                if (std::abs(temp.data[(k * N) + i]) > std::abs(temp.data[(max_row * N) + i]))
                {
                    max_row = k;
                }
            }
            if (std::abs(temp.data[(max_row * N) + i]) <= epsilon)
            {
                continue;  // No pivot in this column
            }
            // Swap rows if needed
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < N; ++k)
                {
                    T swap_temp = temp.data[(i * N) + k];
                    temp.data[(i * N) + k] = temp.data[(max_row * N) + k];
                    temp.data[(max_row * N) + k] = swap_temp;
                }
            }
            ++rank;
            // Eliminate below
            for (std::uint32_t k = i + 1; k < N; ++k)
            {
                T factor = temp.data[(k * N) + i] / temp.data[(i * N) + i];
                for (std::uint32_t j = i; j < N; ++j)
                {
                    temp.data[(k * N) + j] -= factor * temp.data[(i * N) + j];
                }
            }
        }
        return rank;
    }

    /// \brief Determinant of the matrix.
    /// \return The determinant.
    T determinant() const
    {
        Mat<T, N, N> temp = *this;
        T det = T(1);
        // Use epsilon for floating point, zero for integers
        const T epsilon = std::is_floating_point<T>::value ? std::numeric_limits<T>::epsilon() * T(100) : T(0);
        for (std::uint32_t i = 0; i < N; ++i)
        {
            // Find pivot
            std::uint32_t max_row = i;
            for (std::uint32_t k = i + 1; k < N; ++k)
            {
                if (std::abs(temp.data[(k * N) + i]) > std::abs(temp.data[(max_row * N) + i]))
                {
                    max_row = k;
                }
            }
            if (std::abs(temp.data[(max_row * N) + i]) <= epsilon)
            {
                return T(0);
            }
            // Swap rows if needed
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < N; ++k)
                {
                    T swap_temp = temp.data[(i * N) + k];
                    temp.data[(i * N) + k] = temp.data[(max_row * N) + k];
                    temp.data[(max_row * N) + k] = swap_temp;
                }
                det = -det;
            }
            det *= temp.data[(i * N) + i];
            // Eliminate below
            for (std::uint32_t k = i + 1; k < N; ++k)
            {
                T factor = temp.data[(k * N) + i] / temp.data[(i * N) + i];
                for (std::uint32_t j = i; j < N; ++j)
                {
                    temp.data[(k * N) + j] -= factor * temp.data[(i * N) + j];
                }
            }
        }
        return det;
    }

    /// \brief Inverse of the matrix (only for square matrices).
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
            float inv_data[N * N];
            arm_mat_init_f32(&src, N, N, const_cast<float*>(reinterpret_cast<const float*>(this->data)));
            arm_mat_init_f32(&dst, N, N, inv_data);
            arm_status status = arm_mat_inverse_f32(&src, &dst);
            if (status == ARM_MATRIX_SUCCESS)
            {
                SquareMat result;
                for (std::uint32_t i = 0; i < N * N; ++i)
                {
                    result.data[i] = static_cast<T>(inv_data[i]);
                }
                return result;
            }
            // Fall through to Gauss-Jordan if CMSIS fails
        }
#endif
        // Gauss-Jordan elimination with partial pivoting
        Mat<T, N, N> temp = *this;
        SquareMat result;

        // Initialize result as identity
        for (std::uint32_t i = 0; i < N; ++i)
        {
            for (std::uint32_t j = 0; j < N; ++j)
            {
                result.data[(i * N) + j] = (i == j) ? T(1) : T(0);
            }
        }

        // Forward elimination
        for (std::uint32_t i = 0; i < N; ++i)
        {
            // Find pivot
            std::uint32_t max_row = i;
            for (std::uint32_t k = i + 1; k < N; ++k)
            {
                if (std::abs(temp.data[(k * N) + i]) > std::abs(temp.data[(max_row * N) + i]))
                {
                    max_row = k;
                }
            }

            // Swap rows in both matrices
            if (max_row != i)
            {
                for (std::uint32_t k = 0; k < N; ++k)
                {
                    T swap_temp = temp.data[(i * N) + k];
                    temp.data[(i * N) + k] = temp.data[(max_row * N) + k];
                    temp.data[(max_row * N) + k] = swap_temp;

                    swap_temp = result.data[(i * N) + k];
                    result.data[(i * N) + k] = result.data[(max_row * N) + k];
                    result.data[(max_row * N) + k] = swap_temp;
                }
            }

            // Scale pivot row
            const T pivot = temp.data[(i * N) + i];
            for (std::uint32_t k = 0; k < N; ++k)
            {
                temp.data[(i * N) + k] /= pivot;
                result.data[(i * N) + k] /= pivot;
            }

            // Eliminate column
            for (std::uint32_t k = 0; k < N; ++k)
            {
                if (k != i)
                {
                    const T factor = temp.data[(k * N) + i];
                    for (std::uint32_t j = 0; j < N; ++j)
                    {
                        temp.data[(k * N) + j] -= factor * temp.data[(i * N) + j];
                        result.data[(k * N) + j] -= factor * result.data[(i * N) + j];
                    }
                }
            }
        }

        return result;
    }

    /// \brief QR decomposition using Gram-Schmidt orthogonalization.
    /// \return Pair of Q (orthogonal) and R (upper triangular) matrices.
    /// \note Q * R = A (this matrix). Useful for solving least-squares problems.
    ///       For numerical stability, consider using modified Gram-Schmidt.
    std::pair<SquareMat, SquareMat> qr() const
    {
        SquareMat Q;
        SquareMat R;

        // Initialize R to zero
        for (std::uint32_t i = 0; i < N * N; ++i)
        {
            R.data[i] = T(0);
        }

        // Modified Gram-Schmidt (more numerically stable)
        Vec<T, N> columns[N];

        // Extract columns
        for (std::uint32_t j = 0; j < N; ++j)
        {
            for (std::uint32_t i = 0; i < N; ++i)
            {
                columns[j].data[i] = this->data[(i * N) + j];
            }
        }

        // Orthogonalize
        for (std::uint32_t j = 0; j < N; ++j)
        {
            Vec<T, N> q_j = columns[j];

            // Subtract projections onto previous columns
            for (std::uint32_t k = 0; k < j; ++k)
            {
                Vec<T, N> q_k;
                for (std::uint32_t i = 0; i < N; ++i)
                {
                    q_k.data[i] = Q.data[(i * N) + k];
                }

                T dot = q_j.dot(q_k);
                R.data[(k * N) + j] = dot;

                q_j = q_j - q_k * dot;
            }

            // Normalize
            T norm = q_j.length();
            R.data[(j * N) + j] = norm;

            if (norm > std::numeric_limits<T>::epsilon())
            {
                q_j = q_j / norm;
            }

            // Store in Q
            for (std::uint32_t i = 0; i < N; ++i)
            {
                Q.data[(i * N) + j] = q_j.data[i];
            }
        }

        return std::make_pair(Q, R);
    }

    /// \brief Solve linear system Ax = b using QR decomposition.
    /// \param b The right-hand side vector.
    /// \return Solution vector x.
    /// \note Uses QR decomposition for numerical stability. Suitable for least-squares.
    ///       For efficiency, if solving multiple systems with same A, compute QR once.
    Vec<T, N> solve_qr(const Vec<T, N>& b) const
    {
        auto [Q, R] = qr();

        // Compute Q^T * b
        Vec<T, N> y;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            T sum = T(0);
            for (std::uint32_t j = 0; j < N; ++j)
            {
                sum += Q.data[(j * N) + i] * b.data[j];
            }
            y.data[i] = sum;
        }

        // Back substitution: solve Rx = y
        Vec<T, N> x;
        for (int i = N - 1; i >= 0; --i)
        {
            T sum = y.data[i];
            for (std::uint32_t j = i + 1; j < N; ++j)
            {
                sum -= R.data[(i * N) + j] * x.data[j];
            }
            x.data[i] = sum / R.data[(i * N) + i];
        }

        return x;
    }

    /// \brief LU decomposition with partial pivoting (Doolittle algorithm).
    /// \return Tuple of (L, U, P) where PA = LU.
    /// \note L is lower triangular with 1s on diagonal, U is upper triangular.
    ///       P is permutation matrix for numerical stability.
    ///       Time complexity: O(N³), Space: O(N²)
    std::tuple<SquareMat, SquareMat, SquareMat> lu() const
    {
        SquareMat L = SquareMat::identity();
        SquareMat U = *this;
        SquareMat P = SquareMat::identity();

        for (std::uint32_t k = 0; k < N - 1; ++k)
        {
            // Find pivot
            std::uint32_t pivot_row = k;
            T max_val = std::abs(U.data[(k * N) + k]);

            for (std::uint32_t i = k + 1; i < N; ++i)
            {
                T val = std::abs(U.data[(i * N) + k]);
                if (val > max_val)
                {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Swap rows in U and P
            if (pivot_row != k)
            {
                for (std::uint32_t j = 0; j < N; ++j)
                {
                    std::swap(U.data[(k * N) + j], U.data[(pivot_row * N) + j]);
                    std::swap(P.data[(k * N) + j], P.data[(pivot_row * N) + j]);
                    if (j < k)
                    {
                        std::swap(L.data[(k * N) + j], L.data[(pivot_row * N) + j]);
                    }
                }
            }

            // Eliminate column k
            for (std::uint32_t i = k + 1; i < N; ++i)
            {
                if (std::abs(U.data[(k * N) + k]) > std::numeric_limits<T>::epsilon())
                {
                    T factor = U.data[(i * N) + k] / U.data[(k * N) + k];
                    L.data[(i * N) + k] = factor;

                    for (std::uint32_t j = k; j < N; ++j)
                    {
                        U.data[(i * N) + j] -= factor * U.data[(k * N) + j];
                    }
                }
            }
        }

        return std::make_tuple(L, U, P);
    }

    /// \brief Cholesky decomposition for symmetric positive-definite matrices.
    /// \return Lower triangular matrix L where A = L * L^T.
    /// \note Only works for symmetric positive-definite matrices.
    ///       More efficient than LU: O(N³/3) vs O(2N³/3).
    ///       Numerically stable for well-conditioned matrices.
    SquareMat cholesky() const
    {
        SquareMat L;

        for (std::uint32_t i = 0; i < N; ++i)
        {
            for (std::uint32_t j = 0; j <= i; ++j)
            {
                T sum = T(0);

                if (i == j)
                {
                    for (std::uint32_t k = 0; k < j; ++k)
                    {
                        sum += L.data[(j * N) + k] * L.data[(j * N) + k];
                    }

                    T diag = (*this)(j, j) - sum;
                    if (diag <= T(0))
                    {
                        // Matrix not positive definite
                        L.data[(j * N) + j] = T(0);
                    }
                    else
                    {
                        L.data[(j * N) + j] = std::sqrt(diag);
                    }
                }
                else
                {
                    for (std::uint32_t k = 0; k < j; ++k)
                    {
                        sum += L.data[(i * N) + k] * L.data[(j * N) + k];
                    }

                    if (std::abs(L.data[(j * N) + j]) > std::numeric_limits<T>::epsilon())
                    {
                        L.data[(i * N) + j] = ((*this)(i, j) - sum) / L.data[(j * N) + j];
                    }
                    else
                    {
                        L.data[(i * N) + j] = T(0);
                    }
                }
            }
        }

        return L;
    }

    /// \brief Solve linear system using Cholesky decomposition (for SPD matrices).
    /// \param b The right-hand side vector.
    /// \return Solution vector x.
    /// \note Requires symmetric positive-definite matrix. More efficient than LU.
    Vec<T, N> solve_cholesky(const Vec<T, N>& b) const
    {
        SquareMat L = cholesky();

        // Forward substitution: solve Ly = b
        Vec<T, N> y;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            T sum = b.data[i];
            for (std::uint32_t j = 0; j < i; ++j)
            {
                sum -= L.data[(i * N) + j] * y.data[j];
            }
            y.data[i] = sum / L.data[(i * N) + i];
        }

        // Back substitution: solve L^T x = y
        Vec<T, N> x;
        for (int i = N - 1; i >= 0; --i)
        {
            T sum = y.data[i];
            for (std::uint32_t j = i + 1; j < N; ++j)
            {
                sum -= L.data[(j * N) + i] * x.data[j];
            }
            x.data[i] = sum / L.data[(i * N) + i];
        }

        return x;
    }

    /// \brief Power iteration method for dominant eigenvalue/eigenvector.
    /// \param max_iterations Maximum number of iterations (default: 100).
    /// \param tolerance Convergence tolerance (default: 1e-6).
    /// \return Pair of (eigenvalue, eigenvector).
    /// \note Finds largest eigenvalue (by magnitude) and corresponding eigenvector.
    ///       Converges slowly for eigenvalues close in magnitude.
    ///       For small matrices (2x2, 3x3), direct methods may be faster.
    std::pair<T, Vec<T, N>> powerIteration(std::uint32_t max_iterations = 100, T tolerance = T(1e-6)) const
    {
        // Start with random vector
        Vec<T, N> v;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            v.data[i] = T(1);
        }
        v = v.normalized();

        T eigenvalue = T(0);

        for (std::uint32_t iter = 0; iter < max_iterations; ++iter)
        {
            // Multiply matrix by vector
            Vec<T, N> Av;
            for (std::uint32_t i = 0; i < N; ++i)
            {
                T sum = T(0);
                for (std::uint32_t j = 0; j < N; ++j)
                {
                    sum += (*this)(i, j) * v.data[j];
                }
                Av.data[i] = sum;
            }

            // Rayleigh quotient: eigenvalue = v^T * A * v
            T new_eigenvalue = v.dot(Av);

            // Normalize
            v = Av.normalized();

            // Check convergence
            if (std::abs(new_eigenvalue - eigenvalue) < tolerance)
            {
                return std::make_pair(new_eigenvalue, v);
            }

            eigenvalue = new_eigenvalue;
        }

        return std::make_pair(eigenvalue, v);
    }

    /// \brief QR algorithm for eigenvalue computation (iterative).
    /// \param max_iterations Maximum number of iterations (default: 100).
    /// \param tolerance Convergence tolerance (default: 1e-6).
    /// \return Vector of eigenvalues (unsorted).
    /// \note Uses iterative QR algorithm. For small matrices, may not fully converge.
    ///       Works best for symmetric matrices. Non-symmetric may have complex eigenvalues
    ///       (returned as real parts only).
    Vec<T, N> eigenvaluesQR(std::uint32_t max_iterations = 100, T tolerance = T(1e-6)) const
    {
        SquareMat A = *this;

        for (std::uint32_t iter = 0; iter < max_iterations; ++iter)
        {
            auto [Q, R] = A.qr();
            SquareMat A_new = R * Q;

            // Check convergence (off-diagonal elements near zero)
            T off_diag_sum = T(0);
            for (std::uint32_t i = 0; i < N; ++i)
            {
                for (std::uint32_t j = 0; j < N; ++j)
                {
                    if (i != j)
                    {
                        off_diag_sum += std::abs(A_new(i, j));
                    }
                }
            }

            if (off_diag_sum < tolerance)
            {
                // Extract diagonal (eigenvalues)
                Vec<T, N> eigenvalues;
                for (std::uint32_t i = 0; i < N; ++i)
                {
                    eigenvalues.data[i] = A_new(i, i);
                }
                return eigenvalues;
            }

            A = A_new;
        }

        // Extract diagonal even if not fully converged
        Vec<T, N> eigenvalues;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            eigenvalues.data[i] = A(i, i);
        }
        return eigenvalues;
    }

    /// \brief Create a scale matrix.
    /// \param scale The scale factors (one per axis).
    /// \return The scale matrix.
    static SquareMat scale(const Vec<T, N>& scale_vec) noexcept
    {
        SquareMat result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            for (std::uint32_t j = 0; j < N; ++j)
            {
                result.data[(i * N) + j] = (i == j) ? scale_vec.data[i] : T(0);
            }
        }
        return result;
    }

    /// \brief Create a uniform scale matrix.
    /// \param s The uniform scale factor.
    /// \return The scale matrix.
    static SquareMat scale(T s) noexcept
    {
        SquareMat result;
        for (std::uint32_t i = 0; i < N; ++i)
        {
            for (std::uint32_t j = 0; j < N; ++j)
            {
                result.data[(i * N) + j] = (i == j) ? s : T(0);
            }
        }
        return result;
    }
};

// Static asserts for trivial copyability (C++11, replaces deprecated is_pod)
static_assert(std::is_trivially_copyable<Mat<float, 2, 2>>::value, "Mat<float, 2, 2> must be trivially copyable");
static_assert(std::is_trivially_copyable<Mat<float, 3, 3>>::value, "Mat<float, 3, 3> must be trivially copyable");
static_assert(std::is_trivially_copyable<Mat<float, 4, 4>>::value, "Mat<float, 4, 4> must be trivially copyable");
static_assert(std::is_trivially_copyable<SquareMat<float, 2>>::value, "SquareMat<float, 2> must be trivially copyable");
static_assert(std::is_trivially_copyable<SquareMat<float, 3>>::value, "SquareMat<float, 3> must be trivially copyable");
static_assert(std::is_trivially_copyable<SquareMat<float, 4>>::value, "SquareMat<float, 4> must be trivially copyable");

}  // namespace matrixlib

#endif  // _MATRIXLIB_MATRIX_HPP_
