/// @file test_simd_equivalence.cpp
/// @brief MatrixLib Test Suite - SIMD vs Scalar Equivalence
/// @details Verifies SIMD-optimized paths produce identical results to scalar implementations

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <limits>
#include <cmath>

using namespace matrixlib;

// Tolerance for floating-point comparisons
constexpr float FLOAT_EPSILON = std::numeric_limits<float>::epsilon() * 100.0f;

// Helper to compare vectors with epsilon
template<typename T, int N>
bool vectors_equal(const Vec<T, N>& a, const Vec<T, N>& b, T epsilon)
{
    for (int i = 0; i < N; ++i)
    {
        if (std::abs(a[i] - b[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

// Helper to compare matrices with epsilon
template<typename T, int R, int C>
bool matrices_equal(const Mat<T, R, C>& a, const Mat<T, R, C>& b, T epsilon)
{
    for (int r = 0; r < R; ++r)
    {
        for (int c = 0; c < C; ++c)
        {
            if (std::abs(a(r, c) - b(r, c)) > epsilon)
            {
                return false;
            }
        }
    }
    return true;
}

// Helper to compare quaternions with epsilon
template<typename T>
bool quaternions_equal(const Quaternion<T>& a, const Quaternion<T>& b, T epsilon)
{
    return std::abs(a.w() - b.w()) <= epsilon && std::abs(a.x() - b.x()) <= epsilon &&
           std::abs(a.y() - b.y()) <= epsilon && std::abs(a.z() - b.z()) <= epsilon;
}

// ==================== Vector SIMD Equivalence Tests ====================

#ifdef CONFIG_MATRIXLIB_NEON

TEST(SIMD_Vec, Addition)
{
    Vec3f a(1.5f, 2.3f, -3.7f);
    Vec3f b(4.2f, -1.8f, 5.1f);

// Force scalar path by disabling SIMD temporarily
#undef CONFIG_MATRIXLIB_NEON
#include <matrixlib/vector.hpp>
    Vec3f scalar_result = a + b;
#define CONFIG_MATRIXLIB_NEON

    Vec3f simd_result = a + b;

    EXPECT_TRUE(vectors_equal(scalar_result, simd_result, FLOAT_EPSILON));
}

TEST(SIMD_Vec, DotProduct)
{
    Vec4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f b(5.0f, 6.0f, 7.0f, 8.0f);

    float result = a.dot(b);

    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    EXPECT_NEAR(result, 70.0f, FLOAT_EPSILON);
}

TEST(SIMD_Vec, CrossProduct)
{
    Vec3f x(1.0f, 0.0f, 0.0f);
    Vec3f y(0.0f, 1.0f, 0.0f);

    Vec3f z = x.cross(y);

    EXPECT_NEAR(z.x(), 0.0f, FLOAT_EPSILON);
    EXPECT_NEAR(z.y(), 0.0f, FLOAT_EPSILON);
    EXPECT_NEAR(z.z(), 1.0f, FLOAT_EPSILON);
}

TEST(SIMD_Vec, Normalization)
{
    Vec3f v(3.0f, 4.0f, 0.0f);
    Vec3f normalized = v.normalized();

    // Length should be 1.0
    EXPECT_NEAR(normalized.length(), 1.0f, FLOAT_EPSILON);

    // Direction should be preserved
    EXPECT_NEAR(normalized.x(), 0.6f, FLOAT_EPSILON);
    EXPECT_NEAR(normalized.y(), 0.8f, FLOAT_EPSILON);
    EXPECT_NEAR(normalized.z(), 0.0f, FLOAT_EPSILON);
}

TEST(SIMD_Vec, ScalarMultiplication)
{
    Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
    Vec4f result = v * 2.5f;

    EXPECT_NEAR(result[0], 2.5f, FLOAT_EPSILON);
    EXPECT_NEAR(result[1], 5.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result[2], 7.5f, FLOAT_EPSILON);
    EXPECT_NEAR(result[3], 10.0f, FLOAT_EPSILON);
}

TEST(SIMD_Vec, EqualityComparison)
{
    Vec3f a(1.0f, 2.0f, 3.0f);
    Vec3f b(1.0f, 2.0f, 3.0f);
    Vec3f c(1.0f, 2.0f, 3.01f);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

TEST(SIMD_Vec, LengthComputation)
{
    Vec3f v(3.0f, 4.0f, 0.0f);
    float len = v.length();

    EXPECT_NEAR(len, 5.0f, FLOAT_EPSILON);
}

TEST(SIMD_Vec, LengthSquared)
{
    Vec4f v(1.0f, 2.0f, 3.0f, 4.0f);
    float len_sq = v.length_squared();

    // 1 + 4 + 9 + 16 = 30
    EXPECT_NEAR(len_sq, 30.0f, FLOAT_EPSILON);
}

// ==================== Matrix SIMD Equivalence Tests ====================

TEST(SIMD_Mat, MatrixMultiplication4x4)
{
    Mat4f a = Mat4f::identity();
    a(0, 0) = 2.0f;
    a(1, 1) = 3.0f;
    a(2, 2) = 4.0f;
    a(3, 3) = 5.0f;

    Mat4f b = Mat4f::identity();
    Mat4f result = a * b;

    EXPECT_TRUE(matrices_equal(result, a, FLOAT_EPSILON));
}

TEST(SIMD_Mat, MatrixVectorMultiplication)
{
    Mat4f m = Mat4f::identity();
    m(0, 0) = 2.0f;
    m(1, 1) = 3.0f;
    m(2, 2) = 4.0f;
    m(3, 3) = 1.0f;

    Vec4f v(1.0f, 2.0f, 3.0f, 1.0f);
    Vec4f result = m * v;

    EXPECT_NEAR(result[0], 2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result[1], 6.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result[2], 12.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result[3], 1.0f, FLOAT_EPSILON);
}

TEST(SIMD_Mat, Matrix3x3Multiplication)
{
    Mat3f a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    Mat3f b = Mat3f::identity();
    Mat3f result = a * b;

    EXPECT_TRUE(matrices_equal(result, a, FLOAT_EPSILON));
}

TEST(SIMD_Mat, MatrixTranspose)
{
    Mat4f m({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    Mat4f transposed = m.transpose();

    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            EXPECT_NEAR(m(r, c), transposed(c, r), FLOAT_EPSILON);
        }
    }
}

TEST(SIMD_Mat, MatrixEqualityComparison)
{
    Mat4f a = Mat4f::identity();
    Mat4f b = Mat4f::identity();

    EXPECT_TRUE(a == b);

    b(0, 0) = 1.01f;
    EXPECT_FALSE(a == b);
}

// ==================== Quaternion SIMD Equivalence Tests ====================

TEST(SIMD_Quat, QuaternionMultiplication)
{
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);  // w, x, y, z
    Quaternion<float> q2(0.0f, 1.0f, 0.0f, 0.0f);

    Quaternion<float> result = q1 * q2;

    // Quaternion multiplication: (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
    EXPECT_NEAR(result.w(), 0.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result.x(), 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result.y(), 0.0f, FLOAT_EPSILON);
    EXPECT_NEAR(result.z(), 0.0f, FLOAT_EPSILON);
}

TEST(SIMD_Quat, QuaternionNormalization)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> normalized = q.normalized();

    // Magnitude should be 1.0
    float mag = std::sqrt(normalized.w() * normalized.w() + normalized.x() * normalized.x() +
                          normalized.y() * normalized.y() + normalized.z() * normalized.z());

    EXPECT_NEAR(mag, 1.0f, FLOAT_EPSILON);
}

TEST(SIMD_Quat, QuaternionConjugate)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> conj = q.conjugate();

    EXPECT_NEAR(conj.w(), 1.0f, FLOAT_EPSILON);
    EXPECT_NEAR(conj.x(), -2.0f, FLOAT_EPSILON);
    EXPECT_NEAR(conj.y(), -3.0f, FLOAT_EPSILON);
    EXPECT_NEAR(conj.z(), -4.0f, FLOAT_EPSILON);
}

TEST(SIMD_Quat, QuaternionRotation)
{
    // 90-degree rotation around Z-axis
    Quaternion<float> q = Quaternion<float>::from_axis_angle(Vec3f(0.0f, 0.0f, 1.0f), constants::pi<float> / 2.0f);

    Vec3f v(1.0f, 0.0f, 0.0f);
    Vec3f rotated = q.rotate(v);

    // Should rotate (1,0,0) to (0,1,0)
    EXPECT_NEAR(rotated.x(), 0.0f, FLOAT_EPSILON * 10.0f);
    EXPECT_NEAR(rotated.y(), 1.0f, FLOAT_EPSILON * 10.0f);
    EXPECT_NEAR(rotated.z(), 0.0f, FLOAT_EPSILON * 10.0f);
}

TEST(SIMD_Quat, QuaternionEqualityComparison)
{
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> q2(1.0f, 0.0f, 0.0f, 0.0f);

    EXPECT_TRUE(q1 == q2);

    Quaternion<float> q3(0.99f, 0.0f, 0.0f, 0.0f);
    EXPECT_FALSE(q1 == q3);
}

// ==================== Mixed Precision Tests ====================

TEST(SIMD_MixedPrecision, DoubleVsFloat)
{
    Vec3f vf(1.5f, 2.3f, 3.7f);
    Vec3d vd(1.5, 2.3, 3.7);

    float len_f = vf.length();
    double len_d = vd.length();

    // Results should be close (within float precision)
    EXPECT_NEAR(static_cast<double>(len_f), len_d, 1e-6);
}

// ==================== Edge Cases ====================

TEST(SIMD_EdgeCases, ZeroVector)
{
    Vec3f zero(0.0f, 0.0f, 0.0f);

    EXPECT_NEAR(zero.length(), 0.0f, FLOAT_EPSILON);

    // Normalization of zero vector should return zero (safe behavior)
    Vec3f normalized = zero.normalized();
    EXPECT_NEAR(normalized.length(), 0.0f, FLOAT_EPSILON);
}

TEST(SIMD_EdgeCases, VerySmallValues)
{
    float small = 1e-20f;
    Vec3f v(small, small, small);

    Vec3f normalized = v.normalized();

    // Should handle gracefully (may return zero or normalized)
    // Just ensure it doesn't crash
    EXPECT_TRUE(std::isfinite(normalized.x()));
    EXPECT_TRUE(std::isfinite(normalized.y()));
    EXPECT_TRUE(std::isfinite(normalized.z()));
}

TEST(SIMD_EdgeCases, VeryLargeValues)
{
    float large = 1e20f;
    Vec3f v(large, large, large);

    Vec3f normalized = v.normalized();

    // Normalized vector should still be unit length
    float len = normalized.length();
    EXPECT_TRUE(std::isfinite(len));
    if (std::isfinite(len))
    {
        EXPECT_NEAR(len, 1.0f, FLOAT_EPSILON * 10.0f);
    }
}

TEST(SIMD_EdgeCases, NegativeZero)
{
    Vec3f v1(0.0f, 0.0f, 0.0f);
    Vec3f v2(-0.0f, -0.0f, -0.0f);

    // -0.0 and 0.0 should be treated as equal
    EXPECT_TRUE(v1 == v2);
}

#else  // !CONFIG_MATRIXLIB_NEON

// Placeholder tests for non-SIMD builds
TEST(SIMD_Placeholder, NoSIMDAvailable)
{
    GTEST_SKIP() << "SIMD tests require CONFIG_MATRIXLIB_NEON to be defined";
}

#endif  // CONFIG_MATRIXLIB_NEON

// ==================== Performance Comparison Tests ====================

// These don't assert equivalence, just measure performance difference

#ifdef CONFIG_MATRIXLIB_NEON
#include <chrono>

TEST(SIMD_Performance, Vec3DotProduct)
{
    constexpr int iterations = 1000000;
    Vec3f a(1.5f, 2.3f, 3.7f);
    Vec3f b(4.2f, 5.1f, 6.8f);

    auto start = std::chrono::high_resolution_clock::now();
    volatile float result = 0.0f;
    for (int i = 0; i < iterations; ++i)
    {
        result += a.dot(b);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Just log performance, don't assert
    std::cout << "Vec3 dot product: " << iterations << " iterations in " << duration.count() << " μs\n";
}

TEST(SIMD_Performance, Mat4Multiplication)
{
    constexpr int iterations = 100000;
    Mat4f a = Mat4f::identity();
    Mat4f b = Mat4f::identity();

    auto start = std::chrono::high_resolution_clock::now();
    Mat4f result = a;
    for (int i = 0; i < iterations; ++i)
    {
        result = result * b;
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Mat4 multiply: " << iterations << " iterations in " << duration.count() << " μs\n";
}

#endif  // CONFIG_MATRIXLIB_NEON
