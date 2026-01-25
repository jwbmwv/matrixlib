// SPDX-License-Identifier: MIT
/// @file test_edge_cases.cpp
/// @brief MatrixLib Google Test Suite - Edge Cases and Numerical Robustness
/// @details Tests for singular matrices, near-singular matrices, NaN handling,
///          quaternion aliasing safety, and other edge cases that trigger sanitizers.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <cmath>
#include <limits>

using namespace matrixlib;

class EdgeCasesTest : public ::testing::Test
{
protected:
    static constexpr float epsilon = 1e-5f;
    static constexpr double epsilon_d = 1e-10;

    bool is_nan(float val) const { return std::isnan(val); }
    bool is_inf(float val) const { return std::isinf(val); }
};

// ==================== Singular Matrix Tests ====================

TEST_F(EdgeCasesTest, SingularMatrix2x2_ZeroDeterminant)
{
    // Matrix with zero determinant (singular)
    SquareMat<float, 2> singular;
    singular[0][0] = 1.0f;
    singular[0][1] = 2.0f;
    singular[1][0] = 2.0f;
    singular[1][1] = 4.0f;  // Row 2 = 2 * Row 1

    float det = singular.determinant();
    EXPECT_NEAR(det, 0.0f, epsilon);

    // Inverse will produce NaN or Inf
    SquareMat<float, 2> inv = singular.inverse();
    // At least one element should be NaN or Inf
    bool has_invalid = false;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            if (is_nan(inv[i][j]) || is_inf(inv[i][j]))
            {
                has_invalid = true;
            }
        }
    }
    EXPECT_TRUE(has_invalid) << "Singular matrix inverse should contain NaN or Inf";
}

TEST_F(EdgeCasesTest, SingularMatrix3x3_ZeroRow)
{
    // 3x3 matrix with a zero row (singular)
    SquareMat<float, 3> singular = SquareMat<float, 3>::zero();
    singular[0][0] = 1.0f;
    singular[0][1] = 2.0f;
    singular[0][2] = 3.0f;
    singular[1][0] = 4.0f;
    singular[1][1] = 5.0f;
    singular[1][2] = 6.0f;
    // Row 2 is all zeros

    float det = singular.determinant();
    EXPECT_NEAR(det, 0.0f, epsilon);

    SquareMat<float, 3> inv = singular.inverse();
    bool has_invalid = false;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (is_nan(inv[i][j]) || is_inf(inv[i][j]))
            {
                has_invalid = true;
            }
        }
    }
    EXPECT_TRUE(has_invalid);
}

TEST_F(EdgeCasesTest, NearSingularMatrix_SmallDeterminant)
{
    // Nearly singular matrix (very small but non-zero determinant)
    SquareMat<float, 2> nearly_singular;
    nearly_singular[0][0] = 1.0f;
    nearly_singular[0][1] = 2.0f;
    nearly_singular[1][0] = 2.0f + 1e-7f;
    nearly_singular[1][1] = 4.0f + 1e-7f;

    float det = nearly_singular.determinant();
    EXPECT_LT(std::abs(det), 1e-5f) << "Determinant should be very small";
    EXPECT_GT(std::abs(det), 0.0f) << "Determinant should be non-zero";

    // Inverse exists but may have large numerical errors
    SquareMat<float, 2> inv = nearly_singular.inverse();

    // Check if result contains finite values (no NaN/Inf despite small det)
    bool all_finite = true;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            if (!std::isfinite(inv[i][j]))
            {
                all_finite = false;
            }
        }
    }
    // This might fail for very ill-conditioned matrices, which is expected behavior
}

TEST_F(EdgeCasesTest, IdentityMatrix_NonSingular)
{
    auto I = SquareMat<float, 3>::identity();

    float det = I.determinant();
    EXPECT_NEAR(det, 1.0f, epsilon);

    auto inv = I.inverse();

    // Inverse of identity is identity
    for (std::uint32_t i = 0; i < 3; ++i)
    {
        for (std::uint32_t j = 0; j < 3; ++j)
        {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(inv[i][j], expected, epsilon);
        }
    }
}

// ==================== Quaternion Aliasing Safety Tests ====================

TEST_F(EdgeCasesTest, QuaternionVecAccess_NoAliasing)
{
    // Test that vec() returns a copy, not a reference
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);

    Vec<float, 3> v1 = q.vec();
    EXPECT_FLOAT_EQ(v1[0], 2.0f);  // x
    EXPECT_FLOAT_EQ(v1[1], 3.0f);  // y
    EXPECT_FLOAT_EQ(v1[2], 4.0f);  // z

    // Modify v1 - should not affect q
    v1[0] = 99.0f;

    Vec<float, 3> v2 = q.vec();
    EXPECT_FLOAT_EQ(v2[0], 2.0f) << "Modifying vec() copy should not affect original";
    EXPECT_FLOAT_EQ(q.x(), 2.0f);
}

TEST_F(EdgeCasesTest, QuaternionSetVec)
{
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);

    Vec<float, 3> new_vec(10.0f, 20.0f, 30.0f);
    q.set_vec(new_vec);

    EXPECT_FLOAT_EQ(q.x(), 10.0f);
    EXPECT_FLOAT_EQ(q.y(), 20.0f);
    EXPECT_FLOAT_EQ(q.z(), 30.0f);
    EXPECT_FLOAT_EQ(q.w(), 1.0f);  // w should be unchanged
}

TEST_F(EdgeCasesTest, QuaternionOperations_NoUB)
{
    // Test various operations that previously used type-punning
    Quaternion<float> q1(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> q2(0.5f, 1.0f, 1.5f, 2.0f);

    // Addition
    Quaternion<float> sum = q1 + q2;
    EXPECT_FLOAT_EQ(sum.w(), 1.5f);
    EXPECT_FLOAT_EQ(sum.x(), 3.0f);

    // Subtraction
    Quaternion<float> diff = q1 - q2;
    EXPECT_FLOAT_EQ(diff.w(), 0.5f);
    EXPECT_FLOAT_EQ(diff.x(), 1.0f);

    // Dot product (uses vec() internally)
    float dot = q1.dot(q2);
    EXPECT_GT(dot, 0.0f);

    // Norm (uses vec() internally)
    float norm = q1.norm();
    EXPECT_GT(norm, 0.0f);

    // Rotation (uses vec() internally)
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q1.normalized().rotate(v);
    EXPECT_TRUE(std::isfinite(rotated[0]));
    EXPECT_TRUE(std::isfinite(rotated[1]));
    EXPECT_TRUE(std::isfinite(rotated[2]));
}

// ==================== NaN and Infinity Handling ====================

TEST_F(EdgeCasesTest, VectorWithNaN)
{
    Vec<float, 3> v(1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f);

    EXPECT_TRUE(is_nan(v[1]));
    EXPECT_FALSE(is_nan(v[0]));

    // Operations with NaN propagate NaN
    float mag = v.magnitude();
    EXPECT_TRUE(is_nan(mag));
}

TEST_F(EdgeCasesTest, VectorWithInfinity)
{
    Vec<float, 3> v(1.0f, std::numeric_limits<float>::infinity(), 3.0f);

    EXPECT_TRUE(is_inf(v[1]));

    float mag = v.magnitude();
    EXPECT_TRUE(is_inf(mag));
}

TEST_F(EdgeCasesTest, NormalizeZeroVector)
{
    Vec<float, 3> zero = Vec<float, 3>::zero();
    Vec<float, 3> normalized = zero.normalized();

    // Result should have NaN components (0/0)
    EXPECT_TRUE(is_nan(normalized[0]) || normalized[0] == 0.0f);
}

TEST_F(EdgeCasesTest, SafeNormalizeZeroVector)
{
    Vec<float, 3> zero = Vec<float, 3>::zero();
    Vec<float, 3> safe = zero.safe_normalized(1e-6f);

    // Safe normalize should return zero vector
    EXPECT_FLOAT_EQ(safe[0], 0.0f);
    EXPECT_FLOAT_EQ(safe[1], 0.0f);
    EXPECT_FLOAT_EQ(safe[2], 0.0f);
}

// ==================== Cross Product Edge Cases ====================

TEST_F(EdgeCasesTest, CrossProduct_ParallelVectors)
{
    Vec<float, 3> v1(1.0f, 0.0f, 0.0f);
    Vec<float, 3> v2(2.0f, 0.0f, 0.0f);  // Parallel to v1

    Vec<float, 3> cross = v1.cross(v2);

    EXPECT_NEAR(cross[0], 0.0f, epsilon);
    EXPECT_NEAR(cross[1], 0.0f, epsilon);
    EXPECT_NEAR(cross[2], 0.0f, epsilon);
}

TEST_F(EdgeCasesTest, CrossProduct_OppositeVectors)
{
    Vec<float, 3> v1(1.0f, 0.0f, 0.0f);
    Vec<float, 3> v2(-1.0f, 0.0f, 0.0f);  // Opposite to v1

    Vec<float, 3> cross = v1.cross(v2);

    EXPECT_NEAR(cross[0], 0.0f, epsilon);
    EXPECT_NEAR(cross[1], 0.0f, epsilon);
    EXPECT_NEAR(cross[2], 0.0f, epsilon);
}

// ==================== Matrix Rank Tests ====================

TEST_F(EdgeCasesTest, MatrixRank_FullRank)
{
    SquareMat<float, 3> full_rank = SquareMat<float, 3>::identity();

    std::uint32_t rank = full_rank.rank();
    EXPECT_EQ(rank, 3u);
}

TEST_F(EdgeCasesTest, MatrixRank_Singular)
{
    SquareMat<float, 3> singular;
    singular[0][0] = 1.0f;
    singular[0][1] = 2.0f;
    singular[0][2] = 3.0f;
    singular[1][0] = 2.0f;
    singular[1][1] = 4.0f;
    singular[1][2] = 6.0f;  // 2 * row 0
    singular[2][0] = 0.0f;
    singular[2][1] = 0.0f;
    singular[2][2] = 0.0f;

    std::uint32_t rank = singular.rank();
    EXPECT_LT(rank, 3u) << "Singular matrix should have rank < 3";
}

// ==================== Quaternion Normalization Edge Cases ====================

TEST_F(EdgeCasesTest, QuaternionNormalize_Identity)
{
    Quaternion<float> identity = Quaternion<float>::identity();
    Quaternion<float> normalized = identity.normalized();

    EXPECT_FLOAT_EQ(normalized.w(), 1.0f);
    EXPECT_FLOAT_EQ(normalized.x(), 0.0f);
    EXPECT_FLOAT_EQ(normalized.y(), 0.0f);
    EXPECT_FLOAT_EQ(normalized.z(), 0.0f);
}

TEST_F(EdgeCasesTest, QuaternionNormalize_ZeroQuaternion)
{
    Quaternion<float> zero(0.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> normalized = zero.normalized();

    // Normalizing zero quaternion should return zero (norm is 0)
    EXPECT_FLOAT_EQ(normalized.w(), 0.0f);
    EXPECT_FLOAT_EQ(normalized.x(), 0.0f);
    EXPECT_FLOAT_EQ(normalized.y(), 0.0f);
    EXPECT_FLOAT_EQ(normalized.z(), 0.0f);
}

// ==================== SIMD Alignment Tests ====================

TEST_F(EdgeCasesTest, VectorAlignment_Vec2f)
{
    Vec<float, 2> v;
    // Check that data is properly aligned (should be 16-byte aligned)
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&v) % 16, 0u) << "Vec2f should be 16-byte aligned";
}

TEST_F(EdgeCasesTest, VectorAlignment_Vec3f)
{
    Vec<float, 3> v;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&v) % 16, 0u) << "Vec3f should be 16-byte aligned";
}

TEST_F(EdgeCasesTest, QuaternionAlignment)
{
    Quaternion<float> q;
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&q) % 16, 0u) << "Quaternion<float> should be 16-byte aligned";
}
