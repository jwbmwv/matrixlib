// SPDX-License-Identifier: MIT
/// @file test_quat.cpp
/// @brief MatrixLib Google Test Suite - Quaternion<T> Tests
/// @details Comprehensive unit tests for quaternion operations including construction,
///          multiplication, rotation, SLERP interpolation, and matrix conversion.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <gtest/gtest.h>
#include <matrixlib/quaternion.hpp>
#include <cmath>

using namespace matrixlib;

class QuatTest : public ::testing::Test
{
protected:
    static constexpr float epsilon = 0.0001f;

    bool float_eq(float a, float b) const { return std::fabs(a - b) < epsilon; }
};

TEST_F(QuatTest, Construction)
{
    Quaternion<float> q1;                          // Identity
    Quaternion<float> q2(1.0f, 0.0f, 0.0f, 0.0f);  // w, x, y, z

    EXPECT_FLOAT_EQ(q1.w(), 1.0f);
    EXPECT_FLOAT_EQ(q1.x(), 0.0f);
    EXPECT_FLOAT_EQ(q1.y(), 0.0f);
    EXPECT_FLOAT_EQ(q1.z(), 0.0f);

    EXPECT_FLOAT_EQ(q2.w(), 1.0f);
    EXPECT_FLOAT_EQ(q2.x(), 0.0f);
}

TEST_F(QuatTest, AxisAngleConstruction)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    float angle = 3.14159f / 2.0f;  // 90 degrees

    Quaternion<float> q(axis, angle);

    // For 90° around Z: w = cos(45°) ≈ 0.707, z = sin(45°) ≈ 0.707
    EXPECT_TRUE(float_eq(q.w(), 0.7071f));
    EXPECT_TRUE(float_eq(q.z(), 0.7071f));
    EXPECT_TRUE(float_eq(q.x(), 0.0f));
    EXPECT_TRUE(float_eq(q.y(), 0.0f));
}

TEST_F(QuatTest, Conjugate)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> qc = q.conjugate();

    EXPECT_FLOAT_EQ(qc.w(), 1.0f);
    EXPECT_FLOAT_EQ(qc.x(), -2.0f);
    EXPECT_FLOAT_EQ(qc.y(), -3.0f);
    EXPECT_FLOAT_EQ(qc.z(), -4.0f);
}

TEST_F(QuatTest, Norm)
{
    Quaternion<float> q(1.0f, 2.0f, 2.0f, 0.0f);
    float n = q.norm();

    // norm = sqrt(1 + 4 + 4 + 0) = sqrt(9) = 3
    EXPECT_FLOAT_EQ(n, 3.0f);
}

TEST_F(QuatTest, Normalization)
{
    Quaternion<float> q(2.0f, 2.0f, 2.0f, 2.0f);
    Quaternion<float> qn = q.normalized();

    float norm = qn.norm();
    EXPECT_TRUE(float_eq(norm, 1.0f));

    // Each component should be 0.5
    EXPECT_TRUE(float_eq(qn.w(), 0.5f));
    EXPECT_TRUE(float_eq(qn.x(), 0.5f));
    EXPECT_TRUE(float_eq(qn.y(), 0.5f));
    EXPECT_TRUE(float_eq(qn.z(), 0.5f));
}

TEST_F(QuatTest, Multiplication)
{
    Vec<float, 3> z_axis(0.0f, 0.0f, 1.0f);
    Vec<float, 3> x_axis(1.0f, 0.0f, 0.0f);

    Quaternion<float> q1(z_axis, 3.14159f / 2.0f);  // 90° around Z
    Quaternion<float> q2(x_axis, 3.14159f / 2.0f);  // 90° around X

    Quaternion<float> q3 = q1 * q2;

    float norm = q3.norm();
    EXPECT_TRUE(float_eq(norm, 1.0f));
}

TEST_F(QuatTest, Inverse)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 4.0f);

    Quaternion<float> qi = q.inverse();
    Quaternion<float> identity = q * qi;

    EXPECT_TRUE(float_eq(identity.w(), 1.0f));
    EXPECT_TRUE(float_eq(identity.x(), 0.0f));
    EXPECT_TRUE(float_eq(identity.y(), 0.0f));
    EXPECT_TRUE(float_eq(identity.z(), 0.0f));
}

TEST_F(QuatTest, RotateVector)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 2.0f);  // 90° around Z

    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q.rotate(v);

    // Should rotate X to Y
    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(QuatTest, ToRotationMatrix)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 2.0f);

    SquareMat<float, 3> R = q.to_rotation_matrix();
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = R * v;

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(QuatTest, FromRotationMatrix)
{
    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_z(3.14159f / 2.0f);
    Quaternion<float> q = Quaternion<float>::from_rotation_matrix(R);

    EXPECT_TRUE(float_eq(q.w(), 0.7071f));
    EXPECT_TRUE(float_eq(std::fabs(q.z()), 0.7071f));  // Can be ± due to representation
}

TEST_F(QuatTest, SLERP)
{
    Quaternion<float> q_start = Quaternion<float>::identity();

    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q_end(axis, 3.14159f / 2.0f);

    // Interpolate at t=0
    Quaternion<float> q0 = q_start.slerp(q_end, 0.0f);
    EXPECT_TRUE(float_eq(q0.w(), q_start.w()));
    EXPECT_TRUE(float_eq(q0.z(), q_start.z()));

    // Interpolate at t=1
    Quaternion<float> q1 = q_start.slerp(q_end, 1.0f);
    EXPECT_TRUE(float_eq(q1.w(), q_end.w()));
    EXPECT_TRUE(float_eq(q1.z(), q_end.z()));

    // Interpolate at t=0.5 (halfway = 45° rotation)
    Quaternion<float> q_mid = q_start.slerp(q_end, 0.5f);
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q_mid.rotate(v);

    // At 45°, x and y should be equal
    EXPECT_TRUE(float_eq(rotated[0], rotated[1]));
}

TEST_F(QuatTest, Addition)
{
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> q2(0.0f, 1.0f, 0.0f, 0.0f);

    Quaternion<float> sum = q1 + q2;

    EXPECT_FLOAT_EQ(sum.w(), 1.0f);
    EXPECT_FLOAT_EQ(sum.x(), 1.0f);
    EXPECT_FLOAT_EQ(sum.y(), 0.0f);
    EXPECT_FLOAT_EQ(sum.z(), 0.0f);
}

TEST_F(QuatTest, Subtraction)
{
    Quaternion<float> q1(2.0f, 3.0f, 4.0f, 5.0f);
    Quaternion<float> q2(1.0f, 1.0f, 1.0f, 1.0f);

    Quaternion<float> diff = q1 - q2;

    EXPECT_FLOAT_EQ(diff.w(), 1.0f);
    EXPECT_FLOAT_EQ(diff.x(), 2.0f);
    EXPECT_FLOAT_EQ(diff.y(), 3.0f);
    EXPECT_FLOAT_EQ(diff.z(), 4.0f);
}

TEST_F(QuatTest, ScalarMultiplication)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> scaled = q * 2.0f;

    EXPECT_FLOAT_EQ(scaled.w(), 2.0f);
    EXPECT_FLOAT_EQ(scaled.x(), 4.0f);
    EXPECT_FLOAT_EQ(scaled.y(), 6.0f);
    EXPECT_FLOAT_EQ(scaled.z(), 8.0f);
}

TEST_F(QuatTest, Identity)
{
    Quaternion<float> q = Quaternion<float>::identity();

    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> rotated = q.rotate(v);

    // Identity rotation should not change vector
    EXPECT_FLOAT_EQ(rotated[0], v[0]);
    EXPECT_FLOAT_EQ(rotated[1], v[1]);
    EXPECT_FLOAT_EQ(rotated[2], v[2]);
}

TEST_F(QuatTest, VecAccessor)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);

    Vec<float, 3>& v = q.vec();

    EXPECT_FLOAT_EQ(v[0], 2.0f);  // x
    EXPECT_FLOAT_EQ(v[1], 3.0f);  // y
    EXPECT_FLOAT_EQ(v[2], 4.0f);  // z

    // Modify through reference
    v[0] = 5.0f;
    EXPECT_FLOAT_EQ(q.x(), 5.0f);
}

TEST_F(QuatTest, CompoundAssignment)
{
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> a(1.0f, 1.0f, 1.0f, 1.0f);

    q += a;
    EXPECT_FLOAT_EQ(q.w(), 2.0f);
    EXPECT_FLOAT_EQ(q.x(), 1.0f);

    q -= a;
    EXPECT_FLOAT_EQ(q.w(), 1.0f);
    EXPECT_FLOAT_EQ(q.x(), 0.0f);

    q *= 2.0f;
    EXPECT_FLOAT_EQ(q.w(), 2.0f);
}

TEST_F(QuatTest, RotationComposition)
{
    // Rotate 90° around Z, then 90° around X
    Vec<float, 3> z_axis(0.0f, 0.0f, 1.0f);
    Vec<float, 3> x_axis(1.0f, 0.0f, 0.0f);

    Quaternion<float> q_z(z_axis, 3.14159f / 2.0f);
    Quaternion<float> q_x(x_axis, 3.14159f / 2.0f);

    Quaternion<float> q_combined = q_x * q_z;

    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q_combined.rotate(v);

    // First rotates X to Y (by q_z), then Y to Z (by q_x)
    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 0.0f));
    EXPECT_TRUE(float_eq(rotated[2], 1.0f));
}

TEST_F(QuatTest, MatrixRoundTrip)
{
    Vec<float, 3> axis(1.0f, 1.0f, 1.0f);
    axis = axis.normalized();

    Quaternion<float> q_orig(axis, 1.5f);
    SquareMat<float, 3> R = q_orig.to_rotation_matrix();
    Quaternion<float> q_reconstructed = Quaternion<float>::from_rotation_matrix(R);

    // Quaternions q and -q represent the same rotation
    bool same_or_opposite = (float_eq(q_orig.w(), q_reconstructed.w()) && float_eq(q_orig.x(), q_reconstructed.x())) ||
                            (float_eq(q_orig.w(), -q_reconstructed.w()) && float_eq(q_orig.x(), -q_reconstructed.x()));

    EXPECT_TRUE(same_or_opposite);
}
