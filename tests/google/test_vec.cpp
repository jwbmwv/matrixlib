// SPDX-License-Identifier: MIT
/// @file test_vec.cpp
/// @brief MatrixLib Google Test Suite - Vec<T,N> Tests
/// @details Comprehensive unit tests for vector operations including arithmetic,
///          dot/cross products, magnitude, normalization, projections, and rotations.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <cmath>

using namespace matrixlib;

class VecTest : public ::testing::Test
{
protected:
    static constexpr float epsilon = 0.0001f;

    bool float_eq(float a, float b) const { return std::fabs(a - b) < epsilon; }
};

TEST_F(VecTest, Construction)
{
    Vec<float, 3> v1;
    Vec<float, 3> v2(1.0f, 2.0f, 3.0f);
    Vec<float, 3> v3(v2);

    EXPECT_FLOAT_EQ(v2[Vec<float, 3>::X], 1.0f);
    EXPECT_FLOAT_EQ(v2[Vec<float, 3>::Y], 2.0f);
    EXPECT_FLOAT_EQ(v2[Vec<float, 3>::Z], 3.0f);

    EXPECT_FLOAT_EQ(v3[0], v2[0]);
    EXPECT_FLOAT_EQ(v3[1], v2[1]);
    EXPECT_FLOAT_EQ(v3[2], v2[2]);
}

TEST_F(VecTest, Addition)
{
    Vec<float, 3> a(1.0f, 2.0f, 3.0f);
    Vec<float, 3> b(4.0f, 5.0f, 6.0f);
    Vec<float, 3> sum = a + b;

    EXPECT_FLOAT_EQ(sum[0], 5.0f);
    EXPECT_FLOAT_EQ(sum[1], 7.0f);
    EXPECT_FLOAT_EQ(sum[2], 9.0f);
}

TEST_F(VecTest, Subtraction)
{
    Vec<float, 3> a(4.0f, 5.0f, 6.0f);
    Vec<float, 3> b(1.0f, 2.0f, 3.0f);
    Vec<float, 3> diff = a - b;

    EXPECT_FLOAT_EQ(diff[0], 3.0f);
    EXPECT_FLOAT_EQ(diff[1], 3.0f);
    EXPECT_FLOAT_EQ(diff[2], 3.0f);
}

TEST_F(VecTest, ScalarMultiplication)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> scaled = v * 2.0f;
    Vec<float, 3> scaled2 = 2.0f * v;

    EXPECT_FLOAT_EQ(scaled[0], 2.0f);
    EXPECT_FLOAT_EQ(scaled[1], 4.0f);
    EXPECT_FLOAT_EQ(scaled[2], 6.0f);

    EXPECT_FLOAT_EQ(scaled2[0], 2.0f);
    EXPECT_FLOAT_EQ(scaled2[1], 4.0f);
    EXPECT_FLOAT_EQ(scaled2[2], 6.0f);
}

TEST_F(VecTest, ScalarDivision)
{
    Vec<float, 3> v(2.0f, 4.0f, 6.0f);
    Vec<float, 3> divided = v / 2.0f;

    EXPECT_FLOAT_EQ(divided[0], 1.0f);
    EXPECT_FLOAT_EQ(divided[1], 2.0f);
    EXPECT_FLOAT_EQ(divided[2], 3.0f);
}

TEST_F(VecTest, DotProduct)
{
    Vec<float, 3> a(1.0f, 0.0f, 0.0f);
    Vec<float, 3> b(0.0f, 1.0f, 0.0f);
    Vec<float, 3> c(1.0f, 2.0f, 3.0f);
    Vec<float, 3> d(4.0f, 5.0f, 6.0f);

    float dot1 = a.dot(b);
    float dot2 = c.dot(d);

    EXPECT_FLOAT_EQ(dot1, 0.0f);
    EXPECT_FLOAT_EQ(dot2, 32.0f);  // 1*4 + 2*5 + 3*6 = 32
}

TEST_F(VecTest, CrossProduct)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> z = x.cross(y);

    EXPECT_TRUE(float_eq(z[Vec<float, 3>::X], 0.0f));
    EXPECT_TRUE(float_eq(z[Vec<float, 3>::Y], 0.0f));
    EXPECT_TRUE(float_eq(z[Vec<float, 3>::Z], 1.0f));

    // Anti-commutativity: y × x = -(x × y)
    Vec<float, 3> z_rev = y.cross(x);
    EXPECT_TRUE(float_eq(z_rev[Vec<float, 3>::Z], -1.0f));
}

TEST_F(VecTest, Magnitude)
{
    Vec<float, 3> v(3.0f, 4.0f, 0.0f);
    float mag = v.magnitude();
    float mag_sq = v.magnitude_squared();

    EXPECT_FLOAT_EQ(mag, 5.0f);
    EXPECT_FLOAT_EQ(mag_sq, 25.0f);
}

TEST_F(VecTest, Normalization)
{
    Vec<float, 3> v(3.0f, 4.0f, 0.0f);
    Vec<float, 3> n = v.normalized();
    float mag = n.magnitude();

    EXPECT_TRUE(float_eq(mag, 1.0f));
    EXPECT_TRUE(float_eq(n[0], 0.6f));
    EXPECT_TRUE(float_eq(n[1], 0.8f));

    // In-place normalization
    Vec<float, 3> v2(3.0f, 4.0f, 0.0f);
    v2.normalize();
    EXPECT_TRUE(float_eq(v2.magnitude(), 1.0f));
}

TEST_F(VecTest, Angle)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> diag(1.0f, 1.0f, 0.0f);

    float angle1 = x.angle(y);
    float angle2 = x.angle(diag);

    EXPECT_TRUE(float_eq(angle1, 3.14159f / 2.0f));  // 90 degrees
    EXPECT_TRUE(float_eq(angle2, 3.14159f / 4.0f));  // 45 degrees
}

TEST_F(VecTest, ProjectionAndRejection)
{
    Vec<float, 3> a(3.0f, 4.0f, 0.0f);
    Vec<float, 3> b(1.0f, 0.0f, 0.0f);

    Vec<float, 3> proj = a.project(b);
    Vec<float, 3> rej = a.reject(b);

    // Projection should be along b
    EXPECT_TRUE(float_eq(proj[0], 3.0f));
    EXPECT_TRUE(float_eq(proj[1], 0.0f));

    // Rejection should be perpendicular to b
    EXPECT_TRUE(float_eq(rej[0], 0.0f));
    EXPECT_TRUE(float_eq(rej[1], 4.0f));

    // proj + rej should equal original
    Vec<float, 3> sum = proj + rej;
    EXPECT_TRUE(float_eq(sum[0], a[0]));
    EXPECT_TRUE(float_eq(sum[1], a[1]));
}

TEST_F(VecTest, SignedAngle)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> z(0.0f, 0.0f, 1.0f);

    float angle_xy_z = x.signed_angle(y, z);
    float angle_yx_z = y.signed_angle(x, z);

    EXPECT_GT(angle_xy_z, 0.0f);  // Positive rotation from x to y around z
    EXPECT_LT(angle_yx_z, 0.0f);  // Negative rotation from y to x around z
}

TEST_F(VecTest, Accessors)
{
    Vec<float, 4> v(1.0f, 2.0f, 3.0f, 4.0f);

    EXPECT_FLOAT_EQ(v.x(), 1.0f);
    EXPECT_FLOAT_EQ(v.y(), 2.0f);
    EXPECT_FLOAT_EQ(v.z(), 3.0f);
    EXPECT_FLOAT_EQ(v.w(), 4.0f);

    // Mutable accessors
    v.x() = 5.0f;
    EXPECT_FLOAT_EQ(v.x(), 5.0f);
    EXPECT_FLOAT_EQ(v[0], 5.0f);
}

TEST_F(VecTest, CompoundAssignment)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> a(1.0f, 1.0f, 1.0f);

    v += a;
    EXPECT_FLOAT_EQ(v[0], 2.0f);
    EXPECT_FLOAT_EQ(v[1], 3.0f);

    v -= a;
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);

    v *= 2.0f;
    EXPECT_FLOAT_EQ(v[0], 2.0f);
    EXPECT_FLOAT_EQ(v[1], 4.0f);

    v /= 2.0f;
    EXPECT_FLOAT_EQ(v[0], 1.0f);
    EXPECT_FLOAT_EQ(v[1], 2.0f);
}

TEST_F(VecTest, Negation)
{
    Vec<float, 3> v(1.0f, -2.0f, 3.0f);
    Vec<float, 3> neg = -v;

    EXPECT_FLOAT_EQ(neg[0], -1.0f);
    EXPECT_FLOAT_EQ(neg[1], 2.0f);
    EXPECT_FLOAT_EQ(neg[2], -3.0f);
}

TEST_F(VecTest, ElementwiseMultiplication)
{
    Vec<float, 3> a(2.0f, 3.0f, 4.0f);
    Vec<float, 3> b(5.0f, 6.0f, 7.0f);
    Vec<float, 3> prod = a.elem_mult(b);

    EXPECT_FLOAT_EQ(prod[0], 10.0f);
    EXPECT_FLOAT_EQ(prod[1], 18.0f);
    EXPECT_FLOAT_EQ(prod[2], 28.0f);
}

TEST_F(VecTest, Distance)
{
    Vec<float, 3> a(0.0f, 0.0f, 0.0f);
    Vec<float, 3> b(3.0f, 4.0f, 0.0f);

    float dist = a.distance(b);

    EXPECT_FLOAT_EQ(dist, 5.0f);
}

TEST_F(VecTest, Rotation2D)
{
    Vec<float, 2> v(1.0f, 0.0f);
    Vec<float, 2> rotated = v.rotate(3.14159f / 2.0f);  // 90 degrees

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
}

TEST_F(VecTest, Rotation3D)
{
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Vec<float, 3> rotated = v.rotate(axis, 3.14159f / 2.0f);  // 90 degrees around Z

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(VecTest, SizeMethod)
{
    // Test compile-time size() method
    Vec<float, 2> v2;
    Vec<float, 3> v3;
    Vec<float, 4> v4;
    Vec<double, 5> v5;

    EXPECT_EQ(v2.size(), 2u);
    EXPECT_EQ(v3.size(), 3u);
    EXPECT_EQ(v4.size(), 4u);
    EXPECT_EQ(v5.size(), 5u);

    // Should be constexpr - can be used in compile-time contexts
    static_assert(Vec<float, 3>::size() == 3, "size() should be constexpr");
    static_assert(Vec<int, 10>::size() == 10, "size() should work for any dimension");
}
