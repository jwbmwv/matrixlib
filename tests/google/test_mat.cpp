// SPDX-License-Identifier: MIT
/// @file test_mat.cpp
/// @brief MatrixLib Google Test Suite - Mat<T,R,C> Tests
/// @details Comprehensive unit tests for matrix operations including multiplication,
///          transpose, determinant, inverse, rotations, and Euler angle extraction.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <cmath>

using namespace matrixlib;

class MatTest : public ::testing::Test
{
protected:
    static constexpr float epsilon = 0.0001f;

    bool float_eq(float a, float b) const { return std::fabs(a - b) < epsilon; }
};

TEST_F(MatTest, Construction)
{
    Mat<float, 2, 2> m1;
    SquareMat<float, 3> m2 = SquareMat<float, 3>::identity();

    EXPECT_FLOAT_EQ(m2[0][0], 1.0f);
    EXPECT_FLOAT_EQ(m2[0][1], 0.0f);
    EXPECT_FLOAT_EQ(m2[1][1], 1.0f);
    EXPECT_FLOAT_EQ(m2[2][2], 1.0f);
}

TEST_F(MatTest, Addition)
{
    Mat<float, 2, 2> a;
    a[0][0] = 1.0f;
    a[0][1] = 2.0f;
    a[1][0] = 3.0f;
    a[1][1] = 4.0f;

    Mat<float, 2, 2> b;
    b[0][0] = 5.0f;
    b[0][1] = 6.0f;
    b[1][0] = 7.0f;
    b[1][1] = 8.0f;

    Mat<float, 2, 2> sum = a + b;

    EXPECT_FLOAT_EQ(sum[0][0], 6.0f);
    EXPECT_FLOAT_EQ(sum[0][1], 8.0f);
    EXPECT_FLOAT_EQ(sum[1][0], 10.0f);
    EXPECT_FLOAT_EQ(sum[1][1], 12.0f);
}

TEST_F(MatTest, Subtraction)
{
    Mat<float, 2, 2> a;
    a[0][0] = 5.0f;
    a[0][1] = 6.0f;
    a[1][0] = 7.0f;
    a[1][1] = 8.0f;

    Mat<float, 2, 2> b;
    b[0][0] = 1.0f;
    b[0][1] = 2.0f;
    b[1][0] = 3.0f;
    b[1][1] = 4.0f;

    Mat<float, 2, 2> diff = a - b;

    EXPECT_FLOAT_EQ(diff[0][0], 4.0f);
    EXPECT_FLOAT_EQ(diff[0][1], 4.0f);
    EXPECT_FLOAT_EQ(diff[1][0], 4.0f);
    EXPECT_FLOAT_EQ(diff[1][1], 4.0f);
}

TEST_F(MatTest, MatrixMultiplication)
{
    Mat<float, 2, 3> a;
    a[0][0] = 1.0f;
    a[0][1] = 2.0f;
    a[0][2] = 3.0f;
    a[1][0] = 4.0f;
    a[1][1] = 5.0f;
    a[1][2] = 6.0f;

    Mat<float, 3, 2> b;
    b[0][0] = 7.0f;
    b[0][1] = 8.0f;
    b[1][0] = 9.0f;
    b[1][1] = 10.0f;
    b[2][0] = 11.0f;
    b[2][1] = 12.0f;

    Mat<float, 2, 2> c = a * b;

    EXPECT_FLOAT_EQ(c[0][0], 58.0f);   // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(c[0][1], 64.0f);   // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(c[1][0], 139.0f);  // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(c[1][1], 154.0f);  // 4*8 + 5*10 + 6*12
}

TEST_F(MatTest, VectorMultiplication)
{
    SquareMat<float, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    Vec<float, 2> v(5.0f, 6.0f);
    Vec<float, 2> result = m * v;

    EXPECT_FLOAT_EQ(result[0], 17.0f);  // 1*5 + 2*6
    EXPECT_FLOAT_EQ(result[1], 39.0f);  // 3*5 + 4*6
}

TEST_F(MatTest, ScalarMultiplication)
{
    Mat<float, 2, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    Mat<float, 2, 2> scaled = m * 2.0f;

    EXPECT_FLOAT_EQ(scaled[0][0], 2.0f);
    EXPECT_FLOAT_EQ(scaled[0][1], 4.0f);
    EXPECT_FLOAT_EQ(scaled[1][0], 6.0f);
    EXPECT_FLOAT_EQ(scaled[1][1], 8.0f);
}

TEST_F(MatTest, Transpose)
{
    Mat<float, 2, 3> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[0][2] = 3.0f;
    m[1][0] = 4.0f;
    m[1][1] = 5.0f;
    m[1][2] = 6.0f;

    Mat<float, 3, 2> mt = m.transpose();

    EXPECT_FLOAT_EQ(mt[0][0], 1.0f);
    EXPECT_FLOAT_EQ(mt[0][1], 4.0f);
    EXPECT_FLOAT_EQ(mt[1][0], 2.0f);
    EXPECT_FLOAT_EQ(mt[1][1], 5.0f);
    EXPECT_FLOAT_EQ(mt[2][0], 3.0f);
    EXPECT_FLOAT_EQ(mt[2][1], 6.0f);
}

TEST_F(MatTest, Determinant2x2)
{
    SquareMat<float, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    float det = m.determinant();

    EXPECT_FLOAT_EQ(det, -2.0f);  // 1*4 - 2*3 = -2
}

TEST_F(MatTest, Determinant3x3)
{
    SquareMat<float, 3> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[0][2] = 3.0f;
    m[1][0] = 0.0f;
    m[1][1] = 1.0f;
    m[1][2] = 4.0f;
    m[2][0] = 5.0f;
    m[2][1] = 6.0f;
    m[2][2] = 0.0f;

    float det = m.determinant();

    EXPECT_FLOAT_EQ(det, 1.0f);
}

TEST_F(MatTest, Trace)
{
    SquareMat<float, 3> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[0][2] = 3.0f;
    m[1][0] = 4.0f;
    m[1][1] = 5.0f;
    m[1][2] = 6.0f;
    m[2][0] = 7.0f;
    m[2][1] = 8.0f;
    m[2][2] = 9.0f;

    float trace = m.trace();

    EXPECT_FLOAT_EQ(trace, 15.0f);  // 1 + 5 + 9
}

TEST_F(MatTest, Inverse2x2)
{
    SquareMat<float, 2> m;
    m[0][0] = 4.0f;
    m[0][1] = 7.0f;
    m[1][0] = 2.0f;
    m[1][1] = 6.0f;

    SquareMat<float, 2> inv = m.inverse();
    SquareMat<float, 2> identity = m * inv;

    EXPECT_TRUE(float_eq(identity[0][0], 1.0f));
    EXPECT_TRUE(float_eq(identity[0][1], 0.0f));
    EXPECT_TRUE(float_eq(identity[1][0], 0.0f));
    EXPECT_TRUE(float_eq(identity[1][1], 1.0f));
}

TEST_F(MatTest, Rotation2D)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 2> R = SquareMat<float, 2>::rotation(angle);

    Vec<float, 2> v(1.0f, 0.0f);
    Vec<float, 2> rotated = R * v;

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
}

TEST_F(MatTest, Rotation3D_X)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Rx = SquareMat<float, 3>::rotation_x(angle);

    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> rotated = Rx * y;

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 0.0f));
    EXPECT_TRUE(float_eq(rotated[2], 1.0f));
}

TEST_F(MatTest, Rotation3D_Y)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Ry = SquareMat<float, 3>::rotation_y(angle);

    Vec<float, 3> z(0.0f, 0.0f, 1.0f);
    Vec<float, 3> rotated = Ry * z;

    EXPECT_TRUE(float_eq(rotated[0], 1.0f));
    EXPECT_TRUE(float_eq(rotated[1], 0.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(MatTest, Rotation3D_Z)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Rz = SquareMat<float, 3>::rotation_z(angle);

    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = Rz * x;

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(MatTest, RotationAxisAngle)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    float angle = 3.14159f / 2.0f;  // 90 degrees

    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_axis_angle(axis, angle);
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = R * x;

    EXPECT_TRUE(float_eq(rotated[0], 0.0f));
    EXPECT_TRUE(float_eq(rotated[1], 1.0f));
    EXPECT_TRUE(float_eq(rotated[2], 0.0f));
}

TEST_F(MatTest, RotationFromTo)
{
    Vec<float, 3> from(1.0f, 0.0f, 0.0f);
    Vec<float, 3> to(0.0f, 1.0f, 0.0f);

    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_from_to(from, to);
    Vec<float, 3> result = R * from;

    EXPECT_TRUE(float_eq(result[0], to[0]));
    EXPECT_TRUE(float_eq(result[1], to[1]));
    EXPECT_TRUE(float_eq(result[2], to[2]));
}

TEST_F(MatTest, LookAt)
{
    Vec<float, 3> target(1.0f, 0.0f, 0.0f);
    Vec<float, 3> up(0.0f, 0.0f, 1.0f);

    SquareMat<float, 3> look = SquareMat<float, 3>::look_at(target, up);

    // First column should be normalized target direction
    Vec<float, 3> forward(look[0][0], look[1][0], look[2][0]);
    float mag = forward.magnitude();

    EXPECT_TRUE(float_eq(mag, 1.0f));
    EXPECT_TRUE(float_eq(forward[0], 1.0f));
}

TEST_F(MatTest, EulerAngles)
{
    float roll = 0.1f;
    float pitch = 0.2f;
    float yaw = 0.3f;

    SquareMat<float, 3> Rx = SquareMat<float, 3>::rotation_x(roll);
    SquareMat<float, 3> Ry = SquareMat<float, 3>::rotation_y(pitch);
    SquareMat<float, 3> Rz = SquareMat<float, 3>::rotation_z(yaw);

    SquareMat<float, 3> R = Rz * Ry * Rx;
    Vec<float, 3> angles = R.euler_angles();

    EXPECT_TRUE(float_eq(angles[0], roll));
    EXPECT_TRUE(float_eq(angles[1], pitch));
    EXPECT_TRUE(float_eq(angles[2], yaw));
}

TEST_F(MatTest, CompoundAssignment)
{
    Mat<float, 2, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    Mat<float, 2, 2> a;
    a[0][0] = 1.0f;
    a[0][1] = 1.0f;
    a[1][0] = 1.0f;
    a[1][1] = 1.0f;

    m += a;
    EXPECT_FLOAT_EQ(m[0][0], 2.0f);
    EXPECT_FLOAT_EQ(m[1][1], 5.0f);

    m -= a;
    EXPECT_FLOAT_EQ(m[0][0], 1.0f);
    EXPECT_FLOAT_EQ(m[1][1], 4.0f);

    m *= 2.0f;
    EXPECT_FLOAT_EQ(m[0][0], 2.0f);
    EXPECT_FLOAT_EQ(m[1][1], 8.0f);
}

TEST_F(MatTest, SizeRowsColsMethods)
{
    // Test compile-time size(), rows(), cols() methods
    Mat<float, 2, 3> m23;  // 2 rows, 3 columns
    Mat<float, 3, 2> m32;  // 3 rows, 2 columns
    Mat<float, 4, 4> m44;  // 4x4 square matrix

    // Test size() - total element count
    EXPECT_EQ(m23.size(), 6u);   // 2 * 3 = 6
    EXPECT_EQ(m32.size(), 6u);   // 3 * 2 = 6
    EXPECT_EQ(m44.size(), 16u);  // 4 * 4 = 16

    // Test rows()
    EXPECT_EQ(m23.rows(), 2u);
    EXPECT_EQ(m32.rows(), 3u);
    EXPECT_EQ(m44.rows(), 4u);

    // Test cols()
    EXPECT_EQ(m23.cols(), 3u);
    EXPECT_EQ(m32.cols(), 2u);
    EXPECT_EQ(m44.cols(), 4u);

    // Should be constexpr - can be used in compile-time contexts
    static_assert(Mat<float, 3, 4>::size() == 12, "size() should be constexpr");
    static_assert(Mat<float, 3, 4>::rows() == 3, "rows() should be constexpr");
    static_assert(Mat<float, 3, 4>::cols() == 4, "cols() should be constexpr");

    // Test SquareMat
    SquareMat<float, 3> sm3;
    EXPECT_EQ(sm3.size(), 9u);
    EXPECT_EQ(sm3.rows(), 3u);
    EXPECT_EQ(sm3.cols(), 3u);
}
