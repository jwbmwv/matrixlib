// SPDX-License-Identifier: MIT
/// @file test_mat.cpp
/// @brief MatrixLib Zephyr Test Suite - Mat<T,R,C> Tests
/// @details Comprehensive unit tests for matrix operations using Zephyr ztest framework.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#include <zephyr/ztest.h>
#include <matrixlib/matrixlib.hpp>
#include <cmath>

using namespace matrixlib;

// Use epsilon consistent with library's comparison tolerance (epsilon * 100)
// std::numeric_limits<float>::epsilon() is ~1.19e-7, so * 100 gives ~1.19e-5
#define FLOAT_EPSILON (std::numeric_limits<float>::epsilon() * 100.0f)

static bool float_eq(float a, float b)
{
    return std::fabs(a - b) < FLOAT_EPSILON;
}

ZTEST(matrixlib_mat, test_mat_construction)
{
    Mat<float, 2, 2> m1;
    SquareMat<float, 3> m2 = SquareMat<float, 3>::identity();

    zassert_true(float_eq(m2[0][0], 1.0f), "Identity [0][0] should be 1");
    zassert_true(float_eq(m2[0][1], 0.0f), "Identity [0][1] should be 0");
    zassert_true(float_eq(m2[1][1], 1.0f), "Identity [1][1] should be 1");
}

ZTEST(matrixlib_mat, test_mat_addition)
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

    zassert_true(float_eq(sum[0][0], 6.0f), "sum[0][0] should be 6");
    zassert_true(float_eq(sum[1][1], 12.0f), "sum[1][1] should be 12");
}

ZTEST(matrixlib_mat, test_mat_multiplication)
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

    // Expected: [58, 64]
    //           [139, 154]
    zassert_true(float_eq(c[0][0], 58.0f), "c[0][0] should be 58");
    zassert_true(float_eq(c[0][1], 64.0f), "c[0][1] should be 64");
    zassert_true(float_eq(c[1][0], 139.0f), "c[1][0] should be 139");
    zassert_true(float_eq(c[1][1], 154.0f), "c[1][1] should be 154");
}

ZTEST(matrixlib_mat, test_mat_vector_mult)
{
    SquareMat<float, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    Vec<float, 2> v(5.0f, 6.0f);
    Vec<float, 2> result = m * v;

    // Expected: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    zassert_true(float_eq(result[0], 17.0f), "result[0] should be 17");
    zassert_true(float_eq(result[1], 39.0f), "result[1] should be 39");
}

ZTEST(matrixlib_mat, test_mat_transpose)
{
    Mat<float, 2, 3> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[0][2] = 3.0f;
    m[1][0] = 4.0f;
    m[1][1] = 5.0f;
    m[1][2] = 6.0f;

    Mat<float, 3, 2> mt = m.transpose();

    zassert_true(float_eq(mt[0][0], 1.0f), "mt[0][0] should be 1");
    zassert_true(float_eq(mt[0][1], 4.0f), "mt[0][1] should be 4");
    zassert_true(float_eq(mt[1][0], 2.0f), "mt[1][0] should be 2");
    zassert_true(float_eq(mt[2][1], 6.0f), "mt[2][1] should be 6");
}

ZTEST(matrixlib_mat, test_mat_determinant_2x2)
{
    SquareMat<float, 2> m;
    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[1][0] = 3.0f;
    m[1][1] = 4.0f;

    float det = m.determinant();

    // det = 1*4 - 2*3 = -2
    zassert_true(float_eq(det, -2.0f), "Determinant should be -2");
}

ZTEST(matrixlib_mat, test_mat_determinant_3x3)
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

    // det = 1*(1*0-4*6) - 2*(0*0-4*5) + 3*(0*6-1*5) = 1*(-24) - 2*(-20) + 3*(-5) = -24 + 40 - 15 = 1
    zassert_true(float_eq(det, 1.0f), "Determinant should be 1");
}

ZTEST(matrixlib_mat, test_mat_trace)
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

    // trace = 1 + 5 + 9 = 15
    zassert_true(float_eq(trace, 15.0f), "Trace should be 15");
}

ZTEST(matrixlib_mat, test_mat_inverse_2x2)
{
    SquareMat<float, 2> m;
    m[0][0] = 4.0f;
    m[0][1] = 7.0f;
    m[1][0] = 2.0f;
    m[1][1] = 6.0f;

    SquareMat<float, 2> inv = m.inverse();
    SquareMat<float, 2> identity = m * inv;

    zassert_true(float_eq(identity[0][0], 1.0f), "M * M^-1 [0][0] should be 1");
    zassert_true(float_eq(identity[0][1], 0.0f), "M * M^-1 [0][1] should be 0");
    zassert_true(float_eq(identity[1][0], 0.0f), "M * M^-1 [1][0] should be 0");
    zassert_true(float_eq(identity[1][1], 1.0f), "M * M^-1 [1][1] should be 1");
}

ZTEST(matrixlib_mat, test_rotation_2d)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 2> R = SquareMat<float, 2>::rotation(angle);

    Vec<float, 2> v(1.0f, 0.0f);
    Vec<float, 2> rotated = R * v;

    // Should rotate to (0, 1)
    zassert_true(float_eq(rotated[0], 0.0f), "rotated.x should be ~0");
    zassert_true(float_eq(rotated[1], 1.0f), "rotated.y should be ~1");
}

ZTEST(matrixlib_mat, test_rotation_3d_x)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Rx = SquareMat<float, 3>::rotation_x(angle);

    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> rotated = Rx * y;

    // Should rotate Y to Z
    zassert_true(float_eq(rotated[0], 0.0f), "rotated.x should be ~0");
    zassert_true(float_eq(rotated[1], 0.0f), "rotated.y should be ~0");
    zassert_true(float_eq(rotated[2], 1.0f), "rotated.z should be ~1");
}

ZTEST(matrixlib_mat, test_rotation_3d_y)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Ry = SquareMat<float, 3>::rotation_y(angle);

    Vec<float, 3> z(0.0f, 0.0f, 1.0f);
    Vec<float, 3> rotated = Ry * z;

    // Should rotate Z to X
    zassert_true(float_eq(rotated[0], 1.0f), "rotated.x should be ~1");
    zassert_true(float_eq(rotated[1], 0.0f), "rotated.y should be ~0");
    zassert_true(float_eq(rotated[2], 0.0f), "rotated.z should be ~0");
}

ZTEST(matrixlib_mat, test_rotation_3d_z)
{
    float angle = 3.14159f / 2.0f;  // 90 degrees
    SquareMat<float, 3> Rz = SquareMat<float, 3>::rotation_z(angle);

    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = Rz * x;

    // Should rotate X to Y
    zassert_true(float_eq(rotated[0], 0.0f), "rotated.x should be ~0");
    zassert_true(float_eq(rotated[1], 1.0f), "rotated.y should be ~1");
    zassert_true(float_eq(rotated[2], 0.0f), "rotated.z should be ~0");
}

ZTEST(matrixlib_mat, test_rotation_axis_angle)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    float angle = 3.14159f / 2.0f;

    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_axis_angle(axis, angle);
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = R * x;

    zassert_true(float_eq(rotated[1], 1.0f), "Should rotate X to Y around Z axis");
}

ZTEST(matrixlib_mat, test_rotation_from_to)
{
    Vec<float, 3> from(1.0f, 0.0f, 0.0f);
    Vec<float, 3> to(0.0f, 1.0f, 0.0f);

    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_from_to(from, to);
    Vec<float, 3> result = R * from;

    zassert_true(float_eq(result[0], to[0]), "result.x should match to.x");
    zassert_true(float_eq(result[1], to[1]), "result.y should match to.y");
    zassert_true(float_eq(result[2], to[2]), "result.z should match to.z");
}

ZTEST(matrixlib_mat, test_look_at)
{
    Vec<float, 3> target(1.0f, 0.0f, 0.0f);
    Vec<float, 3> up(0.0f, 0.0f, 1.0f);

    SquareMat<float, 3> look = SquareMat<float, 3>::look_at(target, up);

    // First column should be target direction
    Vec<float, 3> forward(look[0][0], look[1][0], look[2][0]);
    float mag = forward.magnitude();

    zassert_true(float_eq(mag, 1.0f), "Forward vector should be normalized");
    zassert_true(float_eq(forward[0], 1.0f), "Should point in target direction");
}

ZTEST(matrixlib_mat, test_euler_angles)
{
    float roll = 0.1f;
    float pitch = 0.2f;
    float yaw = 0.3f;

    SquareMat<float, 3> Rx = SquareMat<float, 3>::rotation_x(roll);
    SquareMat<float, 3> Ry = SquareMat<float, 3>::rotation_y(pitch);
    SquareMat<float, 3> Rz = SquareMat<float, 3>::rotation_z(yaw);

    SquareMat<float, 3> R = Rz * Ry * Rx;
    Vec<float, 3> angles = R.euler_angles();

    // Should extract approximately the same angles
    zassert_true(float_eq(angles[0], roll), "Roll should match");
    zassert_true(float_eq(angles[1], pitch), "Pitch should match");
    zassert_true(float_eq(angles[2], yaw), "Yaw should match");
}
