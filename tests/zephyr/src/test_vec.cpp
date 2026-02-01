// SPDX-License-Identifier: MIT
/// @file test_vec.cpp
/// @brief MatrixLib Zephyr Test Suite - Vec<T,N> Tests
/// @details Comprehensive unit tests for vector operations using Zephyr ztest framework.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#include <zephyr/ztest.h>
#include <matrixlib/matrixlib.hpp>
#include <cmath>

using namespace matrixlib;

#define FLOAT_EPSILON 0.0001f

// Helper to compare floats
static bool float_eq(float a, float b)
{
    return std::fabs(a - b) < FLOAT_EPSILON;
}

ZTEST(matrixlib_vec, test_vec_construction)
{
    Vec<float, 3> v1;
    Vec<float, 3> v2(1.0f, 2.0f, 3.0f);
    Vec<float, 3> v3(v2);

    zassert_true(float_eq(v2[Vec<float, 3>::X], 1.0f), "v2.x should be 1.0");
    zassert_true(float_eq(v2[Vec<float, 3>::Y], 2.0f), "v2.y should be 2.0");
    zassert_true(float_eq(v2[Vec<float, 3>::Z], 3.0f), "v2.z should be 3.0");

    zassert_true(float_eq(v3[0], v2[0]), "Copy constructor failed");
}

ZTEST(matrixlib_vec, test_vec_addition)
{
    Vec<float, 3> a(1.0f, 2.0f, 3.0f);
    Vec<float, 3> b(4.0f, 5.0f, 6.0f);
    Vec<float, 3> sum = a + b;

    zassert_true(float_eq(sum[0], 5.0f), "sum[0] should be 5.0");
    zassert_true(float_eq(sum[1], 7.0f), "sum[1] should be 7.0");
    zassert_true(float_eq(sum[2], 9.0f), "sum[2] should be 9.0");
}

ZTEST(matrixlib_vec, test_vec_subtraction)
{
    Vec<float, 3> a(4.0f, 5.0f, 6.0f);
    Vec<float, 3> b(1.0f, 2.0f, 3.0f);
    Vec<float, 3> diff = a - b;

    zassert_true(float_eq(diff[0], 3.0f), "diff[0] should be 3.0");
    zassert_true(float_eq(diff[1], 3.0f), "diff[1] should be 3.0");
    zassert_true(float_eq(diff[2], 3.0f), "diff[2] should be 3.0");
}

ZTEST(matrixlib_vec, test_vec_scalar_multiplication)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> scaled = v * 2.0f;

    zassert_true(float_eq(scaled[0], 2.0f), "scaled[0] should be 2.0");
    zassert_true(float_eq(scaled[1], 4.0f), "scaled[1] should be 4.0");
    zassert_true(float_eq(scaled[2], 6.0f), "scaled[2] should be 6.0");
}

ZTEST(matrixlib_vec, test_vec_dot_product)
{
    Vec<float, 3> a(1.0f, 0.0f, 0.0f);
    Vec<float, 3> b(0.0f, 1.0f, 0.0f);
    Vec<float, 3> c(1.0f, 2.0f, 3.0f);
    Vec<float, 3> d(4.0f, 5.0f, 6.0f);

    float dot1 = a.dot(b);
    float dot2 = c.dot(d);

    zassert_true(float_eq(dot1, 0.0f), "Orthogonal vectors should have dot product 0");
    zassert_true(float_eq(dot2, 32.0f), "dot(c,d) should be 32.0");
}

ZTEST(matrixlib_vec, test_vec_cross_product)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> z = x.cross(y);

    zassert_true(float_eq(z[Vec<float, 3>::X], 0.0f), "z.x should be 0");
    zassert_true(float_eq(z[Vec<float, 3>::Y], 0.0f), "z.y should be 0");
    zassert_true(float_eq(z[Vec<float, 3>::Z], 1.0f), "z.z should be 1");
}

ZTEST(matrixlib_vec, test_vec_magnitude)
{
    Vec<float, 3> v(3.0f, 4.0f, 0.0f);
    float mag = v.magnitude();

    zassert_true(float_eq(mag, 5.0f), "Magnitude of (3,4,0) should be 5.0");
}

ZTEST(matrixlib_vec, test_vec_normalization)
{
    Vec<float, 3> v(3.0f, 4.0f, 0.0f);
    Vec<float, 3> n = v.normalized();
    float mag = n.magnitude();

    zassert_true(float_eq(mag, 1.0f), "Normalized vector should have magnitude 1.0");
    zassert_true(float_eq(n[0], 0.6f), "n.x should be 0.6");
    zassert_true(float_eq(n[1], 0.8f), "n.y should be 0.8");
}

ZTEST(matrixlib_vec, test_vec_angle)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> diag(1.0f, 1.0f, 0.0f);

    float angle1 = x.angle(y);
    float angle2 = x.angle(diag);

    zassert_true(float_eq(angle1, 3.14159f / 2.0f), "Angle between x and y should be 90°");
    zassert_true(float_eq(angle2, 3.14159f / 4.0f), "Angle should be 45°");
}

ZTEST(matrixlib_vec, test_vec_project_reject)
{
    Vec<float, 3> a(3.0f, 4.0f, 0.0f);
    Vec<float, 3> b(1.0f, 0.0f, 0.0f);

    Vec<float, 3> proj = a.project(b);
    Vec<float, 3> rej = a.reject(b);

    // proj should be along b direction
    zassert_true(float_eq(proj[0], 3.0f), "proj.x should be 3.0");
    zassert_true(float_eq(proj[1], 0.0f), "proj.y should be 0.0");

    // rej should be perpendicular to b
    zassert_true(float_eq(rej[0], 0.0f), "rej.x should be 0.0");
    zassert_true(float_eq(rej[1], 4.0f), "rej.y should be 4.0");

    // proj + rej should equal original
    Vec<float, 3> sum = proj + rej;
    zassert_true(float_eq(sum[0], a[0]), "sum.x should equal a.x");
    zassert_true(float_eq(sum[1], a[1]), "sum.y should equal a.y");
}

ZTEST(matrixlib_vec, test_vec_signed_angle)
{
    Vec<float, 3> x(1.0f, 0.0f, 0.0f);
    Vec<float, 3> y(0.0f, 1.0f, 0.0f);
    Vec<float, 3> z(0.0f, 0.0f, 1.0f);

    float angle_xy_z = x.signed_angle(y, z);
    float angle_yx_z = y.signed_angle(x, z);

    zassert_true(angle_xy_z > 0.0f, "Angle from x to y around z should be positive");
    zassert_true(angle_yx_z < 0.0f, "Angle from y to x around z should be negative");
}

ZTEST(matrixlib_vec, test_vec_accessors)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);

    zassert_true(float_eq(v.x(), 1.0f), "v.x() should be 1.0");
    zassert_true(float_eq(v.y(), 2.0f), "v.y() should be 2.0");
    zassert_true(float_eq(v.z(), 3.0f), "v.z() should be 3.0");

    v.x() = 4.0f;
    zassert_true(float_eq(v.x(), 4.0f), "v.x() = 4.0 should work");
}

ZTEST(matrixlib_vec, test_vec_compound_assignment)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> a(1.0f, 1.0f, 1.0f);

    v += a;
    zassert_true(float_eq(v[0], 2.0f), "v[0] should be 2.0 after +=");

    v -= a;
    zassert_true(float_eq(v[0], 1.0f), "v[0] should be 1.0 after -=");

    v *= 2.0f;
    zassert_true(float_eq(v[0], 2.0f), "v[0] should be 2.0 after *= 2");

    v /= 2.0f;
    zassert_true(float_eq(v[0], 1.0f), "v[0] should be 1.0 after /= 2");
}
