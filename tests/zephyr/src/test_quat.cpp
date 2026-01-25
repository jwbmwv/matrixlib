// SPDX-License-Identifier: MIT
/// @file test_quat.cpp
/// @brief MatrixLib Zephyr Test Suite - Quaternion<T> Tests
/// @details Comprehensive unit tests for quaternion operations using Zephyr ztest framework.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#include <zephyr/ztest.h>
#include <matrixlib/quaternion.hpp>
#include <cmath>

using namespace matrixlib;

#define FLOAT_EPSILON 0.0001f

static bool float_eq(float a, float b)
{
    return std::fabs(a - b) < FLOAT_EPSILON;
}

ZTEST(matrixlib_quat, test_quat_construction)
{
    Quaternion<float> q1;                          // Identity
    Quaternion<float> q2(1.0f, 0.0f, 0.0f, 0.0f);  // w, x, y, z

    zassert_true(float_eq(q1.w(), 1.0f), "Identity w should be 1");
    zassert_true(float_eq(q1.x(), 0.0f), "Identity x should be 0");

    zassert_true(float_eq(q2.w(), 1.0f), "q2.w should be 1");
    zassert_true(float_eq(q2.x(), 0.0f), "q2.x should be 0");
}

ZTEST(matrixlib_quat, test_quat_axis_angle)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    float angle = 3.14159f / 2.0f;  // 90 degrees

    Quaternion<float> q(axis, angle);

    // For 90° around Z: w = cos(45°) ≈ 0.707, z = sin(45°) ≈ 0.707
    zassert_true(float_eq(q.w(), 0.7071f), "w should be ~0.707");
    zassert_true(float_eq(q.z(), 0.7071f), "z should be ~0.707");
    zassert_true(float_eq(q.x(), 0.0f), "x should be 0");
    zassert_true(float_eq(q.y(), 0.0f), "y should be 0");
}

ZTEST(matrixlib_quat, test_quat_conjugate)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> qc = q.conjugate();

    zassert_true(float_eq(qc.w(), 1.0f), "conjugate w unchanged");
    zassert_true(float_eq(qc.x(), -2.0f), "conjugate x negated");
    zassert_true(float_eq(qc.y(), -3.0f), "conjugate y negated");
    zassert_true(float_eq(qc.z(), -4.0f), "conjugate z negated");
}

ZTEST(matrixlib_quat, test_quat_norm)
{
    Quaternion<float> q(1.0f, 2.0f, 2.0f, 0.0f);
    float n = q.norm();

    // norm = sqrt(1 + 4 + 4 + 0) = sqrt(9) = 3
    zassert_true(float_eq(n, 3.0f), "Norm should be 3");
}

ZTEST(matrixlib_quat, test_quat_normalize)
{
    Quaternion<float> q(2.0f, 2.0f, 2.0f, 2.0f);
    Quaternion<float> qn = q.normalized();

    float norm = qn.norm();
    zassert_true(float_eq(norm, 1.0f), "Normalized quaternion should have norm 1");

    // Each component should be 0.5
    zassert_true(float_eq(qn.w(), 0.5f), "qn.w should be 0.5");
    zassert_true(float_eq(qn.x(), 0.5f), "qn.x should be 0.5");
}

ZTEST(matrixlib_quat, test_quat_multiplication)
{
    Vec<float, 3> z_axis(0.0f, 0.0f, 1.0f);
    Vec<float, 3> x_axis(1.0f, 0.0f, 0.0f);

    Quaternion<float> q1(z_axis, 3.14159f / 2.0f);  // 90° around Z
    Quaternion<float> q2(x_axis, 3.14159f / 2.0f);  // 90° around X

    Quaternion<float> q3 = q1 * q2;

    // Combined rotation
    float norm = q3.norm();
    zassert_true(float_eq(norm, 1.0f), "Product should be unit quaternion");
}

ZTEST(matrixlib_quat, test_quat_inverse)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 4.0f);

    Quaternion<float> qi = q.inverse();
    Quaternion<float> identity = q * qi;

    zassert_true(float_eq(identity.w(), 1.0f), "q * q^-1 should be identity");
    zassert_true(float_eq(identity.x(), 0.0f), "q * q^-1 x should be 0");
    zassert_true(float_eq(identity.y(), 0.0f), "q * q^-1 y should be 0");
    zassert_true(float_eq(identity.z(), 0.0f), "q * q^-1 z should be 0");
}

ZTEST(matrixlib_quat, test_quat_rotate_vector)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 2.0f);  // 90° around Z

    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q.rotate(v);

    // Should rotate X to Y
    zassert_true(float_eq(rotated[0], 0.0f), "rotated.x should be ~0");
    zassert_true(float_eq(rotated[1], 1.0f), "rotated.y should be ~1");
    zassert_true(float_eq(rotated[2], 0.0f), "rotated.z should be ~0");
}

ZTEST(matrixlib_quat, test_quat_to_rotation_matrix)
{
    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q(axis, 3.14159f / 2.0f);

    SquareMat<float, 3> R = q.to_rotation_matrix();
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = R * v;

    // Should rotate X to Y
    zassert_true(float_eq(rotated[0], 0.0f), "matrix rotated.x should be ~0");
    zassert_true(float_eq(rotated[1], 1.0f), "matrix rotated.y should be ~1");
}

ZTEST(matrixlib_quat, test_quat_from_rotation_matrix)
{
    SquareMat<float, 3> R = SquareMat<float, 3>::rotation_z(3.14159f / 2.0f);
    Quaternion<float> q = Quaternion<float>::from_rotation_matrix(R);

    // Should be 90° rotation around Z
    zassert_true(float_eq(q.w(), 0.7071f), "w should be ~0.707");
    zassert_true(float_eq(q.z(), 0.7071f), "z should be ~0.707");
}

ZTEST(matrixlib_quat, test_quat_slerp)
{
    Quaternion<float> q_start = Quaternion<float>::identity();

    Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
    Quaternion<float> q_end(axis, 3.14159f / 2.0f);

    // Interpolate at t=0.5 (halfway)
    Quaternion<float> q_mid = q_start.slerp(q_end, 0.5f);

    // Should be 45° rotation
    Vec<float, 3> v(1.0f, 0.0f, 0.0f);
    Vec<float, 3> rotated = q_mid.rotate(v);

    // At 45°, x and y should be equal
    zassert_true(float_eq(rotated[0], rotated[1]), "At 45°, x and y should be equal");
    zassert_true(float_eq(rotated[2], 0.0f), "z should remain 0");
}

ZTEST(matrixlib_quat, test_quat_addition)
{
    Quaternion<float> q1(1.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> q2(0.0f, 1.0f, 0.0f, 0.0f);

    Quaternion<float> sum = q1 + q2;

    zassert_true(float_eq(sum.w(), 1.0f), "sum.w should be 1");
    zassert_true(float_eq(sum.x(), 1.0f), "sum.x should be 1");
    zassert_true(float_eq(sum.y(), 0.0f), "sum.y should be 0");
}

ZTEST(matrixlib_quat, test_quat_scalar_mult)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);
    Quaternion<float> scaled = q * 2.0f;

    zassert_true(float_eq(scaled.w(), 2.0f), "scaled.w should be 2");
    zassert_true(float_eq(scaled.x(), 4.0f), "scaled.x should be 4");
    zassert_true(float_eq(scaled.y(), 6.0f), "scaled.y should be 6");
    zassert_true(float_eq(scaled.z(), 8.0f), "scaled.z should be 8");
}

ZTEST(matrixlib_quat, test_quat_identity)
{
    Quaternion<float> q = Quaternion<float>::identity();

    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    Vec<float, 3> rotated = q.rotate(v);

    // Identity rotation should not change vector
    zassert_true(float_eq(rotated[0], v[0]), "identity preserves x");
    zassert_true(float_eq(rotated[1], v[1]), "identity preserves y");
    zassert_true(float_eq(rotated[2], v[2]), "identity preserves z");
}

ZTEST(matrixlib_quat, test_quat_vec_accessor)
{
    Quaternion<float> q(1.0f, 2.0f, 3.0f, 4.0f);

    Vec<float, 3>& v = q.vec();

    zassert_true(float_eq(v[0], 2.0f), "vec[0] should be x = 2");
    zassert_true(float_eq(v[1], 3.0f), "vec[1] should be y = 3");
    zassert_true(float_eq(v[2], 4.0f), "vec[2] should be z = 4");

    // Modify through reference
    v[0] = 5.0f;
    zassert_true(float_eq(q.x(), 5.0f), "Modifying vec should affect quaternion");
}

ZTEST(matrixlib_quat, test_quat_compound_assignment)
{
    Quaternion<float> q(1.0f, 0.0f, 0.0f, 0.0f);
    Quaternion<float> a(1.0f, 1.0f, 1.0f, 1.0f);

    q += a;
    zassert_true(float_eq(q.w(), 2.0f), "q.w should be 2 after +=");

    q -= a;
    zassert_true(float_eq(q.w(), 1.0f), "q.w should be 1 after -=");

    q *= 2.0f;
    zassert_true(float_eq(q.w(), 2.0f), "q.w should be 2 after *= 2");
}
