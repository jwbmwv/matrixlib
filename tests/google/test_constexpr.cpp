// SPDX-License-Identifier: MIT
/// @file test_constexpr.cpp
/// @brief Compile-time (constexpr) tests for MatrixLib
/// @details These tests verify that factory methods and operations can be evaluated at compile-time.
///          If these tests compile, the constexpr functionality is working correctly.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <gtest/gtest.h>

using namespace matrixlib;

// ==================== Compile-Time Tests (C++14+) ====================
// These static_assert tests only compile with C++14 or later due to relaxed constexpr

#if __cplusplus >= 201402L

// Vector constexpr tests
namespace
{
constexpr Vec<float, 3> vec_zero = Vec<float, 3>::zero();
static_assert(vec_zero[0] == 0.0f, "Vec::zero()[0] should be 0");
static_assert(vec_zero[1] == 0.0f, "Vec::zero()[1] should be 0");
static_assert(vec_zero[2] == 0.0f, "Vec::zero()[2] should be 0");

constexpr Vec<int, 4> vec_one = Vec<int, 4>::one();
static_assert(vec_one[0] == 1, "Vec::one()[0] should be 1");
static_assert(vec_one[1] == 1, "Vec::one()[1] should be 1");
static_assert(vec_one[2] == 1, "Vec::one()[2] should be 1");
static_assert(vec_one[3] == 1, "Vec::one()[3] should be 1");
}  // namespace

// Matrix constexpr tests
namespace
{
constexpr Mat<float, 2, 3> mat_zero = Mat<float, 2, 3>::zero();
static_assert(mat_zero(0, 0) == 0.0f, "Mat::zero() element should be 0");
static_assert(mat_zero(1, 2) == 0.0f, "Mat::zero() element should be 0");

constexpr SquareMat<int, 2> mat2_identity = SquareMat<int, 2>::identity();
static_assert(mat2_identity(0, 0) == 1, "Identity matrix diagonal should be 1");
static_assert(mat2_identity(1, 1) == 1, "Identity matrix diagonal should be 1");
static_assert(mat2_identity(0, 1) == 0, "Identity matrix off-diagonal should be 0");
static_assert(mat2_identity(1, 0) == 0, "Identity matrix off-diagonal should be 0");

constexpr SquareMat<int, 3> mat3_identity = SquareMat<int, 3>::identity();
static_assert(mat3_identity(0, 0) == 1, "3x3 Identity diagonal should be 1");
static_assert(mat3_identity(1, 1) == 1, "3x3 Identity diagonal should be 1");
static_assert(mat3_identity(2, 2) == 1, "3x3 Identity diagonal should be 1");
static_assert(mat3_identity(0, 1) == 0, "3x3 Identity off-diagonal should be 0");
static_assert(mat3_identity(1, 2) == 0, "3x3 Identity off-diagonal should be 0");

constexpr SquareMat<int, 4> mat4_identity = SquareMat<int, 4>::identity();
static_assert(mat4_identity(0, 0) == 1, "4x4 Identity diagonal should be 1");
static_assert(mat4_identity(1, 1) == 1, "4x4 Identity diagonal should be 1");
static_assert(mat4_identity(2, 2) == 1, "4x4 Identity diagonal should be 1");
static_assert(mat4_identity(3, 3) == 1, "4x4 Identity diagonal should be 1");
static_assert(mat4_identity(0, 3) == 0, "4x4 Identity off-diagonal should be 0");
}  // namespace

// Quaternion constexpr tests
namespace
{
constexpr Quaternion<float> quat_identity = Quaternion<float>::identity();
static_assert(quat_identity.w == 1.0f, "Identity quaternion w should be 1");
static_assert(quat_identity.x == 0.0f, "Identity quaternion x should be 0");
static_assert(quat_identity.y == 0.0f, "Identity quaternion y should be 0");
static_assert(quat_identity.z == 0.0f, "Identity quaternion z should be 0");
}  // namespace

// Special angle rotations (constexpr in C++11+)
namespace
{
constexpr auto rot90 = SquareMat<float, 2>::rotation_deg<90>();
static_assert(rot90(0, 0) == 0.0f, "90° rotation matrix element");
static_assert(rot90(0, 1) == -1.0f, "90° rotation matrix element");
static_assert(rot90(1, 0) == 1.0f, "90° rotation matrix element");
static_assert(rot90(1, 1) == 0.0f, "90° rotation matrix element");

constexpr auto rot180 = SquareMat<int, 2>::rotation_deg<180>();
static_assert(rot180(0, 0) == -1, "180° rotation matrix element");
static_assert(rot180(1, 1) == -1, "180° rotation matrix element");

constexpr auto rot270 = SquareMat<double, 2>::rotation_deg<270>();
static_assert(rot270(0, 0) == 0.0, "270° rotation matrix element");
static_assert(rot270(0, 1) == 1.0, "270° rotation matrix element");
static_assert(rot270(1, 0) == -1.0, "270° rotation matrix element");
}  // namespace

#endif  // C++14+

// ==================== Runtime Tests ====================
// These verify constexpr functions work correctly at runtime too

TEST(ConstexprTest, VectorZero)
{
    constexpr Vec<float, 3> v = Vec<float, 3>::zero();
    EXPECT_EQ(v[0], 0.0f);
    EXPECT_EQ(v[1], 0.0f);
    EXPECT_EQ(v[2], 0.0f);
}

TEST(ConstexprTest, VectorOne)
{
    constexpr Vec<double, 2> v = Vec<double, 2>::one();
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.0);
}

TEST(ConstexprTest, MatrixZero)
{
    constexpr Mat<int, 3, 2> m = Mat<int, 3, 2>::zero();
    for (uint32_t r = 0; r < 3; ++r)
    {
        for (uint32_t c = 0; c < 2; ++c)
        {
            EXPECT_EQ(m(r, c), 0);
        }
    }
}

TEST(ConstexprTest, MatrixIdentity2x2)
{
    constexpr SquareMat<float, 2> m = SquareMat<float, 2>::identity();
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(0, 1), 0.0f);
    EXPECT_EQ(m(1, 0), 0.0f);
    EXPECT_EQ(m(1, 1), 1.0f);
}

TEST(ConstexprTest, MatrixIdentity3x3)
{
    constexpr SquareMat<double, 3> m = SquareMat<double, 3>::identity();
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(1, 1), 1.0);
    EXPECT_EQ(m(2, 2), 1.0);
    EXPECT_EQ(m(0, 1), 0.0);
    EXPECT_EQ(m(1, 2), 0.0);
}

TEST(ConstexprTest, MatrixIdentity4x4)
{
    constexpr SquareMat<float, 4> m = SquareMat<float, 4>::identity();
    EXPECT_EQ(m(0, 0), 1.0f);
    EXPECT_EQ(m(1, 1), 1.0f);
    EXPECT_EQ(m(2, 2), 1.0f);
    EXPECT_EQ(m(3, 3), 1.0f);
    EXPECT_EQ(m(0, 2), 0.0f);
    EXPECT_EQ(m(2, 3), 0.0f);
}

TEST(ConstexprTest, QuaternionIdentity)
{
    constexpr Quaternion<float> q = Quaternion<float>::identity();
    EXPECT_EQ(q.w, 1.0f);
    EXPECT_EQ(q.x, 0.0f);
    EXPECT_EQ(q.y, 0.0f);
    EXPECT_EQ(q.z, 0.0f);
}

TEST(ConstexprTest, Rotation2DSpecialAngles)
{
    constexpr auto rot0 = SquareMat<float, 2>::rotation_deg<0>();
    EXPECT_NEAR(rot0(0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(rot0(1, 1), 1.0f, 1e-6f);

    constexpr auto rot90 = SquareMat<float, 2>::rotation_deg<90>();
    EXPECT_NEAR(rot90(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(rot90(0, 1), -1.0f, 1e-6f);
    EXPECT_NEAR(rot90(1, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(rot90(1, 1), 0.0f, 1e-6f);

    constexpr auto rot180 = SquareMat<float, 2>::rotation_deg<180>();
    EXPECT_NEAR(rot180(0, 0), -1.0f, 1e-6f);
    EXPECT_NEAR(rot180(1, 1), -1.0f, 1e-6f);

    constexpr auto rot270 = SquareMat<float, 2>::rotation_deg<270>();
    EXPECT_NEAR(rot270(0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(rot270(0, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(rot270(1, 0), -1.0f, 1e-6f);
}

// Verify constexpr works in array initialization (important for embedded)
TEST(ConstexprTest, ConstexprArrayInitialization)
{
    static constexpr Vec<float, 3> lookup_table[] = {Vec<float, 3>::zero(), Vec<float, 3>::one(),
                                                     Vec<float, 3>::zero()};

    EXPECT_EQ(lookup_table[0][0], 0.0f);
    EXPECT_EQ(lookup_table[1][0], 1.0f);
    EXPECT_EQ(lookup_table[2][1], 0.0f);
}

// Verify constexpr identity matrices can be used at compile time
TEST(ConstexprTest, ConstexprMatrixMultiplication)
{
    constexpr SquareMat<int, 2> identity = SquareMat<int, 2>::identity();

    // This multiplication happens at runtime, but identity was created at compile-time
    SquareMat<int, 2> test;
    test(0, 0) = 2;
    test(0, 1) = 3;
    test(1, 0) = 4;
    test(1, 1) = 5;

    auto result = test * identity;

    EXPECT_EQ(result(0, 0), 2);
    EXPECT_EQ(result(0, 1), 3);
    EXPECT_EQ(result(1, 0), 4);
    EXPECT_EQ(result(1, 1), 5);
}
