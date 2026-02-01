// Property-Based Testing Framework for MatrixLib
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <random>
#include <limits>

using namespace matrixlib;

// Random number generator utilities
class RandomGenerator
{
public:
    RandomGenerator(unsigned seed = 42) : gen(seed), dist(-100.0f, 100.0f) {}

    float nextFloat() { return dist(gen); }

    float nextFloatSmall()
    {
        std::uniform_real_distribution<float> small_dist(-10.0f, 10.0f);
        return small_dist(gen);
    }

    int nextInt(int min, int max)
    {
        std::uniform_int_distribution<int> int_dist(min, max);
        return int_dist(gen);
    }

private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

// Property: Vector addition is commutative
TEST(PropertyBased, VectorAdditionCommutative)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f b(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        Vec3f ab = a + b;
        Vec3f ba = b + a;

        EXPECT_FLOAT_EQ(ab.x(), ba.x());
        EXPECT_FLOAT_EQ(ab.y(), ba.y());
        EXPECT_FLOAT_EQ(ab.z(), ba.z());
    }
}

// Property: Vector addition is associative
TEST(PropertyBased, VectorAdditionAssociative)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f b(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f c(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        Vec3f abc1 = (a + b) + c;
        Vec3f abc2 = a + (b + c);

        EXPECT_NEAR(abc1.x(), abc2.x(), 1e-4f);
        EXPECT_NEAR(abc1.y(), abc2.y(), 1e-4f);
        EXPECT_NEAR(abc1.z(), abc2.z(), 1e-4f);
    }
}

// Property: Vector has additive identity (zero vector)
TEST(PropertyBased, VectorAdditiveIdentity)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f v(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f zero(0, 0, 0);

        Vec3f result = v + zero;

        EXPECT_FLOAT_EQ(result.x(), v.x());
        EXPECT_FLOAT_EQ(result.y(), v.y());
        EXPECT_FLOAT_EQ(result.z(), v.z());
    }
}

// Property: Dot product is commutative
TEST(PropertyBased, DotProductCommutative)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f b(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        float ab = a.dot(b);
        float ba = b.dot(a);

        EXPECT_FLOAT_EQ(ab, ba);
    }
}

// Property: Dot product distributive over addition
TEST(PropertyBased, DotProductDistributive)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloatSmall(), rng.nextFloatSmall(), rng.nextFloatSmall());
        Vec3f b(rng.nextFloatSmall(), rng.nextFloatSmall(), rng.nextFloatSmall());
        Vec3f c(rng.nextFloatSmall(), rng.nextFloatSmall(), rng.nextFloatSmall());

        float lhs = a.dot(b + c);
        float rhs = a.dot(b) + a.dot(c);

        EXPECT_NEAR(lhs, rhs, 1e-3f);
    }
}

// Property: Cross product anti-commutative
TEST(PropertyBased, CrossProductAntiCommutative)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f b(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        Vec3f ab = a.cross(b);
        Vec3f ba = b.cross(a);

        EXPECT_NEAR(ab.x(), -ba.x(), 1e-3f);
        EXPECT_NEAR(ab.y(), -ba.y(), 1e-3f);
        EXPECT_NEAR(ab.z(), -ba.z(), 1e-3f);
    }
}

// Property: Cross product perpendicular to both vectors
TEST(PropertyBased, CrossProductPerpendicular)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f a(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Vec3f b(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        Vec3f c = a.cross(b);

        // Cross product perpendicular to both inputs
        float dot_ac = a.dot(c);
        float dot_bc = b.dot(c);

        EXPECT_NEAR(dot_ac, 0.0f, 1e-2f);
        EXPECT_NEAR(dot_bc, 0.0f, 1e-2f);
    }
}

// Property: Normalization produces unit vector
TEST(PropertyBased, NormalizationProducesUnitVector)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f v(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        if (v.normSquared() < 1e-6f)
            continue;  // Skip near-zero vectors

        Vec3f n = v.normalized();
        float len = n.norm();

        EXPECT_NEAR(len, 1.0f, 1e-5f);
    }
}

// Property: Matrix multiplication associative
TEST(PropertyBased, MatrixMultiplicationAssociative)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 100; ++trial)
    {
        Mat<float, 3, 3> A, B, C;

        for (int i = 0; i < 9; ++i)
        {
            A.data[i] = rng.nextFloatSmall();
            B.data[i] = rng.nextFloatSmall();
            C.data[i] = rng.nextFloatSmall();
        }

        auto ABC1 = (A * B) * C;
        auto ABC2 = A * (B * C);

        for (int i = 0; i < 9; ++i)
        {
            EXPECT_NEAR(ABC1.data[i], ABC2.data[i], 1e-2f);
        }
    }
}

// Property: Matrix has multiplicative identity
TEST(PropertyBased, MatrixMultiplicativeIdentity)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        SquareMat<float, 3> M;
        for (int i = 0; i < 9; ++i)
        {
            M.data[i] = rng.nextFloat();
        }

        auto I = SquareMat<float, 3>::identity();
        auto MI = M * I;
        auto IM = I * M;

        for (int i = 0; i < 9; ++i)
        {
            EXPECT_NEAR(MI.data[i], M.data[i], 1e-4f);
            EXPECT_NEAR(IM.data[i], M.data[i], 1e-4f);
        }
    }
}

// Property: Matrix transpose twice returns original
TEST(PropertyBased, MatrixTransposeInvolution)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Mat<float, 3, 4> M;
        for (int i = 0; i < 12; ++i)
        {
            M.data[i] = rng.nextFloat();
        }

        auto MTT = M.transpose().transpose();

        for (int i = 0; i < 12; ++i)
        {
            EXPECT_FLOAT_EQ(MTT.data[i], M.data[i]);
        }
    }
}

// Property: Transpose of product equals product of transposes (reversed)
TEST(PropertyBased, MatrixTransposeProduct)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 100; ++trial)
    {
        Mat<float, 3, 4> A;
        Mat<float, 4, 2> B;

        for (int i = 0; i < 12; ++i)
            A.data[i] = rng.nextFloatSmall();
        for (int i = 0; i < 8; ++i)
            B.data[i] = rng.nextFloatSmall();

        auto AB_T = (A * B).transpose();
        auto BT_AT = B.transpose() * A.transpose();

        for (int i = 0; i < 6; ++i)
        {
            EXPECT_NEAR(AB_T.data[i], BT_AT.data[i], 1e-3f);
        }
    }
}

// Property: Matrix inverse relationship
TEST(PropertyBased, MatrixInverseRelationship)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 100; ++trial)
    {
        // Create invertible matrix (diagonal dominant)
        SquareMat<float, 3> M;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (i == j)
                {
                    M(i, j) = 5.0f + std::abs(rng.nextFloat());
                }
                else
                {
                    M(i, j) = rng.nextFloatSmall() * 0.1f;
                }
            }
        }

        auto Minv = M.inverse();
        auto I = M * Minv;
        auto Iexpected = SquareMat<float, 3>::identity();

        for (int i = 0; i < 9; ++i)
        {
            EXPECT_NEAR(I.data[i], Iexpected.data[i], 1e-2f);
        }
    }
}

// Property: Quaternion multiplication non-commutative
TEST(PropertyBased, QuaternionMultiplicationNonCommutative)
{
    RandomGenerator rng;

    int non_commutative_count = 0;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Quaternion<float> q1(rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        Quaternion<float> q2(rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        q1 = q1.normalized();
        q2 = q2.normalized();

        auto q12 = q1 * q2;
        auto q21 = q2 * q1;

        // Count how many are not commutative
        if (std::abs(q12.w() - q21.w()) > 1e-4f || std::abs(q12.x() - q21.x()) > 1e-4f ||
            std::abs(q12.y() - q21.y()) > 1e-4f || std::abs(q12.z() - q21.z()) > 1e-4f)
        {
            non_commutative_count++;
        }
    }

    // Most quaternion multiplications should not commute
    EXPECT_GT(non_commutative_count, 900);
}

// Property: Quaternion normalization preserves rotation
TEST(PropertyBased, QuaternionNormalizationPreservesRotation)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Quaternion<float> q(rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        if (q.norm() < 1e-6f)
            continue;

        Vec3f v(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        auto q_norm = q.normalized();
        auto rotated1 = q.rotateVector(v);
        auto rotated2 = q_norm.rotateVector(v);

        // Normalized quaternion should give same rotation
        EXPECT_NEAR(rotated1.x(), rotated2.x(), 1e-2f);
        EXPECT_NEAR(rotated1.y(), rotated2.y(), 1e-2f);
        EXPECT_NEAR(rotated1.z(), rotated2.z(), 1e-2f);
    }
}

// Property: Quaternion conjugate relationships
TEST(PropertyBased, QuaternionConjugateRelationship)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Quaternion<float> q(rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        q = q.normalized();

        auto qconj = q.conjugate();
        auto qqconj = q * qconj;

        // q * q* should be identity (1, 0, 0, 0)
        EXPECT_NEAR(qqconj.w(), 1.0f, 1e-4f);
        EXPECT_NEAR(qqconj.x(), 0.0f, 1e-4f);
        EXPECT_NEAR(qqconj.y(), 0.0f, 1e-4f);
        EXPECT_NEAR(qqconj.z(), 0.0f, 1e-4f);
    }
}

// Property: Rotation matrix determinant is 1
TEST(PropertyBased, RotationMatrixDeterminant)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        float angle = rng.nextFloat();

        // Test rotation around different axes
        auto Rx = SquareMat<float, 3>::rotateX(angle);
        auto Ry = SquareMat<float, 3>::rotateY(angle);
        auto Rz = SquareMat<float, 3>::rotateZ(angle);

        float det_x = Rx.det();
        float det_y = Ry.det();
        float det_z = Rz.det();

        EXPECT_NEAR(det_x, 1.0f, 1e-4f);
        EXPECT_NEAR(det_y, 1.0f, 1e-4f);
        EXPECT_NEAR(det_z, 1.0f, 1e-4f);
    }
}

// Property: Rotation matrix preserves vector length
TEST(PropertyBased, RotationPreservesLength)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f v(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
        float angle = rng.nextFloat();

        auto R = SquareMat<float, 3>::rotateZ(angle);
        auto rotated = R * v;

        float original_len = v.norm();
        float rotated_len = rotated.norm();

        EXPECT_NEAR(original_len, rotated_len, 1e-4f);
    }
}

// Property: Numerical stability - no NaN/Inf
TEST(PropertyBased, NoNaNOrInf)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        Vec3f v(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());

        if (v.normSquared() < 1e-10f)
            continue;

        Vec3f n = v.normalized();

        EXPECT_FALSE(std::isnan(n.x()));
        EXPECT_FALSE(std::isnan(n.y()));
        EXPECT_FALSE(std::isnan(n.z()));
        EXPECT_FALSE(std::isinf(n.x()));
        EXPECT_FALSE(std::isinf(n.y()));
        EXPECT_FALSE(std::isinf(n.z()));
    }
}

// Property: Trace is sum of diagonal elements
TEST(PropertyBased, MatrixTraceSumOfDiagonal)
{
    RandomGenerator rng;

    for (int trial = 0; trial < 1000; ++trial)
    {
        SquareMat<float, 4> M;
        for (int i = 0; i < 16; ++i)
        {
            M.data[i] = rng.nextFloat();
        }

        float trace = M.trace();
        float manual_trace = M(0, 0) + M(1, 1) + M(2, 2) + M(3, 3);

        EXPECT_FLOAT_EQ(trace, manual_trace);
    }
}
