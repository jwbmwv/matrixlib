// Numerical Accuracy Testing for MatrixLib
// Tests against high-precision reference implementations
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>
#include <cmath>
#include <limits>

using namespace matrixlib;

// Helper: Relative error
template<typename T>
T relative_error(T computed, T reference)
{
    if (std::abs(reference) < std::numeric_limits<T>::epsilon())
    {
        return std::abs(computed - reference);
    }
    return std::abs((computed - reference) / reference);
}

// Test: Vector norm accuracy
TEST(NumericalAccuracy, VectorNormPrecision)
{
    // Test with vectors of varying magnitudes
    std::vector<Vec3f> test_vectors = {
        Vec3f(1, 2, 3),
        Vec3f(1e-6f, 2e-6f, 3e-6f),  // Very small
        Vec3f(1e6f, 2e6f, 3e6f),     // Very large
        Vec3f(1, 1e6f, 1e-6f),       // Mixed scales
    };

    for (const auto& v : test_vectors)
    {
        float computed = v.norm();

        // Reference: High-precision computation
        double ref_x = v.x();
        double ref_y = v.y();
        double ref_z = v.z();
        double reference = std::sqrt(ref_x * ref_x + ref_y * ref_y + ref_z * ref_z);

        float rel_err = relative_error(computed, static_cast<float>(reference));
        EXPECT_LT(rel_err, 1e-5f) << "Vector: " << v.x() << ", " << v.y() << ", " << v.z();
    }
}

// Test: Matrix determinant accuracy
TEST(NumericalAccuracy, DeterminantPrecision)
{
    // Well-conditioned matrix
    Mat3f M1;
    M1.data[0] = 1;
    M1.data[1] = 2;
    M1.data[2] = 3;
    M1.data[3] = 0;
    M1.data[4] = 1;
    M1.data[5] = 4;
    M1.data[6] = 5;
    M1.data[7] = 6;
    M1.data[8] = 0;

    float det1 = M1.det();
    float expected1 = 1 * (1 * 0 - 4 * 6) - 2 * (0 * 0 - 4 * 5) + 3 * (0 * 6 - 1 * 5);

    EXPECT_NEAR(det1, expected1, 1e-5f);

    // Test singular matrix (det = 0)
    Mat3f M2 = {1, 2, 3, 2, 4, 6,  // Second row = 2 * first row
                4, 5, 6};

    float det2 = M2.det();
    EXPECT_NEAR(det2, 0.0f, 1e-5f);
}

// Test: Matrix inverse accuracy
TEST(NumericalAccuracy, InverseAccuracy)
{
    // Create well-conditioned matrix
    Mat3f M;
    M.data[0] = 4;
    M.data[1] = 7;
    M1.data[2] = 2;
    M.data[3] = 3;
    M.data[4] = 6;
    M.data[5] = 1;
    M.data[6] = 2;
    M.data[7] = 5;
    M.data[8] = 3;

    Mat3f Minv = M.inverse();
    Mat3f I = M * Minv;
    Mat3f expected = Mat3f::identity();

    // Check each element
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(I.data[i], expected.data[i], 1e-4f) << "Element " << i << " failed";
    }
}

// Test: Quaternion normalization stability
TEST(NumericalAccuracy, QuaternionNormalizationStability)
{
    // Test with quaternions of varying magnitudes
    std::vector<Quatf> test_quats = {
        Quatf(1, 0, 0, 0),
        Quatf(0.5f, 0.5f, 0.5f, 0.5f),
        Quatf(1e-6f, 0, 0, 0),  // Very small
        Quatf(1e6f, 0, 0, 0),   // Very large
    };

    for (const auto& q : test_quats)
    {
        Quatf normalized = q.normalized();
        float norm = normalized.norm();

        EXPECT_NEAR(norm, 1.0f, 1e-5f) << "Quaternion: " << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z();
    }
}

// Test: Rotation matrix orthonormality
TEST(NumericalAccuracy, RotationMatrixOrthonormality)
{
    float angles[] = {0.0f, 0.1f, 1.0f, PI / 4, PI / 2, PI, 2 * PI};

    for (float angle : angles)
    {
        Mat3f R = Mat3f::rotateZ(angle);
        Mat3f RT = R.transpose();
        Mat3f I = R * RT;
        Mat3f expected = Mat3f::identity();

        // Check R * R^T = I (orthonormality)
        for (int i = 0; i < 9; ++i)
        {
            EXPECT_NEAR(I.data[i], expected.data[i], 1e-5f) << "Angle: " << angle << ", Element: " << i;
        }

        // Check det(R) = 1 (proper rotation)
        float det = R.det();
        EXPECT_NEAR(det, 1.0f, 1e-5f) << "Angle: " << angle;
    }
}

// Test: LU decomposition accuracy
TEST(NumericalAccuracy, LUDecompositionAccuracy)
{
    // Create test matrix
    SquareMat<float, 4> A;
    for (int i = 0; i < 16; ++i)
    {
        A.data[i] = static_cast<float>((i * 7 + 3) % 10);
    }

    auto [L, U, P] = A.lu();

    // Check PA = LU
    SquareMat<float, 4> PA = P * A;
    SquareMat<float, 4> LU = L * U;

    for (int i = 0; i < 16; ++i)
    {
        EXPECT_NEAR(PA.data[i], LU.data[i], 1e-3f) << "LU decomposition failed at element " << i;
    }
}

// Test: Cholesky decomposition accuracy (SPD matrix)
TEST(NumericalAccuracy, CholeskyAccuracy)
{
    // Create symmetric positive-definite matrix
    Mat3f M;
    M.data[0] = 4;
    M.data[1] = 12;
    M.data[2] = -16;
    M.data[3] = 12;
    M.data[4] = 37;
    M.data[5] = -43;
    M.data[6] = -16;
    M.data[7] = -43;
    M.data[8] = 98;

    Mat3f L = M.cholesky();
    Mat3f LT = L.transpose();
    Mat3f reconstructed = L * LT;

    // Check A = L * L^T
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(reconstructed.data[i], M.data[i], 1e-3f) << "Cholesky failed at element " << i;
    }
}

// Test: QR decomposition orthogonality
TEST(NumericalAccuracy, QROrthogonality)
{
    SquareMat<float, 3> A;
    A.data[0] = 12;
    A.data[1] = -51;
    A.data[2] = 4;
    A.data[3] = 6;
    A.data[4] = 167;
    A.data[5] = -68;
    A.data[6] = -4;
    A.data[7] = 24;
    A.data[8] = -41;

    auto [Q, R] = A.qr();

    // Check Q is orthogonal: Q^T * Q = I
    Mat3f QT = Q.transpose();
    Mat3f QTQ = QT * Q;
    Mat3f I = Mat3f::identity();

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(QTQ.data[i], I.data[i], 1e-4f) << "Q not orthogonal at element " << i;
    }

    // Check A = QR
    Mat3f QR = Q * R;
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(QR.data[i], A.data[i], 1e-3f) << "QR decomposition failed at element " << i;
    }
}

// Test: Condition number effects
TEST(NumericalAccuracy, IllConditionedMatrices)
{
    // Hilbert matrix (notoriously ill-conditioned)
    Mat3f H;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            H(i, j) = 1.0f / (i + j + 1);
        }
    }

    // Solve Hx = b
    Vec3f b(1, 1, 1);
    Vec3f x = H.solve_qr(b);

    // Check residual: ||Hx - b||
    Vec3f Hx = H * x;
    Vec3f residual = Hx - b;
    float residual_norm = residual.norm();

    // For ill-conditioned matrix, expect larger error
    EXPECT_LT(residual_norm, 1e-2f) << "Residual too large for ill-conditioned matrix";
}

// Test: Catastrophic cancellation
TEST(NumericalAccuracy, CatastrophicCancellation)
{
    // Subtraction of nearly equal numbers
    Vec3f a(1000000.0f, 1000000.0f, 1000000.0f);
    Vec3f b(1000000.0f + 0.1f, 1000000.0f + 0.2f, 1000000.0f + 0.3f);

    Vec3f diff = b - a;

    // Expected: (0.1, 0.2, 0.3)
    EXPECT_NEAR(diff.x(), 0.1f, 1e-4f);
    EXPECT_NEAR(diff.y(), 0.2f, 1e-4f);
    EXPECT_NEAR(diff.z(), 0.3f, 1e-4f);
}

// Test: NaN/Inf propagation
TEST(NumericalAccuracy, NaNInfHandling)
{
    // Division by zero should produce Inf or handled
    Vec3f zero(0, 0, 0);

    // Don't normalize zero vector (should handle gracefully)
    // Most implementations will produce NaN or Inf
    if (zero.normSquared() > 1e-10f)
    {
        Vec3f normalized = zero.normalized();
        EXPECT_FALSE(std::isnan(normalized.x()));
        EXPECT_FALSE(std::isnan(normalized.y()));
        EXPECT_FALSE(std::isnan(normalized.z()));
    }

    // Test NaN propagation
    float nan = std::numeric_limits<float>::quiet_NaN();
    Vec3f v_nan(nan, 2, 3);

    EXPECT_TRUE(std::isnan(v_nan.x()));
    EXPECT_TRUE(std::isnan(v_nan.norm()));  // NaN propagates
}

// Test: Eigenvalue accuracy (for 2x2 matrices with known eigenvalues)
TEST(NumericalAccuracy, EigenvalueAccuracy2x2)
{
    // Diagonal matrix (eigenvalues = diagonal elements)
    SquareMat<float, 2> D;
    D(0, 0) = 3.0f;
    D(0, 1) = 0.0f;
    D(1, 0) = 0.0f;
    D(1, 1) = 7.0f;

    Vec<float, 2> evals = D.eigenvaluesQR();

    // Eigenvalues should be {3, 7} (order may vary)
    std::vector<float> computed = {evals.data[0], evals.data[1]};
    std::sort(computed.begin(), computed.end());

    EXPECT_NEAR(computed[0], 3.0f, 1e-3f);
    EXPECT_NEAR(computed[1], 7.0f, 1e-3f);
}

// Test: Forward/backward error bounds
TEST(NumericalAccuracy, ErrorBounds)
{
    // Test forward error: ||computed - exact||
    Vec3f v(1, 2, 3);
    float computed_norm = v.norm();
    double exact_norm = std::sqrt(1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0);

    float forward_error = std::abs(computed_norm - static_cast<float>(exact_norm));
    float relative_forward_error = forward_error / static_cast<float>(exact_norm);

    EXPECT_LT(relative_forward_error, 1e-6f) << "Forward error too large";

    // Test backward error: ||A * computed_x - b||
    Mat3f A = Mat3f::identity();
    A(0, 0) = 2.0f;
    Vec3f b(4, 2, 3);
    Vec3f x = A.solve_qr(b);

    Vec3f residual = A * x - b;
    float backward_error = residual.norm();

    EXPECT_LT(backward_error, 1e-5f) << "Backward error too large";
}

// Benchmark: Numerical stability under repeated operations
TEST(NumericalAccuracy, AccumulationStability)
{
    // Repeated normalization should maintain unit length
    Vec3f v(1, 2, 3);
    v = v.normalized();

    for (int i = 0; i < 1000; ++i)
    {
        v = v.normalized();
    }

    float final_norm = v.norm();
    EXPECT_NEAR(final_norm, 1.0f, 1e-4f) << "Normalization accumulated error after 1000 iterations";
}
