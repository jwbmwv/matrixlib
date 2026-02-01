# Common Mistakes and How to Avoid Them

This guide covers frequent pitfalls when using MatrixLib and how to avoid them.

## Table of Contents

1. [Matrix/Vector Confusion](#matrixvector-confusion)
2. [Rotation Issues](#rotation-issues)
3. [Numerical Stability](#numerical-stability)
4. [Performance Pitfalls](#performance-pitfalls)
5. [Memory Issues](#memory-issues)
6. [API Misuse](#api-misuse)
7. [Build/Compilation Issues](#buildcompilation-issues)

---

## Matrix/Vector Confusion

### ❌ Mistake: Row-Major vs Column-Major Confusion

```cpp
// OpenGL uses column-major, MatrixLib uses row-major
Mat4f M = /* ... */;
glUniformMatrix4fv(loc, 1, GL_TRUE, M.data);  // ❌ Wrong - need transpose
```

**✅ Solution:**
```cpp
// Option 1: Transpose when uploading
glUniformMatrix4fv(loc, 1, GL_TRUE, M.data);  // GL_TRUE = transpose

// Option 2: Transpose in code
Mat4f MT = M.transpose();
glUniformMatrix4fv(loc, 1, GL_FALSE, MT.data);

// Option 3: Build transposed from start
Mat4f M_col_major = build_column_major();
```

### ❌ Mistake: Matrix Dimension Mismatch

```cpp
Mat<float, 3, 4> A;
Mat<float, 3, 3> B;
auto C = A * B;  // ❌ Compile error: 4 != 3
```

**✅ Solution:**
```cpp
// Dimensions must match: (M×N) * (N×P) = (M×P)
Mat<float, 3, 4> A;  // 3 rows, 4 cols
Mat<float, 4, 2> B;  // 4 rows, 2 cols  - middle dimension matches!
auto C = A * B;      // ✅ Result is 3×2
```

**Tip:** Compiler will catch this at compile-time - read the error message!

---

## Rotation Issues

### ❌ Mistake: Quaternion Multiplication Order

```cpp
Quatf q1 = Quatf::fromAxisAngle(Vec3f(1,0,0), angle1);
Quatf q2 = Quatf::fromAxisAngle(Vec3f(0,1,0), angle2);
Quatf combined = q1 * q2;  // ❌ Applies q2 first, then q1!
```

**✅ Solution:**
```cpp
// Quaternion multiplication is RIGHT-to-LEFT (like matrices)
Quatf combined = q2 * q1;  // ✅ Applies q1 first, then q2

// To apply rotations in order: q1, then q2, then q3
Quatf result = q3 * q2 * q1;  // Right-to-left
```

**Mnemonic:** "Read quaternion chains backwards" or "Matrix multiplication order"

### ❌ Mistake: Gimbal Lock with Euler Angles

```cpp
// Pitch = ±90° causes gimbal lock
float roll = 0.0f;
float pitch = PI / 2.0f;  // ❌ 90 degrees - gimbal lock!
float yaw = /* anything */;
Mat3f R = Mat3f::fromEuler(roll, pitch, yaw);
```

**✅ Solution:**
```cpp
// Use quaternions instead of Euler angles
Quatf q = Quatf::fromAxisAngle(axis, angle);
Mat3f R = q.toMatrix();

// Or use axis-angle directly
Mat3f R = Mat3f::fromAxisAngle(axis, angle);
```

**Explanation:** Euler angles have singularities at ±90°. Quaternions don't have this limitation.

### ❌ Mistake: Not Normalizing Quaternions

```cpp
Quatf q(1, 0.5, 0.5, 0);  // ❌ Not unit quaternion!
Vec3f rotated = q.rotateVector(v);  // ❌ Wrong rotation
```

**✅ Solution:**
```cpp
Quatf q(1, 0.5, 0.5, 0);
q = q.normalized();  // ✅ Always normalize!
Vec3f rotated = q.rotateVector(v);

// Or use factory methods (already normalized)
Quatf q = Quatf::fromAxisAngle(axis, angle);  // ✅ Already normalized
```

### ❌ Mistake: Accumulating Rotation Errors

```cpp
Mat3f R = Mat3f::identity();
for (int i = 0; i < 1000; ++i)
{
    R = R * small_rotation;  // ❌ Numerical errors accumulate!
}
// R is no longer a valid rotation matrix (det ≠ 1)
```

**✅ Solution:**
```cpp
// Option 1: Use quaternions (more stable)
Quatf q = Quatf::identity();
for (int i = 0; i < 1000; ++i)
{
    q = q * small_rotation_quat;
    q = q.normalized();  // ✅ Renormalize periodically
}

// Option 2: Orthonormalize matrix periodically
Mat3f R = Mat3f::identity();
for (int i = 0; i < 1000; ++i)
{
    R = R * small_rotation;
    if (i % 100 == 0)  // Every 100 iterations
    {
        R = R.orthonormalize();  // ✅ Restore orthonormality
    }
}

// Option 3: Store angle, recompute matrix
float total_angle = 0.0f;
for (int i = 0; i < 1000; ++i)
{
    total_angle += small_angle;
}
Mat3f R = Mat3f::rotateZ(total_angle);  // ✅ No accumulation
```

---

## Numerical Stability

### ❌ Mistake: Comparing Floats with ==

```cpp
Vec3f v1(1.0f, 2.0f, 3.0f);
Vec3f v2 = v1.normalized() * v1.norm();
if (v1 == v2)  // ❌ Likely false due to floating-point error!
{
    // May not execute even though mathematically equal
}
```

**✅ Solution:**
```cpp
// Use epsilon comparison
const float EPSILON = 1e-5f;
bool equal = (v1 - v2).normSquared() < EPSILON * EPSILON;

// Or element-wise
bool equal = std::abs(v1.x() - v2.x()) < EPSILON &&
             std::abs(v1.y() - v2.y()) < EPSILON &&
             std::abs(v1.z() - v2.z()) < EPSILON;
```

### ❌ Mistake: Division by Zero

```cpp
Vec3f v(0, 0, 0);
Vec3f n = v.normalized();  // ❌ Division by zero!
```

**✅ Solution:**
```cpp
Vec3f v = /* ... */;
if (v.normSquared() > 1e-10f)  // ✅ Check before normalizing
{
    Vec3f n = v.normalized();
}
else
{
    // Handle zero vector case
    Vec3f n = Vec3f(1, 0, 0);  // Default direction
}
```

### ❌ Mistake: Matrix Inversion of Singular Matrix

```cpp
Mat3f M = /* nearly singular matrix */;
Mat3f Minv = M.inverse();  // ❌ Unstable or invalid result!
```

**✅ Solution:**
```cpp
// Check determinant first
float det = M.det();
if (std::abs(det) > 1e-6f)  // ✅ Check for singularity
{
    Mat3f Minv = M.inverse();
}
else
{
    // Matrix is singular or nearly singular
    // Use pseudo-inverse or different approach
}

// Or use decomposition-based solver
Vec3f x = M.solve_qr(b);  // ✅ More stable than inverse
```

### ❌ Mistake: Loss of Precision in Subtraction

```cpp
Vec3f a(1000000.0f, 2000000.0f, 3000000.0f);
Vec3f b(1000000.1f, 2000000.1f, 3000000.1f);
Vec3f diff = a - b;  // ❌ Loss of precision in large numbers
```

**✅ Solution:**
```cpp
// Store offsets instead of absolute positions
Vec3f origin(1000000.0f, 2000000.0f, 3000000.0f);
Vec3f a_offset(0.0f, 0.0f, 0.0f);
Vec3f b_offset(0.1f, 0.1f, 0.1f);
Vec3f diff = a_offset - b_offset;  // ✅ No precision loss

// Or use double precision
Vec3d a(1000000.0, 2000000.0, 3000000.0);
Vec3d b(1000000.1, 2000000.1, 3000000.1);
Vec3d diff = a - b;  // ✅ Better precision
```

---

## Performance Pitfalls

### ❌ Mistake: Unnecessary Normalization

```cpp
for (int i = 0; i < 10000; ++i)
{
    float len = v.norm();  // ❌ Expensive sqrt every iteration!
    if (len > threshold)
    {
        // ...
    }
}
```

**✅ Solution:**
```cpp
// Compare squared lengths (no sqrt)
float threshold_sq = threshold * threshold;
for (int i = 0; i < 10000; ++i)
{
    float len_sq = v.normSquared();  // ✅ No sqrt!
    if (len_sq > threshold_sq)
    {
        // ...
    }
}
```

### ❌ Mistake: Recomputing Trigonometric Values

```cpp
for (int i = 0; i < 1000; ++i)
{
    Mat3f R = Mat3f::rotateZ(angle);  // ❌ Recomputes sin/cos each time!
    Vec3f rotated = R * points[i];
}
```

**✅ Solution:**
```cpp
// Compute rotation matrix once
Mat3f R = Mat3f::rotateZ(angle);  // ✅ Once!
for (int i = 0; i < 1000; ++i)
{
    Vec3f rotated = R * points[i];
}

// Or cache sin/cos
float c = std::cos(angle);
float s = std::sin(angle);
for (int i = 0; i < 1000; ++i)
{
    Vec2f rotated(
        c * points[i].x() - s * points[i].y(),
        s * points[i].x() + c * points[i].y()
    );
}
```

### ❌ Mistake: Creating Unnecessary Temporaries

```cpp
Vec3f result = (a + b) + (c + d);  // ❌ Creates intermediate temporaries
```

**✅ Solution:**
```cpp
// Manually optimize (compiler may do this anyway)
Vec3f result = a;
result += b;
result += c;
result += d;  // ✅ No temporaries with compound assignment

// Or let compiler optimize
Vec3f result = a + b + c + d;  // Modern compilers optimize this well
```

### ❌ Mistake: Matrix Inverse Instead of Solve

```cpp
Mat3f Ainv = A.inverse();  // ❌ Expensive: O(N³)
Vec3f x = Ainv * b;        // ❌ Another O(N³) multiply!
// Total: O(2N³)
```

**✅ Solution:**
```cpp
// Direct solve is faster and more stable
Vec3f x = A.solve_qr(b);  // ✅ Only O(N³) total
```

---

## Memory Issues

### ❌ Mistake: Stack Overflow with Large Matrices

```cpp
void process()
{
    Mat<float, 1000, 1000> M;  // ❌ 4MB on stack - may overflow!
}
```

**✅ Solution:**
```cpp
// Option 1: Use heap allocation
void process()
{
    auto M = std::make_unique<Mat<float, 1000, 1000>>();
}

// Option 2: Use dynamic allocation
std::vector<float> data(1000 * 1000);

// Option 3: Sparse matrix representation (future feature)
```

### ❌ Mistake: Unaligned SIMD Access

```cpp
float* unaligned_data = new float[16];  // ❌ May not be 16-byte aligned!
float32x4_t vec = vld1q_f32(unaligned_data);  // ❌ May crash on ARM!
```

**✅ Solution:**
```cpp
// Use aligned allocation
alignas(16) float aligned_data[16];  // ✅ Guaranteed 16-byte aligned
float32x4_t vec = vld1q_f32(aligned_data);

// Or use MatrixLib types (already aligned)
Vec4f v;  // ✅ Automatically aligned
Mat4f M;  // ✅ Automatically aligned
```

---

## API Misuse

### ❌ Mistake: Confusing operator[] and operator()

```cpp
Mat3f M;
float elem = M[1][2];  // ❌ Gets row 1, then tries [2] on RowProxy
```

**✅ Solution:**
```cpp
Mat3f M;
float elem = M(1, 2);  // ✅ Use operator(row, col)

// operator[] returns RowProxy, use like this:
auto row = M[1];
float elem = row[2];   // ✅ Two-step access
```

### ❌ Mistake: Modifying const View

```cpp
const Mat4f M = Mat4f::identity();
M(0, 0) = 2.0f;  // ❌ Compile error: M is const
```

**✅ Solution:**
```cpp
Mat4f M = Mat4f::identity();  // ✅ Non-const
M(0, 0) = 2.0f;

// Or use mutable copy
const Mat4f original = Mat4f::identity();
Mat4f M = original;  // ✅ Mutable copy
M(0, 0) = 2.0f;
```

### ❌ Mistake: Using Wrong Solve Method

```cpp
// Matrix is symmetric positive-definite
Mat3f A = /* SPD matrix */;
Vec3f x = A.solve_qr(b);  // ❌ Works but slower than needed
```

**✅ Solution:**
```cpp
// Use Cholesky for SPD matrices (2× faster)
Mat3f A = /* SPD matrix */;
Vec3f x = A.solve_cholesky(b);  // ✅ Faster!

// QR is more general, use for:
// - Non-square matrices
// - Least-squares problems
// - Overdetermined systems
```

---

## Build/Compilation Issues

### ❌ Mistake: Missing SIMD Compiler Flags

```cpp
// Code uses NEON but not enabled
#if defined(CONFIG_MATRIXLIB_NEON)
    // NEON code never executes!
#endif
```

**✅ Solution:**
```cmake
# CMakeLists.txt
option(MATRIXLIB_ENABLE_NEON "Enable NEON" ON)
if(MATRIXLIB_ENABLE_NEON)
    target_compile_definitions(matrixlib INTERFACE CONFIG_MATRIXLIB_NEON)
    target_compile_options(matrixlib INTERFACE -mfpu=neon)
endif()
```

### ❌ Mistake: C++11 vs C++14/17 Features

```cpp
// Using C++14 constexpr in functions
constexpr float compute()  // ❌ Fails in C++11!
{
    float x = 1.0f;
    return x + 1.0f;
}
```

**✅ Solution:**
```cpp
// Use MATRIX_CONSTEXPR macro
MATRIX_CONSTEXPR float compute()  // ✅ Works in C++11 and C++14+
{
    return 2.0f;
}

// Or use compatibility macros
#if __cplusplus >= 201402L
    constexpr
#else
    inline
#endif
float compute() { return 2.0f; }
```

### ❌ Mistake: Debug Assertions in Release

```cpp
// Forgot to define MATRIXLIB_DEBUG
Vec3f v;
v.at(10);  // ❌ No assertion in release, crashes!
```

**✅ Solution:**
```cmake
# Add for debug builds
target_compile_definitions(your_target PRIVATE 
    $<$<CONFIG:Debug>:MATRIXLIB_DEBUG>
)

# Or manually
cmake .. -DCMAKE_BUILD_TYPE=Debug
// Automatically enables assertions
```

---

## Prevention Checklist

Before committing code, check:

- [ ] Quaternions normalized before use
- [ ] Float comparisons use epsilon
- [ ] Matrix dimensions match for operations
- [ ] No division by near-zero values
- [ ] Large arrays use heap, not stack
- [ ] SIMD data is aligned
- [ ] Rotation matrices renormalized if accumulated
- [ ] Use `normSquared()` for length comparisons
- [ ] Proper solver chosen (QR vs Cholesky vs LU)
- [ ] Debug builds have `-DMATRIXLIB_DEBUG`

---

## Additional Resources

- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Cookbook](docs/COOKBOOK.md)
- [Performance Guide](PERFORMANCE.md)
