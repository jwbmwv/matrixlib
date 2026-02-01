# Migration Guides

Guides for migrating from other linear algebra libraries to MatrixLib.

## Table of Contents

1. [Eigen → MatrixLib](#eigen--matrixlib)
2. [GLM → MatrixLib](#glm--matrixlib)
3. [DirectXMath → MatrixLib](#directxmath--matrixlib)
4. [NumPy → MatrixLib](#numpy--matrixlib-for-c-users)

---

## Eigen → MatrixLib

### Philosophy Differences

| Aspect | Eigen | MatrixLib |
|--------|-------|-----------|
| **Target** | Desktop, scientific computing | Embedded, real-time systems |
| **Memory** | Heap + stack, dynamic sizes | Stack only, compile-time sizes |
| **Dependencies** | C++11/14/17 | C++11 minimum |
| **Optimization** | Expression templates, lazy eval | Direct SIMD, eager eval |
| **Size** | Large (100+ headers) | Compact (12 headers) |

### Type Mapping

```cpp
// Eigen → MatrixLib
Eigen::Vector2f         →  Vec2f
Eigen::Vector3f         →  Vec3f
Eigen::Vector4f         →  Vec4f
Eigen::Matrix2f         →  Mat2f
Eigen::Matrix3f         →  Mat3f
Eigen::Matrix4f         →  Mat4f
Eigen::Quaternionf      →  Quatf
Eigen::AngleAxisf       →  Quatf::fromAxisAngle()

// Dynamic → Fixed (MatrixLib uses compile-time sizes)
Eigen::VectorXf         →  Vec<float, N>  // Choose N at compile-time
Eigen::MatrixXf         →  Mat<float, R, C>
```

### Common Operations

#### Vector Operations

```cpp
// Eigen
Eigen::Vector3f a(1, 2, 3);
Eigen::Vector3f b(4, 5, 6);
float dot = a.dot(b);
Eigen::Vector3f cross = a.cross(b);
float len = a.norm();
Eigen::Vector3f normalized = a.normalized();

// MatrixLib (nearly identical!)
Vec3f a(1, 2, 3);
Vec3f b(4, 5, 6);
float dot = a.dot(b);
Vec3f cross = a.cross(b);
float len = a.norm();
Vec3f normalized = a.normalized();
```

#### Matrix Operations

```cpp
// Eigen
Eigen::Matrix3f A, B;
Eigen::Matrix3f C = A * B;
Eigen::Matrix3f T = A.transpose();
Eigen::Matrix3f I = A.inverse();
float d = A.determinant();

// MatrixLib
Mat3f A, B;
Mat3f C = A * B;
Mat3f T = A.transpose();
Mat3f I = A.inverse();
float d = A.det();  // Note: det() not determinant()
```

#### Linear Solvers

```cpp
// Eigen (multiple solver options)
Eigen::Vector3f x = A.lu().solve(b);              // LU decomposition
Eigen::Vector3f x = A.llt().solve(b);             // Cholesky
Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);  // QR

// MatrixLib
Vec3f x = A.solve_qr(b);        // QR decomposition (general)
Vec3f x = A.solve_cholesky(b);  // Cholesky (for SPD matrices)
// Or decompose explicitly:
auto [L, U, P] = A.lu();
auto [Q, R] = A.qr();
Mat3f L = A.cholesky();
```

#### Transformations

```cpp
// Eigen
Eigen::Affine3f transform = Eigen::Affine3f::Identity();
transform.translate(Eigen::Vector3f(1, 2, 3));
transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));

// MatrixLib (build transform matrices directly)
Mat4f T = Mat4f::translate(Vec3f(1, 2, 3));
Mat4f R = Mat4f::fromAxisAngle(Vec3f(0, 0, 1), angle);
Mat4f transform = T * R;
```

### Eigenvalues/Eigenvectors

```cpp
// Eigen
Eigen::EigenSolver<Eigen::Matrix3f> solver(A);
Eigen::Vector3cf eigenvalues = solver.eigenvalues();
Eigen::Matrix3cf eigenvectors = solver.eigenvectors();

// MatrixLib (simplified, real eigenvalues only)
Vec3f eigenvalues = A.eigenvaluesQR();  // Iterative QR algorithm
auto [lambda, v] = A.powerIteration();  // Largest eigenvalue + vector
```

### Migration Checklist

- [ ] Replace `Eigen::VectorXf` with `Vec<float, N>` (choose fixed N)
- [ ] Replace `Eigen::MatrixXf` with `Mat<float, R, C>`
- [ ] Change `.determinant()` → `.det()`
- [ ] Replace `.llt().solve()` → `.solve_cholesky()`
- [ ] Replace `.colPivHouseholderQr().solve()` → `.solve_qr()`
- [ ] Convert `Affine3f` to explicit transformation matrices
- [ ] Remove `#include <Eigen/Dense>`, add `#include <matrixlib/matrixlib.hpp>`

---

## GLM → MatrixLib

### Philosophy Differences

| Aspect | GLM | MatrixLib |
|--------|-----|-----------|
| **Target** | OpenGL graphics | General embedded + graphics |
| **Layout** | Column-major (OpenGL) | Row-major (C/C++) |
| **Swizzling** | Yes (vec.xyz, vec.xxyy) | No (use explicit accessors) |
| **Extensions** | Many (gtx, gtc) | Core only |

### Type Mapping

```cpp
// GLM → MatrixLib
glm::vec2          →  Vec2f
glm::vec3          →  Vec3f
glm::vec4          →  Vec4f
glm::mat2          →  Mat2f
glm::mat3          →  Mat3f
glm::mat4          →  Mat4f
glm::quat          →  Quatf
```

### Common Operations

#### Vector Operations

```cpp
// GLM
glm::vec3 a(1, 2, 3);
glm::vec3 b(4, 5, 6);
float dot = glm::dot(a, b);
glm::vec3 cross = glm::cross(a, b);
float len = glm::length(a);
glm::vec3 normalized = glm::normalize(a);

// MatrixLib (methods instead of free functions)
Vec3f a(1, 2, 3);
Vec3f b(4, 5, 6);
float dot = a.dot(b);      // Method, not free function
Vec3f cross = a.cross(b);
float len = a.norm();       // norm() not length()
Vec3f normalized = a.normalized();
```

#### Matrix Operations

```cpp
// GLM
glm::mat4 proj = glm::perspective(fov, aspect, near, far);
glm::mat4 view = glm::lookAt(eye, center, up);
glm::mat4 model = glm::rotate(glm::mat4(1.0f), angle, axis);

// MatrixLib
Mat4f proj = Mat4f::perspective(fov, aspect, near, far);
Mat4f view = Mat4f::lookAt(eye, center, up);
Mat4f model = Mat4f::fromAxisAngle(axis, angle);
```

#### Transformations

```cpp
// GLM
glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3));
glm::mat4 R = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 1, 0));
glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(2, 2, 2));

// MatrixLib
Mat4f T = Mat4f::translate(Vec3f(1, 2, 3));
Mat4f R = Mat4f::fromAxisAngle(Vec3f(0, 1, 0), angle);
Mat4f S = Mat4f::scale(Vec3f(2, 2, 2));
```

### Important: Column-Major vs Row-Major

```cpp
// GLM (column-major, matches OpenGL)
glm::mat4 M;
glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(M));

// MatrixLib (row-major, need transpose)
Mat4f M;
Mat4f MT = M.transpose();
glUniformMatrix4fv(location, 1, GL_FALSE, MT.data);
// OR use GL_TRUE to transpose:
glUniformMatrix4fv(location, 1, GL_TRUE, M.data);
```

### Migration Checklist

- [ ] Replace `glm::vec*` with `Vec*f`
- [ ] Replace `glm::mat*` with `Mat*f`
- [ ] Change free functions to methods: `glm::dot(a, b)` → `a.dot(b)`
- [ ] Change `glm::length()` → `.norm()`
- [ ] Add `.transpose()` when passing matrices to OpenGL (or use GL_TRUE)
- [ ] Remove swizzling: `v.xyz` → explicit `Vec3f(v.x(), v.y(), v.z())`
- [ ] Replace `#include <glm/glm.hpp>` with `#include <matrixlib/matrixlib.hpp>`

---

## DirectXMath → MatrixLib

### Philosophy Differences

| Aspect | DirectXMath | MatrixLib |
|--------|-------------|-----------|
| **Target** | DirectX graphics (Windows) | Cross-platform embedded |
| **SIMD** | SSE/AVX intrinsics | ARM NEON + SSE (future) |
| **Namespace** | DirectX:: | matrixlib:: |
| **Alignment** | Must use XMVECTOR for locals | Automatic alignment |

### Type Mapping

```cpp
// DirectXMath → MatrixLib
XMVECTOR           →  Vec4f (or Vec3f if only 3 components used)
XMFLOAT2           →  Vec2f
XMFLOAT3           →  Vec3f
XMFLOAT4           →  Vec4f
XMMATRIX           →  Mat4f
XMFLOAT4X4         →  Mat4f
XMVECTOR (quat)    →  Quatf
```

### Common Operations

#### Vector Operations

```cpp
// DirectXMath
XMVECTOR a = XMVectorSet(1, 2, 3, 0);
XMVECTOR b = XMVectorSet(4, 5, 6, 0);
XMVECTOR c = XMVectorAdd(a, b);
XMVECTOR dot = XMVector3Dot(a, b);
XMVECTOR cross = XMVector3Cross(a, b);
XMVECTOR len = XMVector3Length(a);
XMVECTOR norm = XMVector3Normalize(a);

// MatrixLib (simpler, no SIMD types exposed)
Vec3f a(1, 2, 3);
Vec3f b(4, 5, 6);
Vec3f c = a + b;       // Normal operator
float dot = a.dot(b);  // Returns float, not vector
Vec3f cross = a.cross(b);
float len = a.norm();
Vec3f norm = a.normalized();
```

#### Matrix Operations

```cpp
// DirectXMath
XMMATRIX A = XMMatrixIdentity();
XMMATRIX B = XMMatrixRotationZ(angle);
XMMATRIX C = XMMatrixMultiply(A, B);
XMMATRIX T = XMMatrixTranspose(A);
XMVECTOR det = XMMatrixDeterminant(A);
XMMATRIX inv = XMMatrixInverse(&det, A);

// MatrixLib
Mat4f A = Mat4f::identity();
Mat4f B = Mat4f::rotateZ(angle);
Mat4f C = A * B;       // Normal operator
Mat4f T = A.transpose();
float det = A.det();   // Returns float
Mat4f inv = A.inverse();
```

#### Loading/Storing

```cpp
// DirectXMath (explicit load/store)
XMFLOAT3 float3_data(1, 2, 3);
XMVECTOR vec = XMLoadFloat3(&float3_data);
// ... computations ...
XMStoreFloat3(&float3_data, vec);

// MatrixLib (implicit, types are storage + compute)
Vec3f v(1, 2, 3);  // No load needed
// ... computations ...
// No store needed, v.data directly accessible
```

### Migration Checklist

- [ ] Replace `XMVECTOR` with `Vec3f` or `Vec4f`
- [ ] Replace `XMMATRIX` with `Mat4f`
- [ ] Remove `XMLoad*()` and `XMStore*()` calls
- [ ] Change `XMVector*()` functions to methods: `XMVector3Dot(a,b)` → `a.dot(b)`
- [ ] Change `XMMatrix*()` functions to methods/operators
- [ ] Remove `XMFLOAT*` types (use `Vec*f` directly)
- [ ] Replace `#include <DirectXMath.h>` with `#include <matrixlib/matrixlib.hpp>`

---

## NumPy → MatrixLib (For C++ Users)

If you're a Python/NumPy user learning C++ with MatrixLib:

### Concept Mapping

```python
# NumPy → MatrixLib C++
import numpy as np          →  #include <matrixlib/matrixlib.hpp>
                               using namespace matrixlib;

# Vectors
a = np.array([1, 2, 3])    →  Vec3f a(1, 2, 3);
b = np.array([4, 5, 6])    →  Vec3f b(4, 5, 6);

# Operations
c = a + b                   →  Vec3f c = a + b;
dot = np.dot(a, b)         →  float dot = a.dot(b);
cross = np.cross(a, b)     →  Vec3f cross = a.cross(b);
norm = np.linalg.norm(a)   →  float norm = a.norm();

# Matrices
A = np.eye(3)              →  Mat3f A = Mat3f::identity();
B = np.zeros((3, 3))       →  Mat3f B;  // Zero-initialized
C = A @ B                  →  Mat3f C = A * B;
T = A.T                    →  Mat3f T = A.transpose();
inv = np.linalg.inv(A)     →  Mat3f inv = A.inverse();
det = np.linalg.det(A)     →  float det = A.det();

# Linear solve
x = np.linalg.solve(A, b)  →  Vec3f x = A.solve_qr(b);

# Eigenvalues
evals = np.linalg.eigvals(A)  →  Vec3f evals = A.eigenvaluesQR();
```

### Key Differences

1. **Static Typing**: Must declare types in C++
   ```python
   # Python (dynamic)
   a = [1, 2, 3]
   ```
   ```cpp
   // C++ (static)
   Vec3f a(1, 2, 3);  // Must specify type
   ```

2. **Compile-Time Sizes**: Dimensions fixed at compile-time
   ```python
   # Python (dynamic size)
   a = np.zeros(n)  # n determined at runtime
   ```
   ```cpp
   // C++ (size known at compile-time)
   Vec<float, 100> a;  // Size = 100, must be constant
   ```

3. **No Broadcasting**: Must match dimensions exactly
   ```python
   # Python (broadcasts)
   a = np.array([1, 2, 3])
   b = np.array([[1], [2], [3]])
   c = a + b  # Broadcasting works
   ```
   ```cpp
   // C++ (no broadcasting)
   Vec3f a(1, 2, 3);
   Vec3f b(4, 5, 6);
   Vec3f c = a + b;  // ✅ Same dimensions
   
   // Different dimensions = compile error
   Vec2f d(1, 2);
   Vec3f e = a + d;  // ❌ Compile error!
   ```

---

## General Migration Tips

1. **Start small** - Migrate one module at a time
2. **Test incrementally** - Keep original tests passing
3. **Profile** - Verify performance improvements
4. **Use compiler warnings** - Enable `-Wall -Wextra -Wpedantic`
5. **Read docs** - [API Documentation](docs/API_Documentation.md)

## Support

If you encounter migration issues:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [Common Mistakes](COMMON_MISTAKES.md)
- Open an issue on GitHub
