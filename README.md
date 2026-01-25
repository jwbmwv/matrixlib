# MatrixLib

A lightweight, header-only C++11 linear algebra library optimized for embedded systems and real-time applications.

**Recent Update (Jan 2026):** Major SIMD performance enhancements - 2-4× faster operations with FMA, fast reciprocal/rsqrt, optimized Vec3, and full NEON quaternion/matrix support. See [SIMD_Optimizations.md](docs/SIMD_Optimizations.md).

## Features

- **Header-only**: No compilation required, just include the headers
- **Zero allocation**: All operations use stack memory, no dynamic allocation
- **POD types**: All classes are Plain Old Data compatible with C interfaces and DMA
- **SIMD optimized**: ARM NEON (2-4× speedup) for Cortex-A/ARM64, CMSIS-DSP for Cortex-M processors
  - **FMA instructions**: Fused multiply-add for better performance and accuracy
  - **Fast reciprocal/rsqrt**: Newton-Raphson optimized division and normalize
  - **Vec3 optimized**: Eliminated temporary arrays, 1.4× faster
  - **NEON quaternion multiply**: 3.5× faster than scalar
  - **Matrix SIMD**: 3-4× speedup for 3×3 and 4×4 operations
- **C++ standard adaptive**: Progressive optimizations C++11/14/17/20/23/26, baseline C++11
- **Compile-time rotations**: C++26 constexpr trig, special angles for C++11-C++23
- **Namespaced**: All classes in `matrixlib` namespace to avoid pollution
- **Versioned**: Runtime version API for compatibility checking
- **Type-safe**: Compile-time dimension checking prevents errors
- **Const-correct**: Full const correctness throughout the API
- **Sanitizer-clean**: Zero UB, passes ASan/UBSan/MSan without errors
- **Embedded-friendly**: Designed for microcontrollers with limited resources
- **Rich utilities**: Mathematical constants, angle utilities, interpolation, swizzling
- **Sensor fusion ready**: Coordinate transforms, safe normalization, Euler/quaternion conversions

## Quick Start

### As a Git Submodule

```bash
# Add to your project
git submodule add https://github.com/yourusername/matrixlib.git external/matrixlib
git submodule update --init --recursive

# Update CMakeLists.txt
add_subdirectory(external/matrixlib)
target_link_libraries(your_target PUBLIC matrixlib)
```

### As a Zephyr Module

1. Add to your `west.yml`:

```yaml
manifest:
  projects:
    - name: matrixlib
      url: https://github.com/yourusername/matrixlib.git
      revision: main
      path: modules/lib/matrixlib
```

2. In your application's `CMakeLists.txt`:

```cmake
# MatrixLib is automatically available as a Zephyr module
target_link_libraries(app PUBLIC matrixlib)
```

### Direct Include

For simple projects, just copy `include/matrixlib/` to your include path:

```cpp
// Option 1: Include everything (convenience)
#include <matrixlib/matrixlib.hpp>   // Includes all vector/matrix components
#include <matrixlib/quaternion.hpp>
#include <matrixlib/version.hpp>

// Option 2: Modular includes (faster compile times)
#include <matrixlib/constants.hpp>     // Mathematical constants (can be used independently)
#include <matrixlib/vector.hpp>        // Base Vec<T,N> template
#include <matrixlib/vec2D.hpp>         // Vec2f, Vec2d type aliases
#include <matrixlib/vec3D.hpp>         // Vec3f, Vec3d type aliases
#include <matrixlib/matrix.hpp>        // Base Mat<T,R,C> templates
#include <matrixlib/matrix2D.hpp>      // 2D transformations
#include <matrixlib/matrix3D.hpp>      // 3D/4D transformations

using namespace matrixlib;

Vec<float, 3> v(1.0f, 2.0f, 3.0f);
Quaternion<float> q(Vec<float, 3>(0, 0, 1), 1.57f);

// Check library version
if (version_at_least(1, 0, 0)) {
    // Use library features
}
```

## Usage Examples

### Vector Operations

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

Vec<float, 3> a(1.0f, 0.0f, 0.0f);
Vec<float, 3> b(0.0f, 1.0f, 0.0f);

// Basic operations
Vec<float, 3> sum = a + b;              // (1, 1, 0)
float dot = a.dot(b);                   // 0.0
Vec<float, 3> cross = a.cross(b);       // (0, 0, 1)

// Advanced operations
Vec<float, 3> normalized = a.normalized();
float angle = a.angle(b);               // π/2 radians
Vec<float, 3> proj = a.project(b);

// Static factory methods
Vec<float, 3> zero = Vec<float, 3>::zero();
Vec<float, 3> unit_x = Vec<float, 3>::unit_x();

// Swizzling
Vec<float, 2> xy = a.xy();              // (1, 0)
Vec<float, 3> xyz = sum.xyz();          // (1, 1, 0)

// Interpolation
Vec<float, 3> mid = a.lerp(b, 0.5f);
Vec<float, 3> smooth = a.cubic_hermite(b, Vec<float,3>::zero(), Vec<float,3>::zero(), 0.5f);

// Clamping
Vec<float, 3> clamped = sum.clamped(-1.0f, 1.0f);
Vec<float, 3> saturated = sum.saturated();  // [0, 1]
```

### Matrix Operations

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

// 3x3 rotation matrix
SquareMat<float, 3> R = SquareMat<float, 3>::rotation_z(1.57f);

// Matrix operations
Mat<float, 3, 3> RT = R.transpose();
float det = R.determinant();
SquareMat<float, 3> inv = R.inverse();

// Transform vector
Vec<float, 3> v(1.0f, 0.0f, 0.0f);
Vec<float, 3> rotated = R * v;
```

### Quaternion Rotations

```cpp
#include <matrixlib/quaternion.hpp>

using namespace matrixlib;

// Create rotation quaternion
Vec<float, 3> axis(0.0f, 0.0f, 1.0f);
Quaternion<float> q(axis, 1.57f);       // 90° around Z

// Rotate vector
Vec<float, 3> point(1.0f, 0.0f, 0.0f);
Vec<float, 3> rotated = q.rotate(point); // (0, 1, 0)

// Quaternion interpolation
Quaternion<float> q1 = Quaternion<float>::identity();
Quaternion<float> q2(axis, 3.14f);
Quaternion<float> mid = q1.slerp(q2, 0.5f);

// Euler angle conversion
Vec<float, 3> euler = q.to_euler();     // (roll, pitch, yaw)
Quaternion<float> q3 = Quaternion<float>::from_euler(0.1f, 0.2f, 0.3f);
float roll = q.roll();
float pitch = q.pitch();
float yaw = q.yaw();
```

### Mathematical Utilities

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

// Constants
float pi = constants::pi<float>;
float deg90 = deg_to_rad(90.0f);        // Convert to radians
float rad90 = rad_to_deg(1.57f);        // Convert to degrees

// Angle utilities
float wrapped = wrap_pi(3.5f);          // Wrap to [-π, π]
float wrapped2 = wrap_two_pi(3.5f);     // Wrap to [0, 2π]
float shortest = angle_distance(0.1f, 6.2f);  // Shortest angular distance

// Clamping
float val = clamp(1.5f, 0.0f, 1.0f);    // 1.0
float sat = saturate(1.5f);             // 1.0 (clamp to [0,1])

// Homogeneous coordinates
Vec<float, 3> v3(1.0f, 2.0f, 3.0f);
Vec<float, 4> v4 = v3.to_homogeneous();  // (1, 2, 3, 1)
Vec<float, 3> back = v4.from_homogeneous();  // Perspective division
```

### Compile-Time Rotations

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

// C++11-C++23: Special angles (0°, 90°, 180°, 270°) at compile time
constexpr auto R90z = SquareMat<float, 3>::rotation_z_deg<90>();
constexpr auto R180x = SquareMat<float, 3>::rotation_x_deg<180>();
constexpr auto R270y = SquareMat<float, 3>::rotation_y_deg<270>();

// Perfect for fixed sensor orientations - zero runtime cost!
constexpr Vec<float, 3> sensor(1.0f, 0.0f, 0.0f);
constexpr Vec<float, 3> body = R90z * sensor;  // Computed at compile time

// C++26: ANY angle at compile time (constexpr sin/cos)
#if __cplusplus >= 202600L
constexpr auto R = SquareMat<float, 3>::rotation_z(1.234f);  // Any angle!
#endif

// See docs/Cpp_Standard_Optimizations.md for details and workarounds
```

## Examples

The `examples/` directory contains practical demonstrations:

### Basic Usage (`basic_usage.cpp`)
Fundamental operations: vectors, matrices, quaternions, transformations.

### Sensor Fusion (`sensor_fusion.cpp`)
IMU sensor fusion using complementary filter with quaternion orientation estimation:
```cpp
ComplementaryFilter filter(0.98f);
filter.update(gyro, accel, dt);
Vec3f euler = filter.get_euler_angles();
```

### Robotics Kinematics (`robotics_kinematics.cpp`)
2-link planar robot arm with forward/inverse kinematics, Jacobian, and trajectory planning.

### Graphics Pipeline (`graphics_pipeline.cpp`)
3D graphics transformations: model-view-projection matrices, viewport transforms, frustum culling.

### Compile-Time Rotations (`constexpr_rotations.cpp`)
Demonstrates constexpr factory methods and compile-time matrix operations.

**Build examples:**
```bash
cmake .. -DMATRIX_LINEAR_BUILD_EXAMPLES=ON
make
./examples/sensor_fusion
```

## Performance Benchmarks

The `benchmarks/` directory provides Google Benchmark-based performance tests:

- **Matrix Operations**: Multiplication, transpose, determinant, inverse
- **SIMD Comparisons**: Vec3/Vec4/Quaternion operations with/without SIMD
- **Constexpr vs Runtime**: Compile-time initialization performance
- **Vector Operations**: Dot, cross, normalize, interpolation, batch processing

**Run benchmarks:**
```bash
cmake .. -DMATRIX_LINEAR_BUILD_BENCHMARKS=ON
make
./benchmarks/bench_matrix_multiply
./benchmarks/bench_simd --benchmark_filter=Vec3.*
```

See [benchmarks/README.md](benchmarks/README.md) for detailed results and optimization tips.

For detailed performance comparisons with other libraries, see [PERFORMANCE.md](PERFORMANCE.md).

## Documentation

- **[Quick Reference](QUICK_REFERENCE.md)** - Compact API cheat sheet
- **[API Documentation](docs/API_Documentation.md)** - Complete API reference
- **[Migration Guide](MIGRATION.md)** - Switch from Eigen, GLM, or custom code
- **[Performance Comparison](PERFORMANCE.md)** - Benchmarks vs. alternatives
- **[C++ Optimizations](docs/Cpp_Standard_Optimizations.md)** - C++11-26 feature usage
- **[Doxygen](docs/doxygen/html/index.html)** - Generated API docs (run `doxygen`)

## SIMD Optimizations

### ARM NEON (Cortex-A, ARM64, Apple Silicon)

Enable NEON optimizations for high-performance ARM processors:

```cmake
# In CMakeLists.txt
cmake -DMATRIXLIB_ENABLE_NEON=ON ..
```

Optimized operations for `Vec<float, 2/3/4>` and `Quaternion<float>`:
- Addition/subtraction: `vaddq_f32`, `vsubq_f32`
- Scalar multiplication: `vmulq_f32`
- Dot product: `vmulq_f32` + horizontal reduction
- Negation: `vnegq_f32`

### CMSIS-DSP (Cortex-M)

Enable hardware acceleration on ARM Cortex-M processors:

```cmake
# In CMakeLists.txt
target_compile_definitions(your_target PRIVATE CONFIG_MATRIXLIB_CMSIS)
target_link_libraries(your_target PRIVATE CMSIS::DSP)
```

Optimized CMSIS-DSP functions for `float` types:
- Vector operations: `arm_add_f32`, `arm_sub_f32`, `arm_dot_prod_f32`
- Matrix operations: `arm_mat_mult_f32`, `arm_mat_inverse_f32`
- Quaternion operations: `arm_quaternion_product_f32`

**Performance**: Three-tier optimization strategy (NEON → CMSIS → Generic) provides optimal performance across the entire ARM ecosystem.

## Documentation

- [API Documentation](docs/API_Documentation.md) - Complete API reference with examples
- [C++ Standard Optimizations](docs/Cpp_Standard_Optimizations.md) - C++14/17/20 features and performance
- [Sanitizer Safety](docs/Sanitizer_Safety.md) - UB prevention, memory safety, and sanitizer testing
- [Examples](examples/) - Code examples and usage patterns

## Library Components

All classes are in the `matrixlib` namespace:

### Header Organization

The library uses a modular header structure for flexibility:

- **matrixlib.hpp** - Main convenience header that includes all components
- **compiler_features.hpp** - C++ standard feature detection (C++11-C++26 compatibility macros)
- **constants.hpp** - Mathematical constants (pi, e, sqrt2, etc.) - independent, can be used standalone
- **vector.hpp** - Generic `Vec<T,N>` template with all vector operations
- **vec2D.hpp** - 2D vector type aliases (`Vec2f`, `Vec2d`, etc.)
- **vec3D.hpp** - 3D vector type aliases (`Vec3f`, `Vec3d`, etc.)
- **matrix.hpp** - Generic `Mat<T,R,C>` and `SquareMat<T,N>` templates
- **matrix2D.hpp** - 2D transformation matrices (rotation, scale)
- **matrix3D.hpp** - 3D/4D transformation matrices (rotations, look-at, translation)
- **quaternion.hpp** - Quaternion rotations
- **version.hpp** - Runtime version API

### Vec<T, N>
- N-dimensional vectors with type T
- Common indices: `X=0, Y=1, Z=2, W=3`
- Optimized for 2D, 3D, and 4D operations
- NEON-accelerated for `Vec<float, 2/3/4>`

### Mat<T, R, C>
- R×C matrices with row-major storage
- Generic matrix operations
- CMSIS-DSP optimized for `float` types

### SquareMat<T, N>
- Square matrices with additional operations
- Rotation, scale, translation matrices
- Determinant, inverse, eigenvalues

### Quaternion<T>
- Efficient 3D rotation representation
- Memory layout optimized for Vec<T,3> operations
- SLERP interpolation for smooth animations
- NEON-accelerated for `Quaternion<float>`

### Version API
- Runtime version checking: `get_version_string()`, `get_version_number()`
- Compatibility checking: `version_at_least(major, minor, patch)`

## Requirements

- **C++ Standard**: C++11 or later
- **Dependencies**: None (NEON/CMSIS-DSP optional for ARM targets)
- **Compiler Support**: GCC, Clang, MSVC, ARM Compiler 6, IAR Embedded Workbench for ARM
- **Tested Platforms**: 
  - ARM Cortex-A (with NEON)
  - ARM64/AArch64 (with NEON)
  - Apple Silicon M1/M2/M3 (with NEON)
  - ARM Cortex-M0/M0+/M3/M4/M7/M33 (with CMSIS-DSP)
  - x86/x64 desktop platforms
  - Zephyr RTOS 3.0+

## Performance

All types are POD (Plain Old Data) with the following characteristics:
- Zero runtime overhead for abstractions
- Inline-friendly for compiler optimization
- Cache-friendly contiguous memory layout
- 16-byte alignment for SIMD operations
- No virtual functions or dynamic dispatch

## Building Examples and Tests

```bash
mkdir build && cd build
cmake ..
cmake --build .

# Run tests
ctest --output-on-failure
```

## Integration with Zephyr

MatrixLib integrates seamlessly with Zephyr RTOS:

```c
// prj.conf
CONFIG_MATRIX_LINEAR=y
CONFIG_MATRIXLIB_NEON=y   # Optional: Enable NEON (Cortex-A/ARM64)
CONFIG_MATRIXLIB_CMSIS=y  # Optional: Enable CMSIS-DSP (Cortex-M)
CONFIG_MATRIXLIB_MVE=y    # Optional: Enable MVE/Helium (Cortex-M55/M85)

// CMakeLists.txt
target_link_libraries(app PUBLIC matrixlib)
```

## Testing

### Unit Tests

MatrixLib includes comprehensive test suites:

**Google Test Suite** (for desktop/CI):
```bash
cmake .. -DMATRIX_LINEAR_BUILD_TESTS=ON
make
./tests/google/matrixlib_gtests
```

Tests include:
- Compile-time constexpr validation (C++14+)
- Vector/matrix/quaternion operations
- Edge cases and numerical stability
- Type conversions and utilities
- SIMD correctness

**Zephyr Test Suite** (for embedded targets):
```bash
west build -b nrf52840dk_nrf52840 tests/zephyr
west flash
```

### Sanitizers

MatrixLib is sanitizer-clean:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
make && ./tests/google/matrixlib_gtests
```

Passes: AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), MemorySanitizer (MSan)

## Using with IAR Embedded Workbench

MatrixLib is fully compatible with IAR Embedded Workbench for ARM:

**Quick Setup:**
1. Add `matrixlib/include` to your project's include paths
2. Enable C++11: Project Options → C/C++ Compiler → Language → C++ → C++11
3. For NEON optimization: Project Options → C/C++ Compiler → Code → FPU: VFPv4_sp
4. Add preprocessor define: `CONFIG_MATRIXLIB_NEON`

**Example:**
```cpp
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>

using namespace matrixlib;

void compute_rotation() {
    Vec<float, 3> axis(0, 0, 1);
    Quaternion<float> q(axis, 1.57f);
    Vec<float, 3> point(1, 0, 0);
    Vec<float, 3> result = q.rotate(point);
}
```

See [QUICKSTART.md](QUICKSTART.md) for detailed IAR integration instructions.

## AI-Assisted Development

This library was developed with assistance from **GitHub Copilot** using the **Claude Sonnet 4.5** model. The AI helped with:
- Code generation and optimization
- Documentation and comments
- SIMD optimization implementation (NEON/CMSIS-DSP)
- Test coverage and examples

See [.ai-generation-prompt.md](.ai-generation-prompt.md) for the regeneration prompt.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Authors

- Created for embedded systems and robotics applications
- Optimized for ARM Cortex-M microcontrollers
- Designed with real-time constraints in mind

## Version

Current Version: 1.0.0
