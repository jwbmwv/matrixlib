# IAR Embedded Workbench Integration Guide

This guide provides detailed instructions for using MatrixLib with IAR Embedded Workbench for ARM.

## Prerequisites

- IAR Embedded Workbench for ARM 8.x or later
- C++11 support enabled
- Target: ARM Cortex-M or Cortex-A processor

## Integration Methods

### Method 1: Direct Include (Recommended for IAR)

This is the simplest method for IAR projects.

#### Step 1: Add Include Path

1. Open your IAR project
2. Right-click project name → **Options** (Alt+F7)
3. Navigate to **C/C++ Compiler** → **Preprocessor**
4. In **Additional include directories**, add:
   ```
   $PROJ_DIR$\..\matrixlib\include
   ```
   (Adjust path based on where you placed MatrixLib)

#### Step 2: Enable C++11

1. In Project Options → **C/C++ Compiler** → **Language**
2. Set **C++** dialect to **C++11** or later
3. Check **Enable C++ exceptions** if needed (not required for MatrixLib)

#### Step 3: Configure Optimization (Optional)

1. In Project Options → **C/C++ Compiler** → **Optimizations**
2. Set **Level**: **High** (for best performance)
3. Enable **Speed** optimization

#### Step 4: Choose SIMD Optimization (Optional)

MatrixLib supports three optimization levels. Choose based on your target processor:

##### Option A: ARM Cortex-M with CMSIS-DSP (Recommended for M4/M7/M33)

1. Ensure CMSIS-DSP is available in your project:
   - Add CMSIS-DSP library files to your project, OR
   - Link against pre-built CMSIS-DSP library
2. In **C/C++ Compiler** → **Preprocessor** → **Defined symbols**, add:
   ```
   CONFIG_MATRIXLIB_CMSIS
   ```
3. Enable FPU if available (M4F/M7):
   - Project Options → **C/C++ Compiler** → **Code** → **FPU**: VFPv4_sp or VFPv5_d16

##### Option B: ARM Cortex-A with NEON (For Cortex-A series)

1. In Project Options → **C/C++ Compiler** → **Code**
2. Set **FPU**: **VFPv4_sp** (for NEON support)
3. In **C/C++ Compiler** → **Preprocessor** → **Defined symbols**, add:
   ```
   CONFIG_MATRIXLIB_NEON
   ```

##### Option C: No SIMD Optimization (Generic ARM)

No additional configuration needed. MatrixLib will use optimized C++ code without SIMD intrinsics.
- Works on all ARM processors
- Smaller code size
- Still highly optimized

#### Step 5: Use in Your Code

```cpp
// Option 1: Include everything
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <matrixlib/version.hpp>

// Option 2: Modular includes (faster compilation)
#include <matrixlib/vector.hpp>
#include <matrixlib/vec3D.hpp>
#include <matrixlib/matrix3D.hpp>
#include <matrixlib/quaternion.hpp>

using namespace matrixlib;

void example_function() {
    // Check library version
    const char* version = get_version_string().c_str();
    
    // Vector operations
    Vec<float, 3> a(1.0f, 0.0f, 0.0f);
    Vec<float, 3> b(0.0f, 1.0f, 0.0f);
    Vec<float, 3> cross = a.cross(b);
    
    // Quaternion rotation
    Quaternion<float> q(Vec<float, 3>(0, 0, 1), 1.57f);
    Vec<float, 3> rotated = q.rotate(a);
}
```

### Method 2: Using CMake with IAR

If your project uses CMake, you can use the IAR toolchain.

#### Step 1: Create IAR Toolchain File

Create `iar-arm-toolchain.cmake`:

```cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Adjust path to your IAR installation
set(IAR_ARM_ROOT "C:/Program Files/IAR Systems/Embedded Workbench 9.x/arm")

set(CMAKE_C_COMPILER "${IAR_ARM_ROOT}/bin/iccarm.exe")
set(CMAKE_CXX_COMPILER "${IAR_ARM_ROOT}/bin/iccarm.exe")
set(CMAKE_ASM_COMPILER "${IAR_ARM_ROOT}/bin/iasmarm.exe")
set(CMAKE_AR "${IAR_ARM_ROOT}/bin/iarchive.exe")
set(CMAKE_LINKER "${IAR_ARM_ROOT}/bin/ilinkarm.exe")

set(CMAKE_C_COMPILER_ID IAR)
set(CMAKE_CXX_COMPILER_ID IAR)

# Prevent CMake from testing the compiler
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_FORCED 1)

# C++11 support
set(CMAKE_CXX_FLAGS_INIT "--c++ --c++11")

# Common IAR flags
set(CMAKE_C_FLAGS_INIT "--silent")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} --silent")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG_INIT "-Ol --debug")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Ol --debug")

# Release flags
set(CMAKE_C_FLAGS_RELEASE_INIT "-Ohs --no_debug")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Ohs --no_debug")
```

#### Step 2: Configure with CMake

**For Cortex-M with CMSIS-DSP:**
```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=../iar-arm-toolchain.cmake \
      -DMATRIXLIB_ENABLE_CMSIS=ON \
      ..
cmake --build .
```

**For Cortex-A with NEON:**
```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=../iar-arm-toolchain.cmake \
      -DMATRIXLIB_ENABLE_NEON=ON \
      ..
cmake --build .
```

**For generic ARM (no SIMD):**
```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=../iar-arm-toolchain.cmake \
      ..
cmake --build .
```

Or use Unix Makefiles:
```bash
cmake -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=../iar-arm-toolchain.cmake ..
cmake --build .
```

#### Step 3: Import to IAR Workbench

1. Create new IAR project
2. Add generated files to your workspace
3. Configure build settings to match CMake configuration

## Compiler-Specific Considerations

### IAR Compiler Options

MatrixLib automatically detects IAR compiler and applies appropriate settings when using CMake. If configuring manually:

**Required:**
- `--c++` - Enable C++ mode
- `--c++11` - C++11 dialect

**Recommended:**
- `--no_exceptions` - If you don't use exceptions
- `-Ohs` - High speed optimization
- `--silent` - Reduce build output

### SIMD Optimization Configuration

#### ARM Cortex-M with CMSIS-DSP

**Target Processors:** Cortex-M0+, M3, M4, M4F, M7, M33, M55, M85

**Preprocessor Define:**
```
CONFIG_MATRIXLIB_CMSIS
```

**Required Files:**
- CMSIS-DSP library (link or include source)
- `arm_math.h` in include path

**FPU Settings (for M4F/M7):**
- Project Options → C/C++ Compiler → Code → FPU: VFPv4_sp (M4F) or VFPv5_d16 (M7)

**Optimized Operations:** 
- Vec operations: `arm_add_f32`, `arm_sub_f32`, `arm_dot_prod_f32`
- Matrix operations: `arm_mat_mult_f32`, `arm_mat_inverse_f32`
- Quaternion operations: `arm_quaternion_product_f32`

**Performance:** ~1.5-3x speedup for float operations on Cortex-M4F/M7

#### ARM Cortex-A with NEON

**Target Processors:** Cortex-A5, A7, A8, A9, A15, A53, A57, A72, etc.

**Compiler Options:**
```
--fpu=VFPv4_sp
```

**Preprocessor Define:**
```
CONFIG_MATRIXLIB_NEON
```

**Optimized Operations:** 
- Vec<float, 2/3/4>: Addition, subtraction, scalar multiply, dot product
- Quaternion<float>: All arithmetic operations, norm, dot product

**Performance:** ~2-4x speedup for float vector/quaternion operations

#### Generic ARM (No SIMD)

**No special configuration needed.**

**Advantages:**
- Works on all ARM processors (Cortex-M0, M0+, M3 without FPU, etc.)
- Smaller code size
- No external dependencies
- Still highly optimized C++ code

**When to use:**
- Cortex-M0/M0+/M3 without CMSIS-DSP
- Code size is critical
- Maximum portability needed

## Troubleshooting

### Issue: "Identifier 'Vec' is undefined"

**Solution:** Ensure include path is correct and you're using `using namespace matrixlib;` or `matrixlib::Vec<>`

### Issue: "C++11 features not available"

**Solution:** 
1. Check Project Options → C/C++ Compiler → Language → C++
2. Ensure C++11 or later is selected
3. Verify `--c++11` flag is in compiler command line

### Issue: NEON intrinsics not found

**Solution:**
1. Verify target processor supports NEON (Cortex-A series or ARMv8)
2. Check FPU setting: Project Options → C/C++ Compiler → Code → FPU
3. Ensure `CONFIG_MATRIXLIB_NEON` is defined

### Issue: Linker errors with CMSIS-DSP

**Solution:**
1. Add CMSIS-DSP library to linker input
2. Or add CMSIS source files to your project
3. Ensure `arm_math.h` is in include path

### Issue: "Multiple definitions" error

**Solution:** MatrixLib is header-only. Ensure you're not compiling the headers as separate translation units.

## Performance Tips

### Optimization Settings

For best performance with MatrixLib:

1. **Optimization Level**: High speed (`-Ohs`)
2. **Inline Expansion**: Aggressive
3. **Loop Unrolling**: Enabled
4. **Link-Time Optimization**: Enabled (if available)

**Configuration:**
- Project Options → C/C++ Compiler → Optimizations → Level: High, Speed
- Project Options → C/C++ Compiler → Optimizations → Enable transformations: All

### SIMD Usage by Processor

| Processor | Recommended SIMD | Performance Gain | Code Size |
|-----------|------------------|------------------|------------|
| Cortex-M0/M0+ | None | Baseline | Smallest |
| Cortex-M3 | None | Baseline | Small |
| Cortex-M4 | CMSIS-DSP | 1.5-2x | Medium |
| Cortex-M7 | CMSIS-DSP | 2-3x | Medium |
| Cortex-M33 | CMSIS-DSP | 1.5-2x | Medium |
| Cortex-M55/M85 | CMSIS-DSP + MVE* | 3-4x | Larger |
| Cortex-A (NEON) | NEON | 2-4x | Medium |

*MVE (Helium) support coming in future versions

### Choosing the Right Configuration

**Use CMSIS-DSP when:**
- Target is Cortex-M4/M4F, M7, M33, or later
- CMSIS-DSP library is available
- Float performance is critical
- You have 10-20KB flash for CMSIS library

**Use NEON when:**
- Target is Cortex-A series processor
- Working with large datasets
- Float vector/quaternion operations are frequent

**Use Generic (no SIMD) when:**
- Target is Cortex-M0/M0+/M3
- Code size is extremely limited
- Maximum portability required
- CMSIS-DSP not available

### Code Size Optimization

If code size is critical:

1. Use **Size optimization** instead of Speed (`-Ohz`)
2. Disable SIMD optimizations (don't define CONFIG_MATRIXLIB_NEON or CONFIG_MATRIXLIB_CMSIS)
3. Consider using `double` instead of `float` if not using SIMD (smaller code for some operations)

## Example IAR Project Structure

```
MyProject/
├── main.cpp
├── MyProject.ewp              # IAR project file
├── MyProject.eww              # IAR workspace file
├── settings/                  # IAR settings
└── external/
    └── matrixlib/
        └── include/
            └── matrixlib/
                ├── matrixlib.hpp
                ├── quaternion.hpp
                └── version.hpp
```

## Example Code

These examples work with all SIMD configurations (CMSIS-DSP, NEON, or generic).

### Simple Vector Math

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

void calculate_trajectory()
{
    // Works on all ARM processors
    Vec<float, 3> velocity(10.0f, 0.0f, 5.0f);
    Vec<float, 3> gravity(0.0f, 0.0f, -9.81f);
    float dt = 0.01f;
    
    Vec<float, 3> position(0.0f, 0.0f, 0.0f);
    position += velocity * dt;  // Optimized with CMSIS or NEON if enabled
    velocity += gravity * dt;
}
```

### IMU Data Processing (Cortex-M4 with CMSIS-DSP)

```cpp
#include <matrixlib/quaternion.hpp>
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

// Optimized for Cortex-M4F/M7 with CMSIS-DSP
void process_imu_data(float* accel, float* gyro)
{
    // Convert accelerometer to gravity vector
    // CMSIS-DSP accelerates these operations automatically
    Vec<float, 3> gravity(accel[0], accel[1], accel[2]);
    gravity = gravity.normalized();
    
    // Create rotation from gyroscope
    Vec<float, 3> gyro_vec(gyro[0], gyro[1], gyro[2]);
    float angle = gyro_vec.length() * 0.01f; // dt = 10ms
    
    if (angle > 0.001f) {
        Vec<float, 3> axis = gyro_vec.normalized();
        Quaternion<float> rotation(axis, angle);
        
        // Rotate reference vector
        Vec<float, 3> reference(0, 0, 1);
        Vec<float, 3> rotated = rotation.rotate(reference);
    }
}
```

### Matrix Transformation (Works on all configurations)

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

void transform_points(const Vec<float, 3>* input, Vec<float, 3>* output, int count)
{
    // Create transformation matrix
    SquareMat<float, 3> rotation = SquareMat<float, 3>::rotation_z(1.57f);
    Vec<float, 3> scale_factors(2.0f, 2.0f, 2.0f);
    SquareMat<float, 3> scale_mat = SquareMat<float, 3>::scale(scale_factors);
    
    SquareMat<float, 3> transform = rotation * scale_mat;
    
    // Transform all points
    // Uses CMSIS/NEON if available, optimized C++ otherwise
    for (int i = 0; i < count; i++)
    {
        output[i] = transform * input[i];
    }
}
```

### Cortex-M0 Example (No SIMD)

```cpp
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

// Optimized for minimal code size on Cortex-M0/M0+
void simple_sensor_fusion() {
    Vec<float, 2> sensor1(1.0f, 2.0f);
    Vec<float, 2> sensor2(3.0f, 4.0f);
    
    // Even without SIMD, these operations are highly optimized
    Vec<float, 2> fused = (sensor1 + sensor2) * 0.5f;
    float magnitude = fused.length();
}
```

## Additional Resources

- [MatrixLib API Documentation](../docs/API_Documentation.md)
- [MatrixLib README](../README.md)
- [IAR Embedded Workbench User Guide](https://www.iar.com/support/user-guides/)
- [ARM NEON Programmer's Guide](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)

## Support

For IAR-specific issues with MatrixLib, please check:
1. This guide for common solutions
2. MatrixLib GitHub issues
3. IAR Technical Support for compiler-specific problems

---

**Last Updated**: January 25, 2026  
**MatrixLib Version**: 1.0.0  
**Tested with**: IAR Embedded Workbench for ARM 8.x, 9.x
