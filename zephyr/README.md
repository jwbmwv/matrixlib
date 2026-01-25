# Sample Zephyr Project Configuration for MatrixLib

This directory contains Zephyr RTOS integration files for MatrixLib.

## Files

- **module.yml**: Zephyr module metadata
- **CMakeLists.txt**: Zephyr-specific build configuration
- **Kconfig**: Configuration options for MatrixLib

## Using MatrixLib in Zephyr

### 1. Add as a Zephyr Module

In your application's `west.yml`:

```yaml
manifest:
  projects:
    - name: matrixlib
      url: https://github.com/yourusername/matrixlib.git
      revision: main
      path: modules/lib/matrixlib
```

Then run:
```bash
west update
```

### 2. Enable in prj.conf

```conf
# Enable MatrixLib
CONFIG_MATRIXLIB=y

# Optional: Enable ARM NEON optimizations (Cortex-A/ARM64 only)
CONFIG_MATRIXLIB_NEON=y

# Optional: Enable CMSIS-DSP optimizations (Cortex-M only)
CONFIG_MATRIXLIB_CMSIS=y
CONFIG_CMSIS_DSP=y

# Optional: Enable ARM MVE (Helium) optimizations (Cortex-M55/M85)
CONFIG_MATRIXLIB_MVE=y
```

### 3. Use in Your Application

```c
// In CMakeLists.txt
target_link_libraries(app PUBLIC matrixlib)

// In your source code
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <matrixlib/version.hpp>

using namespace matrixlib;

void main(void) {
    printk("MatrixLib %s\n", get_version_string().c_str());
    
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    float len = v.length();
    printk("Vector length: %f\n", len);
}
```

## Configuration Options

### CONFIG_MATRIXLIB
Enable the MatrixLib library. This adds the include path and makes the library available to your application.

### CONFIG_MATRIXLIB_NEON
Enable ARM NEON SIMD optimizations. Requires:
- ARM Cortex-A processor or ARM64/AArch64
- Automatically adds `-mfpu=neon` compiler flag

When enabled, vector and quaternion operations for `Vec<float, 2/3/4>` and `Quaternion<float>` use ARM NEON intrinsics for 2-4x performance improvement.

### CONFIG_MATRIXLIB_CMSIS
Enable CMSIS-DSP hardware acceleration. Requires:
- ARM Cortex-M processor  
- CONFIG_CMSIS_DSP=y

When enabled, vector, matrix, and quaternion operations automatically use optimized CMSIS-DSP functions for `float` types.

### CONFIG_MATRIXLIB_MVE
Enable ARM MVE (Helium) SIMD optimizations. Requires:
- Cortex-M processor with MVE (e.g., Cortex-M55/M85)
- Toolchain support for MVE intrinsics

When enabled, MatrixLib can compile MVE-specific paths where available. (MVE kernels are currently reserved for future optimization work.)

## Example Application

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(matrixlib_demo)

target_sources(app PRIVATE src/main.cpp)
target_link_libraries(app PUBLIC matrixlib)
```

```c++
// src/main.cpp
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <matrixlib/matrixlib.hpp>
#include <matrixlib/quaternion.hpp>
#include <matrixlib/version.hpp>

using namespace matrixlib;

void main(void)
{
    printk("MatrixLib %s Demo\n", get_version_string().c_str());
    
    // Vector operations
    Vec<float, 3> a(1.0f, 0.0f, 0.0f);
    Vec<float, 3> b(0.0f, 1.0f, 0.0f);
    Vec<float, 3> c = a.cross(b);
    
    printk("Cross product: (%f, %f, %f)\n", c[0], c[1], c[2]);
    
    // Quaternion rotation
    Quaternion<float> q(Vec<float, 3>(0, 0, 1), 1.57f);
    Vec<float, 3> rotated = q.rotate(a);
    
    printk("Rotated: (%f, %f, %f)\n", rotated[0], rotated[1], rotated[2]);
}
```

## Performance on Zephyr

MatrixLib is optimized for embedded systems:
- Zero dynamic allocation
- POD types compatible with Zephyr
- Optional CMSIS-DSP acceleration on ARM
- Minimal flash/RAM footprint

Typical footprint (ARM Cortex-M4F, -Os):
- Vec operations: ~100-500 bytes
- Matrix operations: ~500-2000 bytes
- Quaternion operations: ~300-1000 bytes

With CMSIS-DSP, operations can be 2-5× faster while using the same or less code space.
