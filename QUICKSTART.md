# Quick Start Guide

## Repository Structure

```
matrixlib/
├── README.md              # Main documentation
├── LICENSE                # MIT License with SPDX identifier
├── VERSION                # Version number (1.0.0)
├── CMakeLists.txt         # Main CMake configuration
├── .gitignore             # Git ignore rules
├── .ai-generation-prompt.md  # AI regeneration prompt
│
├── include/matrixlib/     # Public headers
│   ├── compiler_features.hpp  # C++ feature detection macros (MATRIX_CONSTEXPR, etc.)
│   ├── matrixlib.hpp    # Convenience header (includes all components)
│   ├── constants.hpp      # Mathematical constants (independent)
│   ├── vector.hpp         # Generic Vec<T,N> template
│   ├── vec2D.hpp          # 2D vector type aliases (Vec2f, Vec2d, etc.)
│   ├── vec3D.hpp          # 3D vector type aliases (Vec3f, Vec3d, etc.)
│   ├── matrix.hpp         # Generic Mat<T,R,C> and SquareMat<T,N>
│   ├── matrix2D.hpp       # 2D transformations (rotation, scale)
│   ├── matrix3D.hpp       # 3D/4D transformations (rotations, look-at, etc.)
│   ├── quaternion.hpp     # Quaternion class
│   └── version.hpp        # Version API
│
├── docs/                  # Documentation
│   └── API_Documentation.md
│
├── examples/              # Example programs
│   ├── basic_usage.cpp
│   └── CMakeLists.txt
│
├── tests/                 # Unit tests
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── google/            # Google Test suite
│   └── zephyr/            # Zephyr ztest suite
│
├── zephyr/               # Zephyr RTOS integration
│   ├── module.yml        # Zephyr module metadata
│   ├── CMakeLists.txt    # Zephyr build configuration
│   ├── Kconfig           # Configuration options (NEON/CMSIS)
│   └── README.md         # Zephyr usage guide
│
└── cmake/                # CMake modules
    └── matrixlib-config.cmake.in
```

## Usage Methods

### Method 1: Git Submodule (Generic Projects)

```bash
# Add submodule
git submodule add https://github.com/yourusername/matrixlib.git external/matrixlib

# In your CMakeLists.txt
add_subdirectory(external/matrixlib)
target_link_libraries(your_app PUBLIC matrixlib)
```

### Method 2: Zephyr Module

Add to `west.yml`:
```yaml
manifest:
  projects:
    - name: matrixlib
      url: https://github.com/yourusername/matrixlib.git
      path: modules/lib/matrixlib
```

Enable in `prj.conf`:
```
CONFIG_MATRIX_LINEAR=y
```

### Method 3: CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
  matrixlib
  GIT_REPOSITORY https://github.com/yourusername/matrixlib.git
  GIT_TAG main
)
FetchContent_MakeAvailable(matrixlib)
target_link_libraries(your_app PUBLIC matrixlib)
```

### Method 4: Direct Copy

Just copy `include/matrixlib/` to your project's include path.

### Method 5: IAR Embedded Workbench for ARM

#### Option A: Add to IAR Project Directly

1. In IAR Embedded Workbench, right-click your project → **Options**
2. Navigate to **C/C++ Compiler** → **Preprocessor** → **Additional include directories**
3. Add the path to `matrixlib/include`
4. In your source files:
   ```cpp
   // Option 1: Include everything
   #include <matrixlib/matrixlib.hpp>
   #include <matrixlib/quaternion.hpp>
   
   // Option 2: Modular includes (faster compilation)
   #include <matrixlib/vector.hpp>      // Base vector template
   #include <matrixlib/vec3D.hpp>       // 3D type aliases
   #include <matrixlib/matrix3D.hpp>    // 3D transformations
   
   using namespace matrixlib;
   ```

#### Option B: Use CMake with IAR

```bash
# Configure with IAR toolchain
cmake -G "Ninja" -DCMAKE_TOOLCHAIN_FILE=iar-toolchain.cmake ..
cmake --build .
```

Create `iar-toolchain.cmake`:
```cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Path to IAR installation
set(IAR_ARM_ROOT "C:/Program Files/IAR Systems/Embedded Workbench 9.x/arm")

set(CMAKE_C_COMPILER "${IAR_ARM_ROOT}/bin/iccarm.exe")
set(CMAKE_CXX_COMPILER "${IAR_ARM_ROOT}/bin/iccarm.exe")
set(CMAKE_ASM_COMPILER "${IAR_ARM_ROOT}/bin/iasmarm.exe")
set(CMAKE_AR "${IAR_ARM_ROOT}/bin/iarchive.exe")

set(CMAKE_C_COMPILER_ID IAR)
set(CMAKE_CXX_COMPILER_ID IAR)

# Prevent CMake from testing the compiler
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
```

#### IAR-Specific Configuration

**Compiler Settings:**
- **Language**: C++11 or later (Options → C/C++ Compiler → Language → C++ → C++11)
- **Optimization**: High for performance (Options → C/C++ Compiler → Optimizations)
- **NEON**: Enable VFPv4_sp for NEON support (Options → C/C++ Compiler → Code)

**Preprocessor Defines:**
```
CONFIG_MATRIXLIB_NEON    // For NEON optimization
```

## Building Examples

```bash
mkdir build && cd build
cmake -DMATRIX_LINEAR_BUILD_EXAMPLES=ON ..
cmake --build .
./examples/basic_usage
```

## Enabling SIMD Optimizations

### ARM NEON (Cortex-A, ARM64, Apple Silicon)

```cmake
# Option 1: CMake option
cmake -DMATRIXLIB_ENABLE_NEON=ON ..

# Option 2: In your CMakeLists.txt
target_compile_definitions(your_app PRIVATE CONFIG_MATRIXLIB_NEON)
```

### CMSIS-DSP (Cortex-M)

```cmake
# Option 1: CMake option
cmake -DMATRIXLIB_ENABLE_CMSIS=ON ..

# Option 2: In your CMakeLists.txt
target_compile_definitions(your_app PRIVATE CONFIG_MATRIXLIB_CMSIS)
target_link_libraries(your_app PRIVATE CMSIS::DSP)
```

### Zephyr Configuration

```ini
# In prj.conf
CONFIG_MATRIX_LINEAR=y
CONFIG_MATRIXLIB_NEON=y   # For Cortex-A/ARM64
CONFIG_MATRIXLIB_CMSIS=y  # For Cortex-M
CONFIG_MATRIXLIB_MVE=y    # For Cortex-M55/M85 (Helium)
```

## Testing

```bash
mkdir build && cd build
cmake -DMATRIX_LINEAR_BUILD_TESTS=ON ..
cmake --build .
ctest --output-on-failure
```

## First Program

```cpp
#include <matrixlib/matrixlib.hpp>   // Includes all vector/matrix components
#include <matrixlib/quaternion.hpp>
#include <matrixlib/version.hpp>
#include <iostream>

// Or use modular includes:
// #include <matrixlib/vector.hpp>
// #include <matrixlib/vec3D.hpp>
// #include <matrixlib/matrix3D.hpp>

using namespace matrixlib;

int main() {
    // Check library version
    std::cout << "MatrixLib " << get_version_string() << "\n";
    
    // Vector operations
    Vec<float, 3> v1(1.0f, 0.0f, 0.0f);
    Vec<float, 3> v2(0.0f, 1.0f, 0.0f);
    Vec<float, 3> cross = v1.cross(v2);
    
    std::cout << "Cross: (" << cross[0] << ", " 
              << cross[1] << ", " << cross[2] << ")\n";
    
    // Quaternion rotation
    Quaternion<float> q(Vec<float, 3>(0, 0, 1), 1.57f);
    Vec<float, 3> rotated = q.rotate(v1);
    
    std::cout << "Rotated: (" << rotated[0] << ", " 
              << rotated[1] << ", " << rotated[2] << ")\n";
    
    return 0;
}
```

## Documentation

- [README.md](README.md) - Overview and quick start
- [docs/API_Documentation.md](docs/API_Documentation.md) - Complete API reference
- [zephyr/README.md](zephyr/README.md) - Zephyr integration guide
- [tests/README.md](tests/README.md) - Testing guide

## Requirements

- C++11 or later
- CMake 3.13.1+ (for building)
- Optional: ARM NEON (for Cortex-A/ARM64 optimization)
- Optional: CMSIS-DSP (for Cortex-M optimization)
- Optional: Zephyr RTOS 3.0+ (for Zephyr integration)

## Version

Current version: 1.0.0 (see [VERSION](VERSION))

## License

MIT License - see [LICENSE](LICENSE)
