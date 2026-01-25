# MatrixLib Sanitizer Safety Documentation

## Overview

MatrixLib has been hardened against undefined behavior (UB) and memory safety issues to ensure clean execution under AddressSanitizer (ASan), UndefinedBehaviorSanitizer (UBSan), and MemorySanitizer (MSan).

## Addressed Issues

### 1. Type-Punning and Strict Aliasing Violations

**Problem**: Using `reinterpret_cast` to create references/pointers to different types violates strict aliasing rules and causes undefined behavior.

**Solution**: Replaced all unsafe type-punning with safe alternatives:

#### MATRIX_BIT_CAST Macro

Location: [compiler_features.hpp](../include/matrixlib/compiler_features.hpp)

```cpp
// C++20 and later: use std::bit_cast
#if __cplusplus >= 202002L
    #define MATRIX_BIT_CAST(T, val) std::bit_cast<T>(val)
#else
    // C++11 fallback: safe memcpy-based cast
    namespace matrixlib { namespace detail {
        template<typename To, typename From>
        inline To bit_cast_memcpy(const From& src) noexcept
        {
            static_assert(sizeof(To) == sizeof(From), "Size mismatch");
            To dst;
            std::memcpy(&dst, &src, sizeof(To));
            return dst;
        }
    }}
    #define MATRIX_BIT_CAST(T, val) ::matrixlib::detail::bit_cast_memcpy<T>(val)
#endif
```

**Why it's safe**:
- C++20: `std::bit_cast` is specifically designed for type conversion
- C++11: `std::memcpy` is explicitly allowed by the standard for type conversion
- Both approaches avoid creating invalid pointer aliases

#### Quaternion vec() Accessor

Location: [quaternion.hpp](../include/matrixlib/quaternion.hpp)

**Before** (Unsafe):
```cpp
Vec<T,3>& vec() {
    return *reinterpret_cast<Vec<T,3>*>(&data[X]);  // UB!
}
```

**After** (Safe):
```cpp
Vec<T,3> vec() const noexcept {
    return Vec<T,3>(data[X], data[Y], data[Z]);  // Return by value
}

void set_vec(const Vec<T,3>& v) noexcept {
    data[X] = v[0];
    data[Y] = v[1];
    data[Z] = v[2];
}
```

**Impact**: Returns by value instead of reference. Compilers optimize this to zero-cost with Return Value Optimization (RVO) and copy elision.

**Updated call sites**:
- `norm()`: Changed from `const Vec<T,3>& imag = *reinterpret_cast<...>` to `const Vec<T,3> imag = vec()`
- `dot()`: Changed both quaternion imaginary parts to use `vec()`
- `rotate()`: Changed to use `vec()` instead of direct reinterpret_cast
- `operator+=`: Changed from `vec() += other.vec()` to direct data[] access
- `operator-=`: Changed from `vec() -= other.vec()` to direct data[] access

### 2. SIMD Intrinsics (Safe Usage)

SIMD code uses `reinterpret_cast<float*>` for intrinsic functions, which is **safe** when:

1. **Type-guarded**: All SIMD paths check `std::is_same<T, float>::value`
2. **Memory layout compatible**: `float[N]` and `Vec<float, N>::data` have identical layout
3. **Standard practice**: SIMD intrinsics are designed to work with float pointers

Example (safe):
```cpp
#ifdef CONFIG_MATRIXLIB_NEON
if (std::is_same<T, float>::value && N == 4)
{
    float32x4_t a = vld1q_f32(reinterpret_cast<const float*>(data));
    float32x4_t b = vld1q_f32(reinterpret_cast<const float*>(other.data));
    float32x4_t r = vaddq_f32(a, b);
    vst1q_f32(reinterpret_cast<float*>(result.data), r);
    return result;
}
#endif
```

### 3. POD Trait Deprecation

**Problem**: `std::is_pod` was deprecated in C++20

**Solution**: Replaced with `std::is_trivially_copyable`

```cpp
// Before (C++11-C++17, deprecated in C++20)
static_assert(std::is_pod<Vec2<float>>::value, "Must be POD");

// After (C++11+, future-proof)
static_assert(std::is_trivially_copyable<Vec2<float>>::value, "Must be trivially copyable");
```

**Files updated**:
- [vec2D.hpp](../include/matrixlib/vec2D.hpp)
- [vec3D.hpp](../include/matrixlib/vec3D.hpp)

### 4. Singular Matrix Handling

**Problem**: `inverse()` methods don't check for singular matrices (determinant ≈ 0), leading to NaN/Inf results

**Solution**: Added documentation warnings and rely on existing pivot checking in Gauss-Jordan elimination

Example documentation:
```cpp
/// @warning For singular or near-singular matrices (det ≈ 0), the result
///          will contain NaN or Inf values. Check determinant() before calling.
/// @note Uses Gauss-Jordan elimination with partial pivoting for numerical stability
```

**User responsibility**: Check `determinant()` before calling `inverse()`:
```cpp
SquareMat<float, 3> m = ...;
if (std::abs(m.determinant()) > 1e-6f) {
    auto inv = m.inverse();
    // Safe to use inv
} else {
    // Matrix is singular
}
```

## Testing for UB

### Edge Case Test Suite

Location: [tests/google/test_edge_cases.cpp](../tests/google/test_edge_cases.cpp)

Comprehensive tests designed to trigger sanitizers if UB exists:

#### Singular Matrix Tests
- Zero determinant matrices
- Zero row/column matrices
- Near-singular matrices (small but non-zero determinant)
- Identity matrix inversion

#### Quaternion Aliasing Tests
- `vec()` returns copy (no reference aliasing)
- `set_vec()` proper modification
- Operations using `vec()` internally (norm, dot, rotate)
- Compound assignment operators

#### NaN and Infinity Tests
- NaN propagation in vectors
- Infinity handling
- Zero vector normalization
- Safe vs unsafe normalization

#### Numerical Robustness
- Cross product with parallel vectors
- Cross product with opposite vectors
- Matrix rank computation
- SIMD alignment verification

### Running with Sanitizers

#### UndefinedBehaviorSanitizer (UBSan)

```bash
cd tests/google
mkdir build && cd build
cmake -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer" ..
make
./matrixlib_gtests
```

Expected: No UB errors reported

#### AddressSanitizer (ASan)

```bash
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" ..
make
./matrixlib_gtests
```

Expected: No memory errors reported

#### MemorySanitizer (MSan)

```bash
cmake -DCMAKE_CXX_FLAGS="-fsanitize=memory -fno-omit-frame-pointer" ..
make
./matrixlib_gtests
```

Expected: No uninitialized memory reads

#### Combined Sanitizers

```bash
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" ..
```

### Static Analysis

#### clang-tidy

```bash
clang-tidy include/matrixlib/*.hpp -- -I./include -std=c++11
```

Checks for:
- Type safety violations
- Undefined behavior patterns
- Modernization opportunities

#### cppcheck

```bash
cppcheck --enable=all --std=c++11 include/matrixlib/
```

## Best Practices

### 1. Type Conversions

**✅ Do**: Use MATRIX_BIT_CAST for type conversions
```cpp
float f = 3.14f;
auto i = MATRIX_BIT_CAST(std::uint32_t, f);
```

**❌ Don't**: Use reinterpret_cast for type conversion
```cpp
float f = 3.14f;
auto i = *reinterpret_cast<std::uint32_t*>(&f);  // UB!
```

### 2. Vector/Quaternion Component Access

**✅ Do**: Return by value when creating new vectors
```cpp
Vec<T,3> get_position() const {
    return Vec<T,3>(x, y, z);
}
```

**❌ Don't**: Return references to reinterpreted data
```cpp
Vec<T,3>& get_position() {
    return *reinterpret_cast<Vec<T,3>*>(data);  // UB!
}
```

### 3. SIMD Operations

**✅ Do**: Guard SIMD casts with type checks
```cpp
if (std::is_same<T, float>::value) {
    float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(data));
}
```

**❌ Don't**: Use SIMD casts without type guards
```cpp
float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(data));  // UB if T != float!
```

### 4. Matrix Operations

**✅ Do**: Check determinant before inversion
```cpp
if (std::abs(m.determinant()) > epsilon) {
    auto inv = m.inverse();
}
```

**❌ Don't**: Assume inverse always succeeds
```cpp
auto inv = m.inverse();  // May contain NaN/Inf!
```

## Performance Impact

All safety improvements have **zero runtime cost** in optimized builds:

- **MATRIX_BIT_CAST**: Compiles to identical assembly as `reinterpret_cast`
- **vec() by value**: RVO eliminates temporary copies
- **memcpy-based casts**: Optimized to direct load/store by compilers

Verified with:
```bash
g++ -O3 -S example.cpp
# Check assembly output
```

## Compiler Support

| Feature | C++11 | C++14 | C++17 | C++20 | C++23 | C++26 |
|---------|-------|-------|-------|-------|-------|-------|
| MATRIX_BIT_CAST | ✅ (memcpy) | ✅ (memcpy) | ✅ (memcpy) | ✅ (std::bit_cast) | ✅ | ✅ |
| is_trivially_copyable | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Quaternion::vec() | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Edge case tests | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Continuous Integration

Add sanitizer checks to CI pipeline:

```yaml
name: Sanitizers

on: [push, pull_request]

jobs:
  ubsan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build with UBSan
        run: |
          cd tests/google
          mkdir build && cd build
          cmake -DCMAKE_CXX_FLAGS="-fsanitize=undefined" ..
          make
          ./matrixlib_gtests
          
  asan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build with ASan
        run: |
          cd tests/google
          mkdir build && cd build
          cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
          make
          ASAN_OPTIONS=detect_leaks=1 ./matrixlib_gtests
```

## References

- [C++ Core Guidelines: Type safety](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#SS-type)
- [LLVM Sanitizers Documentation](https://clang.llvm.org/docs/UsersManual.html#controlling-code-generation)
- [std::bit_cast documentation](https://en.cppreference.com/w/cpp/numeric/bit_cast)
- [Strict aliasing rules explained](https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8)

## Summary

MatrixLib is now **sanitizer-clean** and follows modern C++ safety practices:

✅ No undefined behavior from type-punning  
✅ No strict aliasing violations  
✅ Safe SIMD usage with type guards  
✅ Comprehensive edge case testing  
✅ Zero performance overhead  
✅ C++11 compatible  
✅ Future-proof (C++26 ready)
