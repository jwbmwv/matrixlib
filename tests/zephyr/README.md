# MatrixLib Zephyr Test Suite

This directory contains the Zephyr RTOS test suite for MatrixLib using the ztest framework.

## Test Structure

- **src/test_main.cpp**: Main entry point and test suite registration
- **src/test_vec.cpp**: Tests for `Vec<T,N>` class
- **src/test_mat.cpp**: Tests for `Mat<T,R,C>` and `SquareMat<T,N>` classes
- **src/test_quat.cpp**: Tests for `Quaternion<T>` class

## Building and Running Tests

### Prerequisites

1. Install Zephyr SDK and set up Zephyr development environment
2. Ensure MatrixLib is available as a Zephyr module

### Running on QEMU (Cortex-M3)

```bash
cd tests/zephyr
west build -p auto -b qemu_cortex_m3
west build -t run
```

### Running on Native POSIX

```bash
cd tests/zephyr
west build -p auto -b native_posix
west build -t run
```

### Running on Real Hardware

For example, on Nordic nRF52840 DK:

```bash
cd tests/zephyr
west build -p auto -b nrf52840dk_nrf52840
west flash
```

## Test Coverage

The test suite covers:

### Vector Operations
- Construction and initialization
- Arithmetic operations (+, -, *, /)
- Dot product and cross product
- Magnitude and normalization
- Angle calculation
- Projection and rejection
- Signed angles
- Component accessors (x, y, z, w)
- Compound assignment operators

### Matrix Operations
- Construction and identity matrix
- Matrix addition and multiplication
- Matrix-vector multiplication
- Transpose
- Determinant (2×2, 3×3, 4×4)
- Trace
- Inverse
- 2D and 3D rotations (X, Y, Z axes)
- Arbitrary axis rotation
- Rotation from two vectors
- Look-at matrix
- Euler angle extraction

### Quaternion Operations
- Construction (identity, axis-angle, from matrix)
- Conjugate and inverse
- Norm and normalization
- Quaternion multiplication
- Vector rotation
- Conversion to/from rotation matrix
- SLERP interpolation
- Arithmetic operations
- Component accessors
- Vector accessor for imaginary part

## CMSIS-DSP Testing

To test with CMSIS-DSP optimizations enabled:

1. Uncomment in `prj.conf`:
   ```
   CONFIG_MATRIX_LINEAR_CMSIS=y
   ```

2. Build for an ARM Cortex-M platform:
   ```bash
   west build -p auto -b qemu_cortex_m3
   west build -t run
   ```

## Expected Output

A successful test run should display:

```
*** Booting Zephyr OS build ... ***
MatrixLib Test Suite Starting...
Running TESTSUITE matrixlib_vec
===================================================================
START - test_vec_construction
 PASS - test_vec_construction in 0.001 seconds
START - test_vec_addition
 PASS - test_vec_addition in 0.001 seconds
...
===================================================================
PROJECT EXECUTION SUCCESSFUL
```

## Continuous Integration

The `testcase.yaml` file configures the tests for Zephyr's CI system. Tests will run on:
- QEMU Cortex-M3
- QEMU Cortex-M0
- Native POSIX simulation

## Adding New Tests

To add new test cases:

1. Add test functions using `ZTEST(suite_name, test_name)` macro
2. Use zassert macros for assertions:
   - `zassert_true(condition, message)`
   - `zassert_false(condition, message)`
   - `zassert_equal(a, b, message)`
   - `zassert_not_equal(a, b, message)`

Example:
```cpp
ZTEST(matrixlib_vec, test_new_feature)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    float result = v.new_operation();
    
    zassert_true(float_eq(result, 6.0f), "Result should be 6.0");
}
```

## Troubleshooting

### Build Errors

If you encounter build errors, ensure:
- Zephyr SDK is properly installed
- MatrixLib is correctly registered as a Zephyr module
- `west.yml` includes the correct path to MatrixLib

### Test Failures

For floating-point comparison failures:
- Check `FLOAT_EPSILON` value (default: 0.0001)
- Verify expected values account for floating-point precision
- Consider platform-specific floating-point behavior

### CMSIS-DSP Issues

If CMSIS-DSP tests fail:
- Verify CMSIS-DSP is available in Zephyr
- Check that the platform supports CMSIS-DSP
- Ensure `CONFIG_CMSIS_DSP=y` is set in Zephyr configuration
