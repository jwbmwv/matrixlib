# MatrixLib Google Test Suite

This directory contains the Google Test (gtest) test suite for MatrixLib.

## Test Structure

- **test_constants.cpp**: Tests for `constants` namespace (mathematical constants)
- **test_vec.cpp**: Tests for `Vec<T,N>` class (26 test cases)
- **test_mat.cpp**: Tests for `Mat<T,R,C>` and `SquareMat<T,N>` classes (23 test cases)
- **test_quat.cpp**: Tests for `Quaternion<T>` class (22 test cases)
- **test_edge_cases.cpp**: Edge cases and numerical robustness (sanitizer-ready tests)

## Prerequisites

### Install Google Test

#### Ubuntu/Debian
```bash
sudo apt-get install libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

#### Fedora/RHEL
```bash
sudo dnf install gtest-devel
```

#### macOS (Homebrew)
```bash
brew install googletest
```

#### Windows (vcpkg)
```bash
vcpkg install gtest
```

#### Build from source
```bash
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake ..
make
sudo make install
```

## Building and Running Tests

### Using CMake

```bash
cd tests/google
mkdir build && cd build
cmake ..
make
./matrixlib_gtests
```

### With verbose output

```bash
./matrixlib_gtests --gtest_verbose
```

### Run specific tests

```bash
# Run all Vec tests
./matrixlib_gtests --gtest_filter=VecTest.*

# Run specific test
./matrixlib_gtests --gtest_filter=VecTest.DotProduct

# Run multiple test patterns
./matrixlib_gtests --gtest_filter=VecTest.*:MatTest.Rotation*
```

### Generate XML report

```bash
./matrixlib_gtests --gtest_output=xml:test_results.xml
```

## Test Coverage

### Constants (ConstantsTest)
- **Fundamental Constants**: pi, e, golden_ratio, sqrt2, sqrt3
- **Pi Derivatives**: two_pi, half_pi, quarter_pi
- **Natural Logarithms**: ln2, ln10
- **Conversion Factors**: deg_to_rad, rad_to_deg
- **Epsilon Values**: epsilon, epsilon_f, epsilon_d
- **Physical Constants**: gravity, speed_of_light
- **Type Safety**: Type flexibility and constexpr evaluation
- **Real-World Usage**: Practical calculation examples

### Vector Operations (VecTest)
- **Construction**: Default, parameterized, copy constructors
- **Arithmetic**: Addition, subtraction, scalar multiplication/division
- **Products**: Dot product, cross product, element-wise multiplication
- **Magnitude**: magnitude(), magnitude_squared(), normalization
- **Angles**: angle(), signed_angle()
- **Projections**: project(), reject()
- **Accessors**: x(), y(), z(), w()
- **Rotations**: 2D rotation, 3D axis-angle rotation
- **Operators**: Negation, compound assignment (+=, -=, *=, /=)
- **Utility**: distance()

### Matrix Operations (MatTest)
- **Construction**: Default, identity matrices
- **Arithmetic**: Addition, subtraction, scalar multiplication
- **Multiplication**: Matrix-matrix, matrix-vector
- **Transpose**: Non-square and square matrices
- **Determinant**: 2×2, 3×3, 4×4 matrices
- **Trace**: Sum of diagonal elements
- **Inverse**: 2×2, 3×3 matrices
- **Rotations**: 
  - 2D rotation
  - 3D rotations (X, Y, Z axes)
  - Arbitrary axis rotation
  - Rotation from two vectors
  - Look-at matrix
- **Euler Angles**: Extraction from rotation matrices
- **Operators**: Compound assignment (+=, -=, *=)

### Quaternion Operations (QuatTest)
- **Construction**: Identity, axis-angle, from rotation matrix
- **Components**: Accessors (w, x, y, z, vec)
- **Conjugate**: Quaternion conjugate
- **Norm**: Magnitude and normalization
- **Arithmetic**: Addition, subtraction, scalar multiplication
- **Multiplication**: Quaternion-quaternion product
- **Inverse**: Quaternion inverse
- **Rotation**: Vector rotation via quaternion
- **Conversion**: To/from rotation matrix (round-trip testing)
- **SLERP**: Spherical linear interpolation (t=0, 0.5, 1)
- **Composition**: Multiple rotation composition
- **Identity**: Identity quaternion behavior
- **Operators**: Compound assignment (+=, -=, *=)

### Edge Cases and Numerical Robustness (EdgeCasesTest)
- **Singular Matrices**: Zero determinant handling, NaN/Inf detection
- **Near-Singular Matrices**: Small determinants, numerical stability
- **Quaternion Safety**: vec() aliasing tests, set_vec() verification
- **NaN Handling**: NaN propagation in vector/matrix operations
- **Infinity Handling**: Infinite values in calculations
- **Zero Normalization**: safe_normalized() vs normalized()
- **Cross Product Edge Cases**: Parallel and opposite vectors
- **Matrix Rank**: Full rank and singular matrix rank computation
- **SIMD Alignment**: 16-byte alignment verification
- **UB Prevention**: Type-punning safety, memory aliasing tests

## Adding New Tests

To add new test cases, use the Google Test `TEST_F` macro:

```cpp
TEST_F(VecTest, MyNewTest)
{
    Vec<float, 3> v(1.0f, 2.0f, 3.0f);
    float result = v.my_new_operation();
    
    EXPECT_FLOAT_EQ(result, expected_value);
}
```

### Common Assertions

- `EXPECT_TRUE(condition)` / `ASSERT_TRUE(condition)`
- `EXPECT_FALSE(condition)` / `ASSERT_FALSE(condition)`
- `EXPECT_EQ(a, b)` / `ASSERT_EQ(a, b)` - Exact equality
- `EXPECT_FLOAT_EQ(a, b)` - Float equality (handles precision)
- `EXPECT_NEAR(a, b, epsilon)` - Near equality with custom epsilon
- `EXPECT_LT(a, b)`, `EXPECT_GT(a, b)` - Less than, greater than

Difference between `EXPECT_*` and `ASSERT_*`:
- `EXPECT_*`: Continues test execution after failure
- `ASSERT_*`: Stops test execution on failure

## CMSIS-DSP Testing

To test with CMSIS-DSP optimizations enabled:

1. Ensure CMSIS-DSP is installed
2. Build MatrixLib with CMSIS support:

```bash
cd tests/google
mkdir build && cd build
cmake -DMATRIXLIB_ENABLE_CMSIS=ON ..
make
./matrixlib_gtests
```

The same test suite validates both the standard C++ implementation and CMSIS-optimized code paths.

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: sudo apt-get install -y libgtest-dev cmake
      
    - name: Build tests
      run: |
        cd tests/google
        mkdir build && cd build
        cmake ..
        make
        
    - name: Run tests
      run: |
        cd tests/google/build
        ./matrixlib_gtests --gtest_output=xml:test_results.xml
        
    - name: Publish test results
      uses: EnricoMi/publish-unit-test-result-action@v2
      if: always()
      with:
        files: tests/google/build/test_results.xml
```

## Continuous Integration

Run tests automatically on every commit using:
- GitHub Actions
- GitLab CI
- Jenkins
- Travis CI
- CircleCI

## Benchmarking

For performance testing, consider using Google Benchmark alongside Google Test:

```bash
sudo apt-get install libbenchmark-dev
```

## Troubleshooting

### Google Test not found

If CMake cannot find Google Test:

```bash
# Specify GTest installation path
cmake -DGTEST_ROOT=/path/to/gtest ..

# Or use pkg-config
export PKG_CONFIG_PATH=/path/to/gtest/lib/pkgconfig
```

### Floating-point precision issues

Adjust the epsilon value in test fixtures:

```cpp
static constexpr float epsilon = 0.001f;  // Increase for more tolerance
```

### Tests fail on specific platforms

Platform-specific floating-point behavior may cause minor differences. Use `EXPECT_NEAR` instead of `EXPECT_FLOAT_EQ` for more robust tests across platforms.

## Code Coverage

Generate code coverage reports using gcov/lcov:

```bash
cd tests/google
mkdir build && cd build
cmake -DCMAKE_CXX_FLAGS="--coverage" ..
make
./matrixlib_gtests
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

Open `coverage_html/index.html` in a browser to view the coverage report.

## Expected Output

A successful test run should display:

```
[==========] Running 71 tests from 3 test suites.
[----------] Global test environment set-up.
[----------] 26 tests from VecTest
[ RUN      ] VecTest.Construction
[       OK ] VecTest.Construction (0 ms)
[ RUN      ] VecTest.Addition
[       OK ] VecTest.Addition (0 ms)
...
[----------] 26 tests from VecTest (X ms total)

[----------] 23 tests from MatTest
[ RUN      ] MatTest.Construction
[       OK ] MatTest.Construction (0 ms)
...
[----------] 23 tests from MatTest (X ms total)

[----------] 22 tests from QuatTest
[ RUN      ] QuatTest.Construction
[       OK ] QuatTest.Construction (0 ms)
...
[----------] 22 tests from QuatTest (X ms total)

[----------] Global test environment tear-down
[==========] 71 tests from 3 test suites ran. (X ms total)
[  PASSED  ] 71 tests.
```

## Additional Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [Google Test Advanced Guide](https://google.github.io/googletest/advanced.html)
