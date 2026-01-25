# MatrixLib Tests

This directory is reserved for unit tests.

## Adding Tests

You can use any C++ testing framework:

### Google Test Example

```cpp
#include <gtest/gtest.h>
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

TEST(VecTest, Addition) {
    Vec<float, 3> a(1.0f, 2.0f, 3.0f);
    Vec<float, 3> b(4.0f, 5.0f, 6.0f);
    Vec<float, 3> sum = a + b;
    
    EXPECT_FLOAT_EQ(sum[0], 5.0f);
    EXPECT_FLOAT_EQ(sum[1], 7.0f);
    EXPECT_FLOAT_EQ(sum[2], 9.0f);
}

TEST(VecTest, DotProduct) {
    Vec<float, 3> a(1.0f, 0.0f, 0.0f);
    Vec<float, 3> b(0.0f, 1.0f, 0.0f);
    
    EXPECT_FLOAT_EQ(a.dot(b), 0.0f);
}
```

### Catch2 Example

```cpp
#include <catch2/catch.hpp>
#include <matrixlib/matrixlib.hpp>

using namespace matrixlib;

TEST_CASE("Vector operations", "[vec]") {
    Vec<float, 3> a(1.0f, 2.0f, 3.0f);
    Vec<float, 3> b(4.0f, 5.0f, 6.0f);
    
    SECTION("Addition") {
        Vec<float, 3> sum = a + b;
        REQUIRE(sum[0] == Approx(5.0f));
        REQUIRE(sum[1] == Approx(7.0f));
        REQUIRE(sum[2] == Approx(9.0f));
    }
    
    SECTION("Cross product") {
        Vec<float, 3> x(1, 0, 0);
        Vec<float, 3> y(0, 1, 0);
        Vec<float, 3> z = x.cross(y);
        
        REQUIRE(z[2] == Approx(1.0f));
    }
}
```

## Running Tests

```bash
mkdir build && cd build
cmake -DMATRIX_LINEAR_BUILD_TESTS=ON ..
cmake --build .
ctest --output-on-failure
```
