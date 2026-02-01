# Thread Safety and Concurrency

Documentation of thread safety guarantees and concurrency patterns for MatrixLib.

## Table of Contents

1. [Thread Safety Guarantees](#thread-safety-guarantees)
2. [Concurrent Access Patterns](#concurrent-access-patterns)
3. [Data Race Prevention](#data-race-prevention)
4. [Parallel Algorithms](#parallel-algorithms)
5. [Best Practices](#best-practices)

---

## Thread Safety Guarantees

### Summary

**MatrixLib is thread-safe for const operations on shared data, but not for concurrent modifications.**

| Operation Type | Thread Safe? | Notes |
|----------------|--------------|-------|
| **const methods** | ✅ Yes | Reading shared data is safe |
| **Non-const methods** | ❌ No | Modifications require synchronization |
| **Independent objects** | ✅ Yes | No shared state between objects |
| **Static methods** | ✅ Yes | No mutable static state |

### Details

#### ✅ Safe: Concurrent Reads

Multiple threads can safely read from the same matrix/vector:

```cpp
const Vec3f shared_vector(1, 2, 3);

// Thread 1
float len1 = shared_vector.norm();

// Thread 2 (simultaneously)
float len2 = shared_vector.norm();

// ✅ Safe - both threads reading const data
```

#### ✅ Safe: Independent Objects

Each thread working on its own data is always safe:

```cpp
void thread_func()
{
    // Each thread has its own local matrix
    Mat3f local_matrix = Mat3f::identity();
    local_matrix(0, 0) = 2.0f;  // ✅ Safe - thread-local data
}

std::thread t1(thread_func);
std::thread t2(thread_func);
```

#### ❌ Unsafe: Concurrent Writes

Multiple threads writing to the same object causes data races:

```cpp
Mat3f shared_matrix = Mat3f::identity();

// Thread 1
shared_matrix(0, 0) = 1.0f;  // ❌ UNSAFE!

// Thread 2 (simultaneously)
shared_matrix(1, 1) = 2.0f;  // ❌ UNSAFE!

// Data race - undefined behavior
```

#### ❌ Unsafe: Read + Write

One thread reading while another writes is also unsafe:

```cpp
Vec3f shared_vector(1, 2, 3);

// Thread 1
float x = shared_vector.x();  // ❌ UNSAFE!

// Thread 2 (simultaneously)
shared_vector.setX(5.0f);     // ❌ UNSAFE!

// Data race - undefined behavior
```

---

## Concurrent Access Patterns

### Pattern 1: Read-Only Shared Data (Lock-Free)

**Use Case:** Multiple threads reading the same transformation matrix.

```cpp
// Main thread sets up data
const Mat4f view_matrix = Mat4f::lookAt(eye, center, up);
const Mat4f proj_matrix = Mat4f::perspective(fov, aspect, near, far);

// Worker threads can safely read
void render_object(const Mat4f& view, const Mat4f& proj)
{
    Mat4f mvp = proj * view * model_matrix;
    // Use mvp for rendering
}

// ✅ Safe - const references, no modification
std::thread t1(render_object, std::cref(view_matrix), std::cref(proj_matrix));
std::thread t2(render_object, std::cref(view_matrix), std::cref(proj_matrix));
```

### Pattern 2: Thread-Local Copies

**Use Case:** Each thread works on independent copy of shared data.

```cpp
const Vec3f shared_direction(0, 1, 0);

void worker_thread()
{
    // Make thread-local copy
    Vec3f local_dir = shared_direction;  // ✅ Safe copy
    
    // Modify local copy freely
    local_dir = local_dir.normalized();
    local_dir = local_dir * 2.0f;
    
    // No synchronization needed
}

std::vector<std::thread> threads;
for (int i = 0; i < 4; ++i)
{
    threads.emplace_back(worker_thread);
}
```

### Pattern 3: Mutex-Protected Writes

**Use Case:** Multiple threads need to update shared data.

```cpp
#include <mutex>

Mat3f shared_matrix = Mat3f::identity();
std::mutex matrix_mutex;

void update_matrix(int row, int col, float value)
{
    std::lock_guard<std::mutex> lock(matrix_mutex);  // ✅ Acquire lock
    shared_matrix(row, col) = value;
}  // ✅ Lock released automatically

// Multiple threads can safely call update_matrix
std::thread t1(update_matrix, 0, 0, 1.0f);
std::thread t2(update_matrix, 1, 1, 2.0f);
```

### Pattern 4: Atomic Updates (for Simple Operations)

**Use Case:** Accumulating results from multiple threads.

```cpp
#include <atomic>

std::atomic<float> accumulated_result{0.0f};

void compute_partial_sum(const std::vector<Vec3f>& vectors, size_t start, size_t end)
{
    float local_sum = 0.0f;
    for (size_t i = start; i < end; ++i)
    {
        local_sum += vectors[i].norm();
    }
    
    // Atomic add
    float expected = accumulated_result.load();
    while (!accumulated_result.compare_exchange_weak(expected, expected + local_sum))
    {
        // Retry on failure
    }
}
```

### Pattern 5: Reduce/Map Parallelism

**Use Case:** Parallel processing with reduction.

```cpp
#include <vector>
#include <thread>
#include <numeric>

// Process vectors in parallel, combine results
std::vector<float> results(num_threads);

void process_chunk(const std::vector<Vec3f>& data, size_t start, size_t end, 
                   float& result)
{
    float sum = 0.0f;
    for (size_t i = start; i < end; ++i)
    {
        sum += data[i].normSquared();
    }
    result = sum;  // ✅ Each thread writes to its own result slot
}

std::vector<std::thread> threads;
size_t chunk_size = data.size() / num_threads;

for (size_t i = 0; i < num_threads; ++i)
{
    size_t start = i * chunk_size;
    size_t end = (i == num_threads - 1) ? data.size() : (i + 1) * chunk_size;
    threads.emplace_back(process_chunk, std::cref(data), start, end, 
                         std::ref(results[i]));
}

for (auto& t : threads) t.join();

// Combine results (serial)
float total = std::accumulate(results.begin(), results.end(), 0.0f);
```

---

## Data Race Prevention

### Use ThreadSanitizer (TSan)

Detect data races during development:

```bash
# Compile with ThreadSanitizer
cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
make
./your_test

# TSan will report any data races found
```

### Common Data Race Scenarios

#### Scenario 1: Shared Iterator

```cpp
// ❌ UNSAFE
std::vector<Vec3f> vectors;
auto it = vectors.begin();

// Thread 1
++it;

// Thread 2
++it;

// Data race on iterator!
```

**Fix:** Use thread-local iterators or partition data.

```cpp
// ✅ SAFE
void process_range(std::vector<Vec3f>::iterator start, 
                   std::vector<Vec3f>::iterator end)
{
    for (auto it = start; it != end; ++it)
    {
        // Process *it
    }
}
```

#### Scenario 2: Shared Accumulator

```cpp
// ❌ UNSAFE
float sum = 0.0f;

// Multiple threads
sum += vector.norm();  // Data race!
```

**Fix:** Use thread-local accumulation + reduction.

```cpp
// ✅ SAFE
std::vector<float> thread_sums(num_threads, 0.0f);

// Thread i accumulates to thread_sums[i]
thread_sums[thread_id] += vector.norm();

// Reduce after joining threads
float total = std::accumulate(thread_sums.begin(), thread_sums.end(), 0.0f);
```

---

## Parallel Algorithms

### Parallel Matrix Multiplication (Example)

```cpp
#include <thread>
#include <vector>

Mat<float, 100, 100> parallel_multiply(
    const Mat<float, 100, 100>& A,
    const Mat<float, 100, 100>& B,
    size_t num_threads = 4)
{
    Mat<float, 100, 100> C;
    
    auto compute_rows = [&](size_t start_row, size_t end_row)
    {
        for (size_t i = start_row; i < end_row; ++i)
        {
            for (size_t j = 0; j < 100; ++j)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < 100; ++k)
                {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;  // ✅ Each thread writes to different rows
            }
        }
    };
    
    std::vector<std::thread> threads;
    size_t rows_per_thread = 100 / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start = t * rows_per_thread;
        size_t end = (t == num_threads - 1) ? 100 : (t + 1) * rows_per_thread;
        threads.emplace_back(compute_rows, start, end);
    }
    
    for (auto& thread : threads)
    {
        thread.join();
    }
    
    return C;
}
```

### Parallel Vector Operations

```cpp
// Parallel vector normalization
void parallel_normalize(std::vector<Vec3f>& vectors, size_t num_threads = 4)
{
    auto normalize_range = [&](size_t start, size_t end)
    {
        for (size_t i = start; i < end; ++i)
        {
            if (vectors[i].normSquared() > 1e-6f)
            {
                vectors[i] = vectors[i].normalized();
            }
        }
    };
    
    std::vector<std::thread> threads;
    size_t chunk_size = vectors.size() / num_threads;
    
    for (size_t t = 0; t < num_threads; ++t)
    {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? vectors.size() : (t + 1) * chunk_size;
        threads.emplace_back(normalize_range, start, end);
    }
    
    for (auto& thread : threads)
    {
        thread.join();
    }
}
```

---

## Best Practices

### ✅ DO

1. **Prefer const references** for shared read-only data
   ```cpp
   void process(const Mat4f& transform);  // ✅ Good
   ```

2. **Use thread-local storage** for temporary computation
   ```cpp
   thread_local Mat3f temp_matrix;  // ✅ One per thread
   ```

3. **Partition data** by threads, avoid overlapping writes
   ```cpp
   // Each thread writes to disjoint rows
   for (size_t i = start_row; i < end_row; ++i) { ... }
   ```

4. **Reduce after parallel computation**
   ```cpp
   std::vector<float> partial_results(num_threads);
   // ... parallel computation ...
   float total = std::reduce(partial_results.begin(), partial_results.end());
   ```

5. **Use synchronization primitives** when necessary
   ```cpp
   std::lock_guard<std::mutex> lock(mutex);
   shared_data.modify();
   ```

### ❌ DON'T

1. **Don't share mutable state without synchronization**
   ```cpp
   // ❌ BAD
   Mat3f shared;
   // Thread 1 writes, Thread 2 reads - DATA RACE
   ```

2. **Don't assume atomic operations on multi-element types**
   ```cpp
   // ❌ BAD
   Vec3f shared_vec;
   shared_vec = other_vec;  // NOT atomic! (3 float writes)
   ```

3. **Don't use static mutable data**
   ```cpp
   // ❌ BAD
   static Mat3f cached_result;  // Shared across threads!
   ```

4. **Don't over-synchronize**
   ```cpp
   // ❌ BAD
   for (size_t i = 0; i < 1000000; ++i)
   {
       std::lock_guard<std::mutex> lock(mutex);  // Lock every iteration!
       // ... tiny amount of work ...
   }
   ```

---

## Performance Considerations

### Lock Granularity

```cpp
// ❌ Fine-grained (high contention)
std::mutex element_mutex;
for (size_t i = 0; i < N; ++i)
{
    std::lock_guard<std::mutex> lock(element_mutex);
    result += data[i];  // Lock held for tiny operation
}

// ✅ Coarse-grained (less contention)
float local_result = 0.0f;
for (size_t i = 0; i < N; ++i)
{
    local_result += data[i];  // No lock
}
std::lock_guard<std::mutex> lock(mutex);
result += local_result;  // Lock once
```

### False Sharing

```cpp
// ❌ False sharing - threads write to nearby memory
struct ThreadData
{
    float result;  // Only 4 bytes, but cache line is 64 bytes!
};
ThreadData data[NUM_THREADS];  // Adjacent in memory

// ✅ Pad to cache line size
struct alignas(64) ThreadData
{
    float result;
    char padding[60];  // Pad to 64 bytes
};
```

---

## Testing for Thread Safety

### Example Test

```cpp
#include <gtest/gtest.h>
#include <thread>
#include <vector>

TEST(ThreadSafety, ConcurrentReads)
{
    const Vec3f shared(1, 2, 3);
    std::vector<float> results(100);
    
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 100; ++i)
    {
        threads.emplace_back([&, i]()
        {
            results[i] = shared.norm();  // Concurrent reads
        });
    }
    
    for (auto& t : threads) t.join();
    
    // All results should be identical
    for (const auto& result : results)
    {
        EXPECT_FLOAT_EQ(result, shared.norm());
    }
}
```

---

## Summary

| Pattern | Use Case | Thread Safe? |
|---------|----------|--------------|
| Const shared data | Read-only access | ✅ Yes |
| Thread-local copies | Independent computation | ✅ Yes |
| Mutex-protected writes | Shared mutable state | ✅ Yes (with lock) |
| Partitioned data | Parallel processing | ✅ Yes (disjoint writes) |
| Unprotected shared writes | N/A | ❌ Never safe |

**Key Takeaway:** MatrixLib objects follow standard C++ thread safety rules - const operations are thread-safe, non-const operations require external synchronization.
