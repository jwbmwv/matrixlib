// SPDX-License-Identifier: MIT
/// @file bench_vector_ops.cpp
/// @brief Vector operation performance benchmarks
/// @copyright Copyright (c) 2026 James Baldwin

#include <matrixlib/matrixlib.hpp>
#include <benchmark/benchmark.h>
#include <cmath>

using namespace matrixlib;

// Benchmark vector length calculation
static void BM_Vec3_Length(benchmark::State& state)
{
    Vec3f v{3.0f, 4.0f, 5.0f};

    for (auto _ : state)
    {
        float len = v.length();
        benchmark::DoNotOptimize(len);
    }
}
BENCHMARK(BM_Vec3_Length);

// Benchmark squared length (avoids sqrt)
static void BM_Vec3_LengthSquared(benchmark::State& state)
{
    Vec3f v{3.0f, 4.0f, 5.0f};

    for (auto _ : state)
    {
        float len2 = v.length_squared();
        benchmark::DoNotOptimize(len2);
    }
}
BENCHMARK(BM_Vec3_LengthSquared);

// Benchmark distance between vectors
static void BM_Vec3_Distance(benchmark::State& state)
{
    Vec3f a{1.0f, 2.0f, 3.0f};
    Vec3f b{4.0f, 5.0f, 6.0f};

    for (auto _ : state)
    {
        float dist = a.distance(b);
        benchmark::DoNotOptimize(dist);
    }
}
BENCHMARK(BM_Vec3_Distance);

// Benchmark linear interpolation
static void BM_Vec3_Lerp(benchmark::State& state)
{
    Vec3f a{0.0f, 0.0f, 0.0f};
    Vec3f b{10.0f, 10.0f, 10.0f};
    float t = 0.5f;

    for (auto _ : state)
    {
        Vec3f result = a.lerp(b, t);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Lerp);

// Benchmark reflection
static void BM_Vec3_Reflect(benchmark::State& state)
{
    Vec3f incident{1.0f, -1.0f, 0.0f};
    Vec3f normal{0.0f, 1.0f, 0.0f};

    for (auto _ : state)
    {
        Vec3f result = incident.reflect(normal);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Reflect);

// Benchmark projection
static void BM_Vec3_Project(benchmark::State& state)
{
    Vec3f v{1.0f, 2.0f, 3.0f};
    Vec3f onto{1.0f, 0.0f, 0.0f};

    for (auto _ : state)
    {
        Vec3f result = v.project(onto);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Project);

// Benchmark clamping
static void BM_Vec3_Clamp(benchmark::State& state)
{
    Vec3f v{5.0f, -5.0f, 15.0f};
    Vec3f min_v{0.0f, 0.0f, 0.0f};
    Vec3f max_v{10.0f, 10.0f, 10.0f};

    for (auto _ : state)
    {
        Vec3f result = v.clamp(min_v, max_v);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Clamp);

// Benchmark swizzle operations
static void BM_Vec3_Swizzle(benchmark::State& state)
{
    Vec3f v{1.0f, 2.0f, 3.0f};

    for (auto _ : state)
    {
        Vec3f result = v.swizzle<2, 1, 0>();  // z, y, x
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Swizzle);

// Benchmark angle between vectors
static void BM_Vec3_Angle(benchmark::State& state)
{
    Vec3f a{1.0f, 0.0f, 0.0f};
    Vec3f b{0.0f, 1.0f, 0.0f};

    for (auto _ : state)
    {
        float angle = a.angle(b);
        benchmark::DoNotOptimize(angle);
    }
}
BENCHMARK(BM_Vec3_Angle);

// Benchmark component-wise operations
static void BM_Vec3_ComponentWiseMultiply(benchmark::State& state)
{
    Vec3f a{1.0f, 2.0f, 3.0f};
    Vec3f b{4.0f, 5.0f, 6.0f};

    for (auto _ : state)
    {
        Vec3f result{a[0] * b[0], a[1] * b[1], a[2] * b[2]};
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_ComponentWiseMultiply);

// Benchmark safe normalization
static void BM_Vec3_SafeNormalize(benchmark::State& state)
{
    Vec3f v{0.001f, 0.001f, 0.001f};  // Very small vector

    for (auto _ : state)
    {
        Vec3f result = v.normalized();
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_SafeNormalize);

// Benchmark batch operations
static void BM_Vec3_Batch_Normalize(benchmark::State& state)
{
    const int N = 100;
    Vec3f vectors[N];
    for (int i = 0; i < N; ++i)
    {
        vectors[i] = Vec3f{static_cast<float>(i + 1), static_cast<float>(i + 2), static_cast<float>(i + 3)};
    }

    for (auto _ : state)
    {
        for (int i = 0; i < N; ++i)
        {
            vectors[i] = vectors[i].normalized();
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Vec3_Batch_Normalize);
