// SPDX-License-Identifier: MIT
/// @file bench_constexpr.cpp
/// @brief Constexpr vs runtime initialization benchmarks
/// @copyright Copyright (c) 2026 James Baldwin

#include <matrixlib/matrixlib.hpp>
#include <benchmark/benchmark.h>

using namespace matrixlib;

// Benchmark compile-time identity matrix usage
static void BM_Identity_Constexpr(benchmark::State& state)
{
    for (auto _ : state)
    {
        constexpr SquareMat<float, 4> identity = SquareMat<float, 4>::identity();
        benchmark::DoNotOptimize(identity);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Identity_Constexpr);

// Benchmark runtime identity matrix creation
static void BM_Identity_Runtime(benchmark::State& state)
{
    for (auto _ : state)
    {
        SquareMat<float, 4> identity = SquareMat<float, 4>::identity();
        benchmark::DoNotOptimize(identity);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Identity_Runtime);

// Benchmark compile-time zero vector
static void BM_Zero_Vector_Constexpr(benchmark::State& state)
{
    for (auto _ : state)
    {
        constexpr Vec3f zero = Vec3f::zero();
        benchmark::DoNotOptimize(zero);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Zero_Vector_Constexpr);

// Benchmark runtime zero vector
static void BM_Zero_Vector_Runtime(benchmark::State& state)
{
    for (auto _ : state)
    {
        Vec3f zero = Vec3f::zero();
        benchmark::DoNotOptimize(zero);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Zero_Vector_Runtime);

// Benchmark compile-time rotation matrix (special angles)
static void BM_Rotation_CompileTime(benchmark::State& state)
{
    for (auto _ : state)
    {
        constexpr auto rot90 = SquareMat<float, 2>::rotation_deg<90>();
        benchmark::DoNotOptimize(rot90);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Rotation_CompileTime);

// Benchmark runtime rotation matrix
static void BM_Rotation_Runtime(benchmark::State& state)
{
    const float angle = 1.5707963f;  // 90 degrees in radians

    for (auto _ : state)
    {
        auto rot = SquareMat<float, 2>::rotation(angle);
        benchmark::DoNotOptimize(rot);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Rotation_Runtime);

// Benchmark using pre-computed constexpr lookups
static void BM_Rotation_Lookup_Table(benchmark::State& state)
{
    static constexpr SquareMat<float, 2> rotations[] = {
        SquareMat<float, 2>::rotation_deg<0>(), SquareMat<float, 2>::rotation_deg<90>(),
        SquareMat<float, 2>::rotation_deg<180>(), SquareMat<float, 2>::rotation_deg<270>()};

    int idx = 0;
    for (auto _ : state)
    {
        const auto& rot = rotations[idx % 4];
        benchmark::DoNotOptimize(rot);
        idx++;
    }
}
BENCHMARK(BM_Rotation_Lookup_Table);

// Benchmark identity quaternion
static void BM_Quaternion_Identity_Constexpr(benchmark::State& state)
{
    for (auto _ : state)
    {
        constexpr Quaternion<float> q = Quaternion<float>::identity();
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Quaternion_Identity_Constexpr);

// Benchmark mathematical constants usage
static void BM_Constants_CompileTime(benchmark::State& state)
{
    for (auto _ : state)
    {
        constexpr float pi = constants::pi<float>;
        constexpr float two_pi = constants::two_pi<float>;
        float result = pi * two_pi;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Constants_CompileTime);
