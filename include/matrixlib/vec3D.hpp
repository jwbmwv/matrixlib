// SPDX-License-Identifier: MIT
/// @file vec3D.hpp
/// @brief 3D vector specializations and extensions
/// @details This header provides 3D-specific vector functionality and type aliases.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#pragma once

#ifndef _MATRIXLIB_VEC3D_HPP_
#define _MATRIXLIB_VEC3D_HPP_

#include "vector.hpp"

namespace matrixlib
{

// 3D vector type aliases for common types
template<typename T>
using Vec3 = Vec<T, 3>;

using Vec3f = Vec<float, 3>;
using Vec3d = Vec<double, 3>;
using Vec3i = Vec<int, 3>;
using Vec3u = Vec<unsigned int, 3>;

// Static asserts for trivial copyability (C++11, replaces deprecated is_pod)
static_assert(std::is_trivially_copyable<Vec<float, 3>>::value, "Vec<float, 3> must be trivially copyable");
static_assert(std::is_trivially_copyable<Vec<float, 4>>::value, "Vec<float, 4> must be trivially copyable");

}  // namespace matrixlib

#endif  // _MATRIXLIB_VEC3D_HPP_
