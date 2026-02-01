// SPDX-License-Identifier: MIT
/// @file vec2D.hpp
/// @brief 2D vector specializations and extensions
/// @details This header provides 2D-specific vector functionality and type aliases.
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin
/// @date 2026

#pragma once

#ifndef _MATRIXLIB_VEC2D_HPP_
#define _MATRIXLIB_VEC2D_HPP_

#include "vector.hpp"

namespace matrixlib
{

// 2D vector type aliases for common types
template<typename T>
using Vec2 = Vec<T, 2>;

using Vec2f = Vec<float, 2>;
using Vec2d = Vec<double, 2>;
using Vec2i = Vec<int, 2>;
using Vec2u = Vec<unsigned int, 2>;

}  // namespace matrixlib

#endif  // _MATRIXLIB_VEC2D_HPP_
