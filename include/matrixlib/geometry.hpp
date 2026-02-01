// SPDX-License-Identifier: MIT
/// @file geometry.hpp
/// @brief Geometric primitives and utilities for collision detection and ray tracing
/// @details Provides AABB, spheres, rays, planes, and intersection tests
/// @copyright Copyright (c) 2026 James Baldwin
/// @author James Baldwin

#pragma once

#include "vec3D.hpp"
#include "matrix.hpp"
#include <optional>
#include <algorithm>
#include <cmath>

namespace matrixlib
{
namespace geometry
{

// ==================== Ray ====================

/// @brief Ray defined by origin and direction
template<typename T>
struct Ray
{
    Vec<T, 3> origin;
    Vec<T, 3> direction;  ///< Should be normalized for accurate distance calculations

    /// Construct ray from origin and direction
    Ray(const Vec<T, 3>& o, const Vec<T, 3>& d) : origin(o), direction(d) {}

    /// Get point along ray at parameter t
    Vec<T, 3> at(T t) const { return origin + direction * t; }
};

using Rayf = Ray<float>;
using Rayd = Ray<double>;

// ==================== Plane ====================

/// @brief Plane defined by normal and distance from origin
template<typename T>
struct Plane
{
    Vec<T, 3> normal;  ///< Unit normal vector
    T distance;        ///< Signed distance from origin

    /// Construct plane from normal and distance
    Plane(const Vec<T, 3>& n, T d) : normal(n.normalized()), distance(d) {}

    /// Construct plane from normal and point on plane
    static Plane from_point_normal(const Vec<T, 3>& point, const Vec<T, 3>& normal)
    {
        Vec<T, 3> n = normal.normalized();
        return Plane(n, n.dot(point));
    }

    /// Construct plane from three points (counter-clockwise)
    static Plane from_points(const Vec<T, 3>& p0, const Vec<T, 3>& p1, const Vec<T, 3>& p2)
    {
        Vec<T, 3> v1 = p1 - p0;
        Vec<T, 3> v2 = p2 - p0;
        Vec<T, 3> normal = v1.cross(v2).normalized();
        return Plane(normal, normal.dot(p0));
    }

    /// Signed distance from point to plane (positive = in front)
    T distance_to(const Vec<T, 3>& point) const { return normal.dot(point) - distance; }

    /// Project point onto plane
    Vec<T, 3> project(const Vec<T, 3>& point) const { return point - normal * distance_to(point); }
};

using Planef = Plane<float>;
using Planed = Plane<double>;

// ==================== Axis-Aligned Bounding Box ====================

/// @brief Axis-aligned bounding box
template<typename T>
struct AABB
{
    Vec<T, 3> min;  ///< Minimum corner
    Vec<T, 3> max;  ///< Maximum corner

    /// Default constructor (empty box)
    AABB() : min(Vec<T, 3>()), max(Vec<T, 3>()) {}

    /// Construct from min and max corners
    AABB(const Vec<T, 3>& minimum, const Vec<T, 3>& maximum) : min(minimum), max(maximum) {}

    /// Construct from center and extents
    static AABB from_center_extents(const Vec<T, 3>& center, const Vec<T, 3>& extents)
    {
        return AABB(center - extents, center + extents);
    }

    /// Get center point
    Vec<T, 3> center() const { return (min + max) * static_cast<T>(0.5); }

    /// Get half-extents (distance from center to faces)
    Vec<T, 3> extents() const { return (max - min) * static_cast<T>(0.5); }

    /// Get size (full width, height, depth)
    Vec<T, 3> size() const { return max - min; }

    /// Check if point is inside AABB
    bool contains(const Vec<T, 3>& point) const
    {
        return (point.x() >= min.x() && point.x() <= max.x()) && (point.y() >= min.y() && point.y() <= max.y()) &&
               (point.z() >= min.z() && point.z() <= max.z());
    }

    /// Check if two AABBs intersect
    bool intersects(const AABB& other) const
    {
        return (min.x() <= other.max.x() && max.x() >= other.min.x()) &&
               (min.y() <= other.max.y() && max.y() >= other.min.y()) &&
               (min.z() <= other.max.z() && max.z() >= other.min.z());
    }

    /// Merge two AABBs
    AABB merge(const AABB& other) const
    {
        return AABB(Vec<T, 3>(std::min(min.x(), other.min.x()), std::min(min.y(), other.min.y()),
                              std::min(min.z(), other.min.z())),
                    Vec<T, 3>(std::max(max.x(), other.max.x()), std::max(max.y(), other.max.y()),
                              std::max(max.z(), other.max.z())));
    }

    /// Expand AABB to include point
    void expand(const Vec<T, 3>& point)
    {
        min = Vec<T, 3>(std::min(min.x(), point.x()), std::min(min.y(), point.y()), std::min(min.z(), point.z()));
        max = Vec<T, 3>(std::max(max.x(), point.x()), std::max(max.y(), point.y()), std::max(max.z(), point.z()));
    }

    /// Get surface area (useful for BVH cost heuristics)
    T surface_area() const
    {
        Vec<T, 3> s = size();
        return static_cast<T>(2) * (s.x() * s.y() + s.y() * s.z() + s.z() * s.x());
    }

    /// Get volume
    T volume() const
    {
        Vec<T, 3> s = size();
        return s.x() * s.y() * s.z();
    }
};

using AABBf = AABB<float>;
using AABBd = AABB<double>;

// ==================== Sphere ====================

/// @brief Sphere defined by center and radius
template<typename T>
struct Sphere
{
    Vec<T, 3> center;
    T radius;

    /// Construct sphere from center and radius
    Sphere(const Vec<T, 3>& c, T r) : center(c), radius(r) {}

    /// Check if point is inside sphere
    bool contains(const Vec<T, 3>& point) const
    {
        Vec<T, 3> diff = point - center;
        return diff.dot(diff) <= radius * radius;
    }

    /// Check if two spheres intersect
    bool intersects(const Sphere& other) const
    {
        Vec<T, 3> diff = center - other.center;
        T dist_sq = diff.dot(diff);
        T radius_sum = radius + other.radius;
        return dist_sq <= radius_sum * radius_sum;
    }

    /// Get AABB containing this sphere
    AABB<T> to_aabb() const
    {
        Vec<T, 3> extents(radius, radius, radius);
        return AABB<T>(center - extents, center + extents);
    }

    /// Get surface area
    T surface_area() const { return static_cast<T>(4) * constants::pi<T> * radius * radius; }

    /// Get volume
    T volume() const { return static_cast<T>(4.0 / 3.0) * constants::pi<T> * radius * radius * radius; }
};

using Spheref = Sphere<float>;
using Sphered = Sphere<double>;

// ==================== Intersection Tests ====================

/// @brief Ray-Sphere intersection
/// @return Distance along ray to intersection point, or std::nullopt if no intersection
template<typename T>
std::optional<T> intersect(const Ray<T>& ray, const Sphere<T>& sphere)
{
    Vec<T, 3> oc = ray.origin - sphere.center;

    T a = ray.direction.dot(ray.direction);
    T b = static_cast<T>(2) * oc.dot(ray.direction);
    T c = oc.dot(oc) - sphere.radius * sphere.radius;

    T discriminant = b * b - static_cast<T>(4) * a * c;

    if (discriminant < static_cast<T>(0))
    {
        return std::nullopt;  // No intersection
    }

    T sqrt_disc = std::sqrt(discriminant);
    T t = (-b - sqrt_disc) / (static_cast<T>(2) * a);

    if (t < static_cast<T>(0))
    {
        t = (-b + sqrt_disc) / (static_cast<T>(2) * a);
    }

    return (t >= static_cast<T>(0)) ? std::optional<T>(t) : std::nullopt;
}

/// @brief Ray-Plane intersection
/// @return Distance along ray to intersection point, or std::nullopt if no intersection
template<typename T>
std::optional<T> intersect(const Ray<T>& ray, const Plane<T>& plane)
{
    T denom = plane.normal.dot(ray.direction);

    // Check if ray is parallel to plane
    if (std::abs(denom) < std::numeric_limits<T>::epsilon())
    {
        return std::nullopt;
    }

    T t = (plane.distance - plane.normal.dot(ray.origin)) / denom;

    return (t >= static_cast<T>(0)) ? std::optional<T>(t) : std::nullopt;
}

/// @brief Ray-AABB intersection (slab method)
/// @return Distance along ray to intersection point, or std::nullopt if no intersection
template<typename T>
std::optional<T> intersect(const Ray<T>& ray, const AABB<T>& aabb)
{
    T t_min = static_cast<T>(0);
    T t_max = std::numeric_limits<T>::infinity();

    for (int i = 0; i < 3; ++i)
    {
        T inv_d = static_cast<T>(1) / ray.direction[i];
        T t0 = (aabb.min[i] - ray.origin[i]) * inv_d;
        T t1 = (aabb.max[i] - ray.origin[i]) * inv_d;

        if (inv_d < static_cast<T>(0))
        {
            std::swap(t0, t1);
        }

        t_min = std::max(t_min, t0);
        t_max = std::min(t_max, t1);

        if (t_max < t_min)
        {
            return std::nullopt;
        }
    }

    return (t_min >= static_cast<T>(0)) ? std::optional<T>(t_min) : std::nullopt;
}

/// @brief Sphere-AABB intersection
template<typename T>
bool intersects(const Sphere<T>& sphere, const AABB<T>& aabb)
{
    // Find closest point on AABB to sphere center
    Vec<T, 3> closest(std::clamp(sphere.center.x(), aabb.min.x(), aabb.max.x()),
                      std::clamp(sphere.center.y(), aabb.min.y(), aabb.max.y()),
                      std::clamp(sphere.center.z(), aabb.min.z(), aabb.max.z()));

    Vec<T, 3> diff = closest - sphere.center;
    return diff.dot(diff) <= sphere.radius * sphere.radius;
}

// ==================== Triangle ====================

/// @brief Triangle defined by three vertices
template<typename T>
struct Triangle
{
    Vec<T, 3> v0, v1, v2;

    /// Construct triangle from three vertices
    Triangle(const Vec<T, 3>& a, const Vec<T, 3>& b, const Vec<T, 3>& c) : v0(a), v1(b), v2(c) {}

    /// Get triangle normal (counter-clockwise)
    Vec<T, 3> normal() const
    {
        Vec<T, 3> e1 = v1 - v0;
        Vec<T, 3> e2 = v2 - v0;
        return e1.cross(e2).normalized();
    }

    /// Get triangle area
    T area() const
    {
        Vec<T, 3> e1 = v1 - v0;
        Vec<T, 3> e2 = v2 - v0;
        return e1.cross(e2).length() * static_cast<T>(0.5);
    }

    /// Get centroid
    Vec<T, 3> centroid() const { return (v0 + v1 + v2) * (static_cast<T>(1) / static_cast<T>(3)); }

    /// Get AABB
    AABB<T> to_aabb() const
    {
        AABB<T> box;
        box.expand(v0);
        box.expand(v1);
        box.expand(v2);
        return box;
    }
};

using Trianglef = Triangle<float>;
using Triangled = Triangle<double>;

/// @brief Ray-Triangle intersection (Möller-Trumbore algorithm)
/// @return Distance along ray and barycentric coordinates (u, v), or std::nullopt if no intersection
template<typename T>
std::optional<std::tuple<T, T, T>> intersect(const Ray<T>& ray, const Triangle<T>& tri)
{
    constexpr T epsilon = std::numeric_limits<T>::epsilon();

    Vec<T, 3> edge1 = tri.v1 - tri.v0;
    Vec<T, 3> edge2 = tri.v2 - tri.v0;

    Vec<T, 3> h = ray.direction.cross(edge2);
    T a = edge1.dot(h);

    // Ray parallel to triangle
    if (std::abs(a) < epsilon)
    {
        return std::nullopt;
    }

    T f = static_cast<T>(1) / a;
    Vec<T, 3> s = ray.origin - tri.v0;
    T u = f * s.dot(h);

    if (u < static_cast<T>(0) || u > static_cast<T>(1))
    {
        return std::nullopt;
    }

    Vec<T, 3> q = s.cross(edge1);
    T v = f * ray.direction.dot(q);

    if (v < static_cast<T>(0) || u + v > static_cast<T>(1))
    {
        return std::nullopt;
    }

    T t = f * edge2.dot(q);

    if (t > epsilon)
    {
        return std::make_tuple(t, u, v);
    }

    return std::nullopt;
}

// ==================== Frustum ====================

/// @brief View frustum defined by 6 planes
template<typename T>
struct Frustum
{
    std::array<Plane<T>, 6> planes;  ///< Left, Right, Bottom, Top, Near, Far

    enum PlaneIndex
    {
        LEFT = 0,
        RIGHT = 1,
        BOTTOM = 2,
        TOP = 3,
        NEAR = 4,
        FAR = 5
    };

    /// Construct frustum from view-projection matrix
    static Frustum from_matrix(const Mat<T, 4, 4>& vp)
    {
        Frustum f;

        // Extract planes from VP matrix (Gribb-Hartmann method)
        // Left plane
        f.planes[LEFT] =
            Plane<T>(Vec<T, 3>(vp(0, 3) + vp(0, 0), vp(1, 3) + vp(1, 0), vp(2, 3) + vp(2, 0)), vp(3, 3) + vp(3, 0));

        // Right plane
        f.planes[RIGHT] =
            Plane<T>(Vec<T, 3>(vp(0, 3) - vp(0, 0), vp(1, 3) - vp(1, 0), vp(2, 3) - vp(2, 0)), vp(3, 3) - vp(3, 0));

        // Bottom plane
        f.planes[BOTTOM] =
            Plane<T>(Vec<T, 3>(vp(0, 3) + vp(0, 1), vp(1, 3) + vp(1, 1), vp(2, 3) + vp(2, 1)), vp(3, 3) + vp(3, 1));

        // Top plane
        f.planes[TOP] =
            Plane<T>(Vec<T, 3>(vp(0, 3) - vp(0, 1), vp(1, 3) - vp(1, 1), vp(2, 3) - vp(2, 1)), vp(3, 3) - vp(3, 1));

        // Near plane
        f.planes[NEAR] =
            Plane<T>(Vec<T, 3>(vp(0, 3) + vp(0, 2), vp(1, 3) + vp(1, 2), vp(2, 3) + vp(2, 2)), vp(3, 3) + vp(3, 2));

        // Far plane
        f.planes[FAR] =
            Plane<T>(Vec<T, 3>(vp(0, 3) - vp(0, 2), vp(1, 3) - vp(1, 2), vp(2, 3) - vp(2, 2)), vp(3, 3) - vp(3, 2));

        return f;
    }

    /// Check if point is inside frustum
    bool contains(const Vec<T, 3>& point) const
    {
        for (const auto& plane : planes)
        {
            if (plane.distance_to(point) < static_cast<T>(0))
            {
                return false;
            }
        }
        return true;
    }

    /// Check if sphere intersects frustum
    bool intersects(const Sphere<T>& sphere) const
    {
        for (const auto& plane : planes)
        {
            if (plane.distance_to(sphere.center) < -sphere.radius)
            {
                return false;
            }
        }
        return true;
    }

    /// Check if AABB intersects frustum
    bool intersects(const AABB<T>& aabb) const
    {
        for (const auto& plane : planes)
        {
            // Find positive vertex (farthest along plane normal)
            Vec<T, 3> p(plane.normal.x() >= static_cast<T>(0) ? aabb.max.x() : aabb.min.x(),
                        plane.normal.y() >= static_cast<T>(0) ? aabb.max.y() : aabb.min.y(),
                        plane.normal.z() >= static_cast<T>(0) ? aabb.max.z() : aabb.min.z());

            if (plane.distance_to(p) < static_cast<T>(0))
            {
                return false;
            }
        }
        return true;
    }
};

using Frustumf = Frustum<float>;
using Frustumd = Frustum<double>;

}  // namespace geometry
}  // namespace matrixlib
