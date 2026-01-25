// SPDX-License-Identifier: MIT
/// @file version.hpp
/// @brief MatrixLib version information and API
/// @copyright Copyright (c) 2026 James Baldwin

#pragma once

#ifndef _MATRIXLIB_VERSION_HPP_
#define _MATRIXLIB_VERSION_HPP_

/// @brief MatrixLib major version number
#define MATRIXLIB_VERSION_MAJOR 1

/// @brief MatrixLib minor version number
#define MATRIXLIB_VERSION_MINOR 0

/// @brief MatrixLib patch version number
#define MATRIXLIB_VERSION_PATCH 0

/// @brief MatrixLib version string
#define MATRIXLIB_VERSION_STRING "1.0.0"

/// @brief Combined version number (MAJOR * 10000 + MINOR * 100 + PATCH)
#define MATRIXLIB_VERSION_NUMBER \
    ((MATRIXLIB_VERSION_MAJOR * 10000) + (MATRIXLIB_VERSION_MINOR * 100) + MATRIXLIB_VERSION_PATCH)

namespace matrixlib
{

/// @brief Version information structure
struct Version
{
    int major;  ///< Major version number
    int minor;  ///< Minor version number
    int patch;  ///< Patch version number

    /// @brief Get the version as a string
    /// @return Version string in the format "major.minor.patch"
    static const char* string() { return MATRIXLIB_VERSION_STRING; }

    /// @brief Get the combined version number
    /// @return Version number (major * 10000 + minor * 100 + patch)
    static int number() { return MATRIXLIB_VERSION_NUMBER; }

    /// @brief Get the major version number
    /// @return Major version
    static int get_major() { return MATRIXLIB_VERSION_MAJOR; }

    /// @brief Get the minor version number
    /// @return Minor version
    static int get_minor() { return MATRIXLIB_VERSION_MINOR; }

    /// @brief Get the patch version number
    /// @return Patch version
    static int get_patch() { return MATRIXLIB_VERSION_PATCH; }
};

/// @brief Get the library version as a string
/// @return Version string
inline const char* get_version_string()
{
    return Version::string();
}

/// @brief Get the library version as a number
/// @return Version number
inline int get_version_number()
{
    return Version::number();
}

/// @brief Check if the library version is at least the specified version
/// @param major Required major version
/// @param minor Required minor version
/// @param patch Required patch version
/// @return true if library version >= required version
inline bool version_at_least(int major, int minor, int patch)
{
    return MATRIXLIB_VERSION_NUMBER >= ((major * 10000) + (minor * 100) + patch);
}

}  // namespace matrixlib

#endif  // _MATRIXLIB_VERSION_HPP_
