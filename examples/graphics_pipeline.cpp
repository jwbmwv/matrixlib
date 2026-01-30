// SPDX-License-Identifier: MIT
/// @file graphics_pipeline.cpp
/// @brief 3D graphics transformation pipeline using MatrixLib
/// @details Demonstrates view, projection, and viewport transformations
/// @copyright Copyright (c) 2026 James Baldwin

#include <matrixlib/matrixlib.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace matrixlib;

/// @brief Simple 3D mesh vertex
struct Vertex
{
    Vec3f position;
    Vec3f color;
};

/// @brief Simple 3D camera
class Camera
{
private:
    Vec3f position_;
    Vec3f target_;
    Vec3f up_;

public:
    Camera(const Vec3f& pos, const Vec3f& target, const Vec3f& up = Vec3f{0.0f, 1.0f, 0.0f})
        : position_(pos), target_(target), up_(up)
    {
    }

    /// @brief Get view matrix (world to camera space)
    SquareMat<float, 4> get_view_matrix() const { return SquareMat<float, 4>::look_at(position_, target_, up_); }

    /// @brief Get perspective projection matrix
    /// @param fov_y Field of view in Y direction (radians)
    /// @param aspect Aspect ratio (width/height)
    /// @param near Near clipping plane
    /// @param far Far clipping plane
    static SquareMat<float, 4> get_projection_matrix(float fov_y, float aspect, float near_plane, float far_plane)
    {
        return SquareMat<float, 4>::perspective(fov_y, aspect, near_plane, far_plane);
    }

    /// @brief Get orthographic projection matrix
    static SquareMat<float, 4> get_ortho_matrix(float left, float right, float bottom, float top, float near_plane,
                                                float far_plane)
    {
        return SquareMat<float, 4>::orthographic(left, right, bottom, top, near_plane, far_plane);
    }

    void set_position(const Vec3f& pos) { position_ = pos; }
    void set_target(const Vec3f& target) { target_ = target; }
    Vec3f get_position() const { return position_; }
};

/// @brief Transform vertex through graphics pipeline
Vec3f transform_vertex(const Vertex& v, const SquareMat<float, 4>& mvp)
{
    // Homogeneous coordinates
    Vec<float, 4> pos_h{v.position[0], v.position[1], v.position[2], 1.0f};

    // Apply MVP transform
    Vec<float, 4> clip_pos = mvp * pos_h;

    // Perspective divide
    if (std::abs(clip_pos[3]) > 1e-6f)
    {
        clip_pos[0] /= clip_pos[3];
        clip_pos[1] /= clip_pos[3];
        clip_pos[2] /= clip_pos[3];
    }

    // Return NDC coordinates (Normalized Device Coordinates)
    return Vec3f{clip_pos[0], clip_pos[1], clip_pos[2]};
}

/// @brief Convert NDC to screen coordinates
Vec2f ndc_to_screen(const Vec3f& ndc, int width, int height)
{
    // NDC range [-1, 1] -> Screen range [0, width-1], [0, height-1]
    float screen_x = (ndc[0] + 1.0f) * 0.5f * width;
    float screen_y = (1.0f - ndc[1]) * 0.5f * height;  // Flip Y (screen Y goes down)
    return Vec2f{screen_x, screen_y};
}

int main()
{
    std::cout << "MatrixLib - Graphics Pipeline Example\n";
    std::cout << "======================================\n\n";

    std::cout << std::fixed << std::setprecision(3);

    // Define a simple cube mesh
    std::vector<Vertex> cube_vertices = {
        {{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},  // 0: Front-bottom-left (red)
        {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},   // 1: Front-bottom-right (green)
        {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},    // 2: Front-top-right (blue)
        {{-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, 0.0f}},   // 3: Front-top-left (yellow)
        {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 1.0f}},   // 4: Back-bottom-left (magenta)
        {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}},    // 5: Back-bottom-right (cyan)
        {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}},     // 6: Back-top-right (white)
        {{-1.0f, 1.0f, 1.0f}, {0.5f, 0.5f, 0.5f}}     // 7: Back-top-left (gray)
    };

    // Setup camera
    Vec3f cam_pos{5.0f, 5.0f, 5.0f};
    Vec3f cam_target{0.0f, 0.0f, 0.0f};
    Camera camera(cam_pos, cam_target);

    std::cout << "=== Camera Setup ===\n";
    std::cout << "Position: " << cam_pos << "\n";
    std::cout << "Target: " << cam_target << "\n\n";

    // Model matrix (rotate cube)
    float angle = deg_to_rad(45.0f);
    SquareMat<float, 4> model = SquareMat<float, 4>::rotation(angle, Vec3f{0.0f, 1.0f, 0.0f});

    // View matrix
    SquareMat<float, 4> view = camera.get_view_matrix();

    // Projection matrix
    float fov = deg_to_rad(60.0f);
    float aspect = 16.0f / 9.0f;
    SquareMat<float, 4> projection = Camera::get_projection_matrix(fov, aspect, 0.1f, 100.0f);

    // Combined MVP matrix
    SquareMat<float, 4> mvp = projection * view * model;

    std::cout << "=== Transformation Matrices ===\n\n";

    std::cout << "Model Matrix (45° rotation around Y):\n";
    for (uint32_t i = 0; i < 4; ++i)
    {
        std::cout << "  [";
        for (uint32_t j = 0; j < 4; ++j)
        {
            std::cout << std::setw(8) << model(i, j);
        }
        std::cout << " ]\n";
    }

    std::cout << "\nView Matrix:\n";
    for (uint32_t i = 0; i < 4; ++i)
    {
        std::cout << "  [";
        for (uint32_t j = 0; j < 4; ++j)
        {
            std::cout << std::setw(8) << view(i, j);
        }
        std::cout << " ]\n";
    }

    std::cout << "\nProjection Matrix (FOV=" << rad_to_deg(fov) << "°):\n";
    for (uint32_t i = 0; i < 4; ++i)
    {
        std::cout << "  [";
        for (uint32_t j = 0; j < 4; ++j)
        {
            std::cout << std::setw(8) << projection(i, j);
        }
        std::cout << " ]\n";
    }

    // Transform vertices
    std::cout << "\n=== Vertex Transformation ===\n";
    std::cout << "World -> NDC -> Screen (800x600)\n\n";
    std::cout << "Vertex  World Position        NDC Position          Screen Position\n";
    std::cout << "------  ------------------    ------------------    ---------------\n";

    int screen_width = 800;
    int screen_height = 600;

    for (size_t i = 0; i < 4; ++i)  // Show first 4 vertices
    {
        const Vertex& v = cube_vertices[i];
        Vec3f ndc = transform_vertex(v, mvp);
        Vec2f screen = ndc_to_screen(ndc, screen_width, screen_height);

        std::cout << "  " << i << "     (" << std::setw(6) << v.position[0] << ", " << std::setw(6) << v.position[1]
                  << ", " << std::setw(6) << v.position[2] << ")   (" << std::setw(6) << ndc[0] << ", " << std::setw(6)
                  << ndc[1] << ", " << std::setw(6) << ndc[2] << ")   (" << std::setw(6) << screen[0] << ", "
                  << std::setw(6) << screen[1] << ")\n";
    }

    // Demonstrate different projections
    std::cout << "\n=== Orthographic Projection ===\n";
    SquareMat<float, 4> ortho = Camera::get_ortho_matrix(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 100.0f);
    SquareMat<float, 4> mvp_ortho = ortho * view * model;

    std::cout << "Vertex 0 in orthographic projection:\n";
    Vec3f ndc_ortho = transform_vertex(cube_vertices[0], mvp_ortho);
    Vec2f screen_ortho = ndc_to_screen(ndc_ortho, screen_width, screen_height);
    std::cout << "  NDC: (" << ndc_ortho[0] << ", " << ndc_ortho[1] << ", " << ndc_ortho[2] << ")\n";
    std::cout << "  Screen: (" << screen_ortho[0] << ", " << screen_ortho[1] << ")\n";

    // Camera movement demo
    std::cout << "\n=== Camera Animation ===\n";
    std::cout << "Orbiting camera around origin:\n\n";

    for (int frame = 0; frame <= 4; ++frame)
    {
        float t = frame / 4.0f;
        float orbit_angle = t * constants::two_pi<float>;
        float radius = 5.0f;

        Vec3f new_pos{radius * std::cos(orbit_angle), 5.0f, radius * std::sin(orbit_angle)};

        camera.set_position(new_pos);
        SquareMat<float, 4> new_view = camera.get_view_matrix();
        SquareMat<float, 4> new_mvp = projection * new_view * model;

        Vec3f ndc_frame = transform_vertex(cube_vertices[0], new_mvp);

        std::cout << "Frame " << frame << " (angle=" << std::setw(5) << rad_to_deg(orbit_angle) << "°): " << "NDC=("
                  << ndc_frame[0] << ", " << ndc_frame[1] << ", " << ndc_frame[2] << ")\n";
    }

    // Frustum culling example
    std::cout << "\n=== Frustum Culling ===\n";
    std::cout << "Vertices in/out of view frustum (NDC range: [-1, 1]):\n";

    for (size_t i = 0; i < cube_vertices.size(); ++i)
    {
        Vec3f ndc = transform_vertex(cube_vertices[i], mvp);
        bool in_frustum = (ndc[0] >= -1.0f && ndc[0] <= 1.0f && ndc[1] >= -1.0f && ndc[1] <= 1.0f && ndc[2] >= -1.0f &&
                           ndc[2] <= 1.0f);

        std::cout << "  Vertex " << i << ": " << (in_frustum ? "VISIBLE" : "CULLED") << "\n";
    }

    return 0;
}
