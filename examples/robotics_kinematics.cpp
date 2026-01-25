// SPDX-License-Identifier: MIT
/// @file robotics_kinematics.cpp
/// @brief Forward and inverse kinematics for a 2-link planar robot arm
/// @details Demonstrates practical robotics calculations using MatrixLib
/// @copyright Copyright (c) 2026 James Baldwin

#include <matrixlib/matrixlib.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace matrixlib;

/// @brief 2-link planar robot arm (operates in XY plane)
class PlanarRobotArm
{
private:
    float link1_length_;  // Length of first link (meters)
    float link2_length_;  // Length of second link (meters)

public:
    PlanarRobotArm(float l1, float l2) : link1_length_(l1), link2_length_(l2) {}

    /// @brief Forward kinematics: joint angles -> end effector position
    /// @param theta1 Joint 1 angle (radians)
    /// @param theta2 Joint 2 angle (radians)
    /// @return End effector position in world frame
    Vec2f forward_kinematics(float theta1, float theta2) const
    {
        // Position of link1 end
        float x1 = link1_length_ * std::cos(theta1);
        float y1 = link1_length_ * std::sin(theta1);

        // Position of link2 end (end effector)
        float x2 = x1 + link2_length_ * std::cos(theta1 + theta2);
        float y2 = y1 + link2_length_ * std::sin(theta1 + theta2);

        return Vec2f{x2, y2};
    }

    /// @brief Inverse kinematics: end effector position -> joint angles
    /// @param target Target end effector position
    /// @param elbow_up True for elbow-up solution, false for elbow-down
    /// @return Joint angles [theta1, theta2] (radians), or NaN if unreachable
    Vec2f inverse_kinematics(const Vec2f& target, bool elbow_up = true) const
    {
        float x = target[0];
        float y = target[1];
        float r = std::sqrt(x * x + y * y);

        // Check if target is reachable
        if (r > link1_length_ + link2_length_ || r < std::abs(link1_length_ - link2_length_))
        {
            std::cerr << "Target unreachable! Distance: " << r << " meters\n";
            return Vec2f{NAN, NAN};
        }

        // Law of cosines for theta2
        float cos_theta2 = (r * r - link1_length_ * link1_length_ - link2_length_ * link2_length_) /
                           (2.0f * link1_length_ * link2_length_);
        cos_theta2 = clamp(cos_theta2, -1.0f, 1.0f);

        float theta2 = elbow_up ? std::acos(cos_theta2) : -std::acos(cos_theta2);

        // Solve for theta1
        float k1 = link1_length_ + link2_length_ * std::cos(theta2);
        float k2 = link2_length_ * std::sin(theta2);
        float theta1 = std::atan2(y, x) - std::atan2(k2, k1);

        return Vec2f{theta1, theta2};
    }

    /// @brief Compute Jacobian matrix (relates joint velocities to end effector velocities)
    /// @param theta1 Joint 1 angle (radians)
    /// @param theta2 Joint 2 angle (radians)
    /// @return 2x2 Jacobian matrix
    Mat2f jacobian(float theta1, float theta2) const
    {
        float s1 = std::sin(theta1);
        float c1 = std::cos(theta1);
        float s12 = std::sin(theta1 + theta2);
        float c12 = std::cos(theta1 + theta2);

        Mat2f J;
        J(0, 0) = -link1_length_ * s1 - link2_length_ * s12;
        J(0, 1) = -link2_length_ * s12;
        J(1, 0) = link1_length_ * c1 + link2_length_ * c12;
        J(1, 1) = link2_length_ * c12;

        return J;
    }

    /// @brief Get workspace boundary points (circle of reachable positions)
    /// @param num_points Number of boundary points to generate
    void print_workspace(int num_points = 16) const
    {
        std::cout << "\nWorkspace boundary (max reach = " << link1_length_ + link2_length_ << " m):\n";

        for (int i = 0; i < num_points; ++i)
        {
            float angle = (2.0f * constants::pi<float> * i) / num_points;
            float x = (link1_length_ + link2_length_) * std::cos(angle);
            float y = (link1_length_ + link2_length_) * std::sin(angle);
            std::cout << "  (" << std::setw(6) << x << ", " << std::setw(6) << y << ")\n";
        }
    }
};

int main()
{
    std::cout << "MatrixLib - Robot Kinematics Example\n";
    std::cout << "=====================================\n\n";

    // Create 2-link robot with 1m links
    PlanarRobotArm robot(1.0f, 1.0f);

    std::cout << std::fixed << std::setprecision(4);

    // Test 1: Forward kinematics
    std::cout << "=== Forward Kinematics ===\n";
    float theta1 = deg_to_rad(45.0f);
    float theta2 = deg_to_rad(30.0f);

    Vec2f end_pos = robot.forward_kinematics(theta1, theta2);
    std::cout << "Joint angles: θ1=" << rad_to_deg(theta1) << "°, θ2=" << rad_to_deg(theta2) << "°\n";
    std::cout << "End effector: (" << end_pos[0] << ", " << end_pos[1] << ") meters\n";

    // Test 2: Inverse kinematics
    std::cout << "\n=== Inverse Kinematics ===\n";
    Vec2f target{1.5f, 1.0f};
    std::cout << "Target position: (" << target[0] << ", " << target[1] << ") meters\n";

    Vec2f angles_up = robot.inverse_kinematics(target, true);
    if (!std::isnan(angles_up[0]))
    {
        std::cout << "Elbow-up solution: θ1=" << rad_to_deg(angles_up[0]) << "°, θ2=" << rad_to_deg(angles_up[1])
                  << "°\n";

        // Verify
        Vec2f verify = robot.forward_kinematics(angles_up[0], angles_up[1]);
        std::cout << "  Verification: (" << verify[0] << ", " << verify[1] << ")\n";
    }

    Vec2f angles_down = robot.inverse_kinematics(target, false);
    if (!std::isnan(angles_down[0]))
    {
        std::cout << "Elbow-down solution: θ1=" << rad_to_deg(angles_down[0]) << "°, θ2=" << rad_to_deg(angles_down[1])
                  << "°\n";
    }

    // Test 3: Jacobian and velocity kinematics
    std::cout << "\n=== Jacobian & Velocity Kinematics ===\n";
    Mat2f J = robot.jacobian(theta1, theta2);
    std::cout << "Jacobian at θ1=" << rad_to_deg(theta1) << "°, θ2=" << rad_to_deg(theta2) << "°:\n";
    std::cout << "  [" << J(0, 0) << "  " << J(0, 1) << "]\n";
    std::cout << "  [" << J(1, 0) << "  " << J(1, 1) << "]\n";

    // Joint velocities -> end effector velocity
    Vec2f joint_vel{deg_to_rad(10.0f), deg_to_rad(5.0f)};  // 10°/s, 5°/s
    Vec2f ee_vel = J * joint_vel;
    std::cout << "\nJoint velocities: " << rad_to_deg(joint_vel[0]) << "°/s, " << rad_to_deg(joint_vel[1]) << "°/s\n";
    std::cout << "End effector velocity: (" << ee_vel[0] << ", " << ee_vel[1] << ") m/s\n";

    // Test 4: Trajectory planning
    std::cout << "\n=== Trajectory Planning ===\n";
    Vec2f start{1.0f, 1.0f};
    Vec2f end{1.5f, 0.5f};
    std::cout << "Linear path from " << start << " to " << end << ":\n";

    for (int i = 0; i <= 5; ++i)
    {
        float t = i / 5.0f;
        Vec2f waypoint = start.lerp(end, t);
        Vec2f angles = robot.inverse_kinematics(waypoint, true);

        if (!std::isnan(angles[0]))
        {
            std::cout << "  t=" << t << ": pos(" << waypoint[0] << ", " << waypoint[1] << ") -> joints("
                      << rad_to_deg(angles[0]) << "°, " << rad_to_deg(angles[1]) << "°)\n";
        }
    }

    // Test 5: Workspace visualization
    robot.print_workspace();

    return 0;
}
