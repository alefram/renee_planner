#include "renee_planner/scan_plan_generator.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace renee_planner
{
namespace
{

constexpr double kEpsilon = 1e-9;

geometry_msgs::msg::Quaternion yawToQuaternion(const double yaw)
{
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.x = 0.0;
  quaternion.y = 0.0;
  quaternion.z = std::sin(yaw * 0.5);
  quaternion.w = std::cos(yaw * 0.5);
  return quaternion;
}

geometry_msgs::msg::Quaternion rotationMatrixToQuaternion(
  const double r00, const double r01, const double r02,
  const double r10, const double r11, const double r12,
  const double r20, const double r21, const double r22)
{
  geometry_msgs::msg::Quaternion quaternion;
  const double trace = r00 + r11 + r22;

  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    const double invs = 1.0 / s;
    quaternion.w = 0.25 * s;
    quaternion.x = (r21 - r12) * invs;
    quaternion.y = (r02 - r20) * invs;
    quaternion.z = (r10 - r01) * invs;
  } else if (r00 > r11 && r00 > r22) {
    const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    const double invs = 1.0 / s;
    quaternion.w = (r21 - r12) * invs;
    quaternion.x = 0.25 * s;
    quaternion.y = (r01 + r10) * invs;
    quaternion.z = (r02 + r20) * invs;
  } else if (r11 > r22) {
    const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    const double invs = 1.0 / s;
    quaternion.w = (r02 - r20) * invs;
    quaternion.x = (r01 + r10) * invs;
    quaternion.y = 0.25 * s;
    quaternion.z = (r12 + r21) * invs;
  } else {
    const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    const double invs = 1.0 / s;
    quaternion.w = (r10 - r01) * invs;
    quaternion.x = (r02 + r20) * invs;
    quaternion.y = (r12 + r21) * invs;
    quaternion.z = 0.25 * s;
  }

  const double norm = std::sqrt(
    quaternion.x * quaternion.x +
    quaternion.y * quaternion.y +
    quaternion.z * quaternion.z +
    quaternion.w * quaternion.w);
  if (norm <= kEpsilon) {
    throw std::runtime_error("Cannot normalize zero-length quaternion");
  }

  const double inv_norm = 1.0 / norm;
  quaternion.x *= inv_norm;
  quaternion.y *= inv_norm;
  quaternion.z *= inv_norm;
  quaternion.w *= inv_norm;
  return quaternion;
}

geometry_msgs::msg::Quaternion lookAtQuaternion(
  const geometry_msgs::msg::Point & position,
  const geometry_msgs::msg::Point & target)
{
  double forward_x = target.x - position.x;
  double forward_y = target.y - position.y;
  double forward_z = target.z - position.z;
  const double forward_norm = std::sqrt(
    forward_x * forward_x + forward_y * forward_y + forward_z * forward_z);

  if (forward_norm <= kEpsilon) {
    throw std::invalid_argument("End-effector position cannot match the machine center");
  }

  forward_x /= forward_norm;
  forward_y /= forward_norm;
  forward_z /= forward_norm;

  double up_x = 0.0;
  double up_y = 0.0;
  double up_z = 1.0;

  if (std::abs(forward_z) > 0.99) {
    up_x = 1.0;
    up_y = 0.0;
    up_z = 0.0;
  }

  double right_x = up_y * forward_z - up_z * forward_y;
  double right_y = up_z * forward_x - up_x * forward_z;
  double right_z = up_x * forward_y - up_y * forward_x;
  const double right_norm = std::sqrt(
    right_x * right_x + right_y * right_y + right_z * right_z);

  if (right_norm <= kEpsilon) {
    throw std::runtime_error("Cannot compute look-at orientation");
  }

  right_x /= right_norm;
  right_y /= right_norm;
  right_z /= right_norm;

  const double corrected_up_x = forward_y * right_z - forward_z * right_y;
  const double corrected_up_y = forward_z * right_x - forward_x * right_z;
  const double corrected_up_z = forward_x * right_y - forward_y * right_x;

  // tool0 convention (see ur_macro.xacro flange-tool0 joint): X+ left, Y+ up, Z+ front.
  // "right" here is actually the left-pointing vector (up x forward), matching X+ left.
  return rotationMatrixToQuaternion(
    right_x, corrected_up_x, forward_x,
    right_y, corrected_up_y, forward_y,
    right_z, corrected_up_z, forward_z);
}

void validateConfig(const ScanConfig & config)
{
  if (config.global_frame_id.empty()) {
    throw std::invalid_argument("global_frame_id cannot be empty");
  }
  if (config.global_frame_id != "robot_map") {
    throw std::invalid_argument(
      "Scanning navigation requires machine.frame_id to be 'robot_map'");
  }
  if (!config.pattern) {
    throw std::invalid_argument("A scan pattern must be configured");
  }
  if (config.inspection_viewpoints.empty()) {
    throw std::invalid_argument("At least one inspection height is required");
  }
}

}  // namespace

ScanPlan ScanPlanGenerator::generateInspectionPath(const ScanConfig & config) const
{
  validateConfig(config);

  ScanPlan plan;
  plan.frame_id = config.global_frame_id;
  plan.machine_center = config.machine_center;
  plan.structure = config.structure;

  const auto points = config.pattern->generatePoints(config.machine_center);
  plan.waypoints.reserve(points.size() * config.inspection_viewpoints.size());

  // Loop order is point (base position) outer, height inner, so every height
  // at a given point is visited before the base moves on to the next one --
  // otherwise the base would revisit every point once per height tier
  // instead of covering the whole pattern in a single pass.
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    const auto & point = points[point_index];

    geometry_msgs::msg::PoseStamped base_pose;
    base_pose.header.frame_id = config.global_frame_id;
    base_pose.pose.position.x = point.base_x;
    base_pose.pose.position.y = point.base_y;
    base_pose.pose.position.z = config.base_height;

    const double base_yaw = std::atan2(
      config.machine_center.y - base_pose.pose.position.y,
      config.machine_center.x - base_pose.pose.position.x);
    base_pose.pose.orientation = yawToQuaternion(base_yaw);

    for (const auto & viewpoint : config.inspection_viewpoints) {
      ScanWaypoint waypoint;
      char idbuf[64];
      std::snprintf(idbuf, sizeof(idbuf), "%s_%zu", viewpoint.id.c_str(), point_index);
      waypoint.id = idbuf;

      waypoint.base_pose = base_pose;

      waypoint.end_effector_pose.header.frame_id = config.global_frame_id;
      waypoint.end_effector_pose.pose.position.x = point.end_effector_x;
      waypoint.end_effector_pose.pose.position.y = point.end_effector_y;
      waypoint.end_effector_pose.pose.position.z = viewpoint.height;

      // Each band observes the structure horizontally at its own height.
      geometry_msgs::msg::Point target = config.machine_center;
      target.z = viewpoint.height;
      waypoint.end_effector_pose.pose.orientation = lookAtQuaternion(
        waypoint.end_effector_pose.pose.position,
        target);

      plan.waypoints.emplace_back(std::move(waypoint));
    }
  }

  return plan;
}

}  // namespace renee_planner
