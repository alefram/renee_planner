#ifndef RENEE_PLANNER__SCAN_TYPES_HPP_
#define RENEE_PLANNER__SCAN_TYPES_HPP_

#include "renee_planner/scan_pattern.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <memory>
#include <string>
#include <vector>

namespace renee_planner
{

struct InspectionViewpoint
{
  std::string id;
  double height{0.0};
};

struct StructureVisualization
{
  std::string model;
  geometry_msgs::msg::Point origin;
  double yaw{0.0};
  std::vector<std::string> xacro_args;
  bool configured{false};
};

struct ScanConfig
{
  std::string global_frame_id{"world"};
  geometry_msgs::msg::Point machine_center;
  StructureVisualization structure;

  double base_height{0.0};
  // Owns the planar (x, y) trajectory shape (circular, or whatever gets
  // added later -- see scan_pattern.hpp). Radius/points-per-ring/etc. are
  // pattern-specific and live inside the concrete ScanPattern, not here.
  std::shared_ptr<ScanPattern> pattern;

  std::vector<InspectionViewpoint> inspection_viewpoints;
};

struct ScanWaypoint
{
  std::string id;
  geometry_msgs::msg::PoseStamped base_pose;
  geometry_msgs::msg::PoseStamped end_effector_pose;
};

struct ScanPlan
{
  std::string frame_id;
  geometry_msgs::msg::Point machine_center;
  StructureVisualization structure;
  std::vector<ScanWaypoint> waypoints;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__SCAN_TYPES_HPP_
