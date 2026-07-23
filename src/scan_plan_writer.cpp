#include "renee_planner/scan_plan_writer.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <stdexcept>

namespace renee_planner
{
namespace
{

YAML::Node poseToYaml(const geometry_msgs::msg::PoseStamped & pose)
{
  YAML::Node node;
  node["frame_id"] = pose.header.frame_id;

  YAML::Node position;
  position.push_back(pose.pose.position.x);
  position.push_back(pose.pose.position.y);
  position.push_back(pose.pose.position.z);
  node["position"] = position;

  YAML::Node orientation;
  orientation.push_back(pose.pose.orientation.x);
  orientation.push_back(pose.pose.orientation.y);
  orientation.push_back(pose.pose.orientation.z);
  orientation.push_back(pose.pose.orientation.w);
  node["orientation"] = orientation;

  return node;
}

}  // namespace

void ScanPlanWriter::writeYamlFile(const ScanPlan & plan, const std::string & output_file) const
{
  if (output_file.empty()) {
    throw std::invalid_argument("output_file cannot be empty");
  }

  YAML::Node root;
  root["frame_id"] = plan.frame_id;

  YAML::Node machine_center;
  machine_center.push_back(plan.machine_center.x);
  machine_center.push_back(plan.machine_center.y);
  machine_center.push_back(plan.machine_center.z);
  root["machine_center"] = machine_center;

  YAML::Node waypoints;
  for (const auto & waypoint : plan.waypoints) {
    YAML::Node waypoint_node;
    waypoint_node["id"] = waypoint.id;
    waypoint_node["base_pose"] = poseToYaml(waypoint.base_pose);
    waypoint_node["end_effector_pose"] = poseToYaml(waypoint.end_effector_pose);
    waypoints.push_back(waypoint_node);
  }
  root["scan_waypoints"] = waypoints;

  std::ofstream output(output_file);
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open output file: " + output_file);
  }
  output << root << '\n';
}

}  // namespace renee_planner
