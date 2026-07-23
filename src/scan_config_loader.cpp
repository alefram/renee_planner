#include "renee_planner/scan_config_loader.hpp"

#include "renee_planner/circular_scan_pattern.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace renee_planner
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double degreesToRadians(const double degrees)
{
  return degrees * kPi / 180.0;
}

geometry_msgs::msg::Point parsePoint3(const YAML::Node & node, const std::string & field_name)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::invalid_argument(field_name + " must be a 3-value sequence");
  }

  geometry_msgs::msg::Point point;
  point.x = node[0].as<double>();
  point.y = node[1].as<double>();
  point.z = node[2].as<double>();
  return point;
}

template<typename T>
T requireScalar(const YAML::Node & node, const std::string & field_name)
{
  if (!node) {
    throw std::invalid_argument(field_name + " is required");
  }
  return node.as<T>();
}

std::vector<double> parseHeights(const YAML::Node & node)
{
  if (!node || !node.IsSequence() || node.size() == 0) {
    throw std::invalid_argument("scan.heights must be a non-empty sequence");
  }

  std::vector<double> heights;
  heights.reserve(node.size());
  for (const auto & height : node) {
    heights.push_back(height.as<double>());
  }
  return heights;
}

std::shared_ptr<ScanPattern> parseCircularPattern(const YAML::Node & scan)
{
  CircularScanPatternParams params;
  params.base_radius = requireScalar<double>(scan["base_radius"], "scan.base_radius");
  params.end_effector_radius =
    requireScalar<double>(scan["end_effector_radius"], "scan.end_effector_radius");
  params.points_per_ring = requireScalar<int>(scan["points_per_ring"], "scan.points_per_ring");
  params.start_angle_rad = scan["start_angle_deg"] ?
    degreesToRadians(scan["start_angle_deg"].as<double>()) : 0.0;
  return std::make_shared<CircularScanPattern>(params);
}

// To add a new trajectory shape: implement a ScanPattern (see
// circular_scan_pattern.hpp/.cpp), add a parseXxxPattern(scan) helper above
// that reads its YAML fields, and add a branch here for its
// `scan.pattern_type` name. Nothing outside this function needs to change.
std::shared_ptr<ScanPattern> parsePattern(const YAML::Node & scan)
{
  const std::string pattern_type = scan["pattern_type"] ?
    scan["pattern_type"].as<std::string>() : std::string("circular");

  if (pattern_type == "circular") {
    return parseCircularPattern(scan);
  }

  throw std::invalid_argument("Unknown scan.pattern_type: " + pattern_type);
}

}  // namespace

ScanConfig ScanConfigLoader::loadFromYamlFile(const std::string & config_file) const
{
  if (config_file.empty()) {
    throw std::invalid_argument("config_file cannot be empty");
  }

  const YAML::Node root = YAML::LoadFile(config_file);
  const YAML::Node machine = root["machine"];
  const YAML::Node scan = root["scan"];

  if (!machine) {
    throw std::invalid_argument("Missing required 'machine' section");
  }
  if (!scan) {
    throw std::invalid_argument("Missing required 'scan' section");
  }

  ScanConfig config;
  config.global_frame_id = machine["frame_id"] ?
    machine["frame_id"].as<std::string>() : std::string("world");
  config.machine_center = parsePoint3(machine["center"], "machine.center");
  config.base_height = scan["base_height"] ? scan["base_height"].as<double>() : 0.0;
  config.pattern = parsePattern(scan);

  const auto heights = parseHeights(scan["heights"]);
  config.inspection_viewpoints.reserve(heights.size());
  for (std::size_t index = 0; index < heights.size(); ++index) {
    InspectionViewpoint viewpoint;
    viewpoint.id = "height_" + std::to_string(index);
    viewpoint.height = heights[index];
    config.inspection_viewpoints.push_back(viewpoint);
  }

  return config;
}

}  // namespace renee_planner
