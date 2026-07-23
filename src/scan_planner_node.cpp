#include "renee_planner/scan_planner_node.hpp"

#include <exception>

namespace renee_planner
{

ScanPlannerNode::ScanPlannerNode(const rclcpp::NodeOptions & options)
: Node("scan_planner_node", options)
{
  config_file_ = this->declare_parameter<std::string>("config_file", "");
  output_file_ = this->declare_parameter<std::string>("output_file", "");

  RCLCPP_INFO(this->get_logger(), "Scan planner node initialized.");
  RCLCPP_INFO(this->get_logger(), "Config file: '%s'", config_file_.c_str());
  RCLCPP_INFO(this->get_logger(), "Output file: '%s'", output_file_.c_str());

  if (!config_file_.empty() && !output_file_.empty()) {
    try {
      const auto config = config_loader_.loadFromYamlFile(config_file_);
      const auto plan = generator_.generateInspectionPath(config);
      plan_writer_.writeYamlFile(plan, output_file_);
      RCLCPP_INFO(
        this->get_logger(),
        "Generated scan plan with %zu waypoints.",
        plan.waypoints.size());
    } catch (const std::exception & exception) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to generate scan plan: %s",
        exception.what());
    }
  }
}

}  // namespace renee_planner
