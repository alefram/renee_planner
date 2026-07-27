#include "renee_planner/planner_node.hpp"

#include <exception>

namespace renee_planner
{

PlannerNode::PlannerNode(const rclcpp::NodeOptions & options)
: Node("planner_node", options)
{
  config_file_ = this->declare_parameter<std::string>("config_file", "");
  output_file_ = this->declare_parameter<std::string>("output_file", "");

  RCLCPP_INFO(this->get_logger(), "Planner node initialized.");
  RCLCPP_INFO(this->get_logger(), "Config file: '%s'", config_file_.c_str());
  RCLCPP_INFO(this->get_logger(), "Output file: '%s'", output_file_.c_str());
}

int PlannerNode::run()
{
  if (config_file_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "config_file parameter is required");
    return 1;
  }
  if (output_file_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "output_file parameter is required");
    return 1;
  }

  try {
    const auto config = config_loader_.loadFromYamlFile(config_file_);
    const auto plan = generator_.generateInspectionPath(config);
    plan_writer_.writeYamlFile(plan, output_file_);
    RCLCPP_INFO(
      this->get_logger(),
      "Generated scan plan with %zu waypoints: %s",
      plan.waypoints.size(),
      output_file_.c_str());
    return 0;
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(
      this->get_logger(),
      "Failed to generate scan plan: %s",
      exception.what());
    return 1;
  }
}

}  // namespace renee_planner
