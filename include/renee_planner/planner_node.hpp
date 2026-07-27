#ifndef RENEE_PLANNER__PLANNER_NODE_HPP_
#define RENEE_PLANNER__PLANNER_NODE_HPP_

#include "renee_planner/scan_config_loader.hpp"
#include "renee_planner/scan_plan_generator.hpp"
#include "renee_planner/scan_plan_writer.hpp"

#include <rclcpp/rclcpp.hpp>

#include <string>

namespace renee_planner
{

class PlannerNode : public rclcpp::Node
{
public:
  explicit PlannerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  int run();

private:
  ScanConfigLoader config_loader_;
  ScanPlanGenerator generator_;
  ScanPlanWriter plan_writer_;
  std::string config_file_;
  std::string output_file_;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__PLANNER_NODE_HPP_
