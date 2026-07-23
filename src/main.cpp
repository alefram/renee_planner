#include "renee_planner/scan_planner_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<renee_planner::ScanPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
