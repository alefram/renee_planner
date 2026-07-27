#include "renee_planner/planner_node.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<renee_planner::PlannerNode>();
  const int result = node->run();
  rclcpp::shutdown();
  return result;
}
