#include "renee_planner/circular_pattern.hpp"
#include "renee_planner/scan_plan_generator.hpp"
#include "renee_planner/scan_types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{

constexpr double kTolerance = 1e-9;
constexpr double kPi = 3.14159265358979323846;

std::shared_ptr<renee_planner::ScanPattern> makeCircularPattern(int points_per_ring = 4)
{
  renee_planner::CircularPatternParams params;
  params.base_radius = 2.0;
  params.end_effector_radius = 1.0;
  params.points_per_ring = points_per_ring;
  params.start_angle_rad = 0.0;
  return std::make_shared<renee_planner::CircularPattern>(params);
}

renee_planner::ScanConfig makeBaseConfig()
{
  renee_planner::ScanConfig config;
  config.global_frame_id = "robot_map";
  config.machine_center.x = 0.0;
  config.machine_center.y = 0.0;
  config.machine_center.z = 0.0;
  config.base_height = 0.0;
  config.pattern = makeCircularPattern();
  config.inspection_viewpoints.push_back(
    renee_planner::InspectionViewpoint{
      "height_0",
      1.0});
  return config;
}

geometry_msgs::msg::Point rotateLocalZAxis(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  const double x = quaternion.x;
  const double y = quaternion.y;
  const double z = quaternion.z;
  const double w = quaternion.w;

  geometry_msgs::msg::Point axis;
  axis.x = 2.0 * (x * z + y * w);
  axis.y = 2.0 * (y * z - x * w);
  axis.z = 1.0 - 2.0 * (x * x + y * y);
  return axis;
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  const double sin_yaw = 2.0 * (
    quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double cos_yaw = 1.0 - 2.0 * (
    quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return std::atan2(sin_yaw, cos_yaw);
}

}  // namespace

TEST(ScanPlanGeneratorTest, GeneratesOneWaypointPerHeightAndPoint)
{
  const auto config = makeBaseConfig();
  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  EXPECT_EQ(plan.frame_id, "robot_map");
  EXPECT_NEAR(plan.machine_center.x, 0.0, kTolerance);
  EXPECT_NEAR(plan.machine_center.y, 0.0, kTolerance);
  EXPECT_NEAR(plan.machine_center.z, 0.0, kTolerance);
  ASSERT_EQ(plan.waypoints.size(), 4u);
  EXPECT_EQ(plan.waypoints.front().id, "height_0_0");
}

TEST(ScanPlanGeneratorTest, PreservesGlobalFrameInGeneratedPoses)
{
  const auto config = makeBaseConfig();
  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  ASSERT_FALSE(plan.waypoints.empty());
  EXPECT_EQ(plan.waypoints.front().base_pose.header.frame_id, "robot_map");
  EXPECT_EQ(plan.waypoints.front().end_effector_pose.header.frame_id, "robot_map");
}

TEST(ScanPlanGeneratorTest, GeneratesCircularBaseAndEndEffectorPositions)
{
  const auto config = makeBaseConfig();
  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  ASSERT_EQ(plan.waypoints.size(), 4u);

  const auto & first = plan.waypoints.at(0);
  EXPECT_NEAR(first.base_pose.pose.position.x, 2.0, kTolerance);
  EXPECT_NEAR(first.base_pose.pose.position.y, 0.0, kTolerance);
  EXPECT_NEAR(first.base_pose.pose.position.z, 0.0, kTolerance);
  EXPECT_NEAR(first.end_effector_pose.pose.position.x, 1.0, kTolerance);
  EXPECT_NEAR(first.end_effector_pose.pose.position.y, 0.0, kTolerance);
  EXPECT_NEAR(first.end_effector_pose.pose.position.z, 1.0, kTolerance);

  const auto & second = plan.waypoints.at(1);
  EXPECT_NEAR(second.base_pose.pose.position.x, 0.0, kTolerance);
  EXPECT_NEAR(second.base_pose.pose.position.y, 2.0, kTolerance);
  EXPECT_NEAR(second.end_effector_pose.pose.position.x, 0.0, kTolerance);
  EXPECT_NEAR(second.end_effector_pose.pose.position.y, 1.0, kTolerance);
}

TEST(ScanPlanGeneratorTest, GeneratesExpectedRobotMapPoseAtNinetyDegrees)
{
  auto config = makeBaseConfig();
  config.machine_center.x = -2.7352;
  config.machine_center.y = -3.4334;
  config.inspection_viewpoints.front().height = 0.9;

  renee_planner::CircularPatternParams params;
  params.base_radius = 3.0;
  params.end_effector_radius = 2.6;
  params.points_per_ring = 1;
  params.start_angle_rad = kPi / 2.0;
  config.pattern = std::make_shared<renee_planner::CircularPattern>(params);

  const renee_planner::ScanPlanGenerator generator;
  const auto plan = generator.generateInspectionPath(config);

  ASSERT_EQ(plan.waypoints.size(), 1u);
  const auto & waypoint = plan.waypoints.front();
  EXPECT_EQ(waypoint.base_pose.header.frame_id, "robot_map");
  EXPECT_NEAR(waypoint.base_pose.pose.position.x, -2.7352, 1e-6);
  EXPECT_NEAR(waypoint.base_pose.pose.position.y, -0.4334, 1e-6);
  EXPECT_NEAR(yawFromQuaternion(waypoint.base_pose.pose.orientation), -kPi / 2.0, 1e-6);
  EXPECT_NEAR(waypoint.end_effector_pose.pose.position.x, -2.7352, 1e-6);
  EXPECT_NEAR(waypoint.end_effector_pose.pose.position.y, -0.8334, 1e-6);
  EXPECT_NEAR(waypoint.end_effector_pose.pose.position.z, 0.9, 1e-6);
}

TEST(ScanPlanGeneratorTest, GeneratesEightCircularStations)
{
  auto config = makeBaseConfig();
  config.pattern = makeCircularPattern(/*points_per_ring=*/8);
  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  ASSERT_EQ(plan.waypoints.size(), 8u);
  for (const auto & waypoint : plan.waypoints) {
    EXPECT_EQ(waypoint.base_pose.header.frame_id, "robot_map");
    EXPECT_EQ(waypoint.end_effector_pose.header.frame_id, "robot_map");
  }
}

TEST(ScanPlanGeneratorTest, OrientsEndEffectorHorizontallyTowardMachineCenter)
{
  const auto config = makeBaseConfig();
  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  ASSERT_FALSE(plan.waypoints.empty());
  const auto & waypoint = plan.waypoints.front();
  // tool0 convention (ur_macro.xacro flange-tool0 joint): Z+ is front, so the
  // local Z axis must be the one aimed at the machine center.
  const auto forward_axis = rotateLocalZAxis(waypoint.end_effector_pose.pose.orientation);

  EXPECT_NEAR(forward_axis.x, -1.0, 1e-6);
  EXPECT_NEAR(forward_axis.y, 0.0, 1e-6);
  EXPECT_NEAR(forward_axis.z, 0.0, 1e-6);
}

TEST(ScanPlanGeneratorTest, RejectsFrameOtherThanRobotMap)
{
  auto config = makeBaseConfig();
  config.global_frame_id = "world";
  const renee_planner::ScanPlanGenerator generator;

  EXPECT_THROW(generator.generateInspectionPath(config), std::invalid_argument);
}

TEST(ScanPlanGeneratorTest, GeneratesWaypointsForMultipleHeights)
{
  auto config = makeBaseConfig();
  config.pattern = makeCircularPattern(/*points_per_ring=*/2);
  config.inspection_viewpoints.push_back(
    renee_planner::InspectionViewpoint{"height_1", 1.5});

  const renee_planner::ScanPlanGenerator generator;

  const auto plan = generator.generateInspectionPath(config);

  ASSERT_EQ(plan.waypoints.size(), 4u);
  // Height is the inner loop: both heights at point 0 come before point 1,
  // so the base only has to visit each ring position once.
  EXPECT_EQ(plan.waypoints.at(0).id, "height_0_0");
  EXPECT_EQ(plan.waypoints.at(1).id, "height_1_0");
  EXPECT_EQ(plan.waypoints.at(2).id, "height_0_1");
  EXPECT_EQ(plan.waypoints.at(3).id, "height_1_1");
  EXPECT_NEAR(plan.waypoints.at(1).end_effector_pose.pose.position.z, 1.5, kTolerance);

  EXPECT_NEAR(
    plan.waypoints.at(0).base_pose.pose.position.x,
    plan.waypoints.at(1).base_pose.pose.position.x, kTolerance);
  EXPECT_NEAR(
    plan.waypoints.at(0).base_pose.pose.position.y,
    plan.waypoints.at(1).base_pose.pose.position.y, kTolerance);
}

TEST(ScanPlanGeneratorTest, ThrowsWhenPatternMissing)
{
  auto config = makeBaseConfig();
  config.pattern.reset();
  const renee_planner::ScanPlanGenerator generator;

  EXPECT_THROW(generator.generateInspectionPath(config), std::invalid_argument);
}

TEST(CircularPatternTest, ThrowsOnInvalidParams)
{
  renee_planner::CircularPatternParams params;
  params.base_radius = 2.0;
  params.end_effector_radius = 1.0;
  params.points_per_ring = 0;
  params.start_angle_rad = 0.0;

  EXPECT_THROW(renee_planner::CircularPattern{params}, std::invalid_argument);
}
