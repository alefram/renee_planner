#include "renee_planner/circular_scan_pattern.hpp"

#include <cmath>
#include <stdexcept>

namespace renee_planner
{
namespace
{
constexpr double kTwoPi = 6.28318530717958647692;
}  // namespace

CircularScanPattern::CircularScanPattern(const CircularScanPatternParams & params)
: params_(params)
{
  if (params_.base_radius <= 0.0) {
    throw std::invalid_argument("base_radius must be greater than zero");
  }
  if (params_.end_effector_radius <= 0.0) {
    throw std::invalid_argument("end_effector_radius must be greater than zero");
  }
  if (params_.points_per_ring <= 0) {
    throw std::invalid_argument("points_per_ring must be greater than zero");
  }
}

std::vector<ScanPatternPoint> CircularScanPattern::generatePoints(
  const geometry_msgs::msg::Point & machine_center) const
{
  std::vector<ScanPatternPoint> points;
  points.reserve(static_cast<size_t>(params_.points_per_ring));

  const double delta = kTwoPi / static_cast<double>(params_.points_per_ring);
  const double cos_delta = std::cos(delta);
  const double sin_delta = std::sin(delta);

  double cos_a = std::cos(params_.start_angle_rad);
  double sin_a = std::sin(params_.start_angle_rad);

  for (int point_index = 0; point_index < params_.points_per_ring; ++point_index) {
    ScanPatternPoint point;
    point.base_x = machine_center.x + params_.base_radius * cos_a;
    point.base_y = machine_center.y + params_.base_radius * sin_a;
    point.end_effector_x = machine_center.x + params_.end_effector_radius * cos_a;
    point.end_effector_y = machine_center.y + params_.end_effector_radius * sin_a;
    points.emplace_back(point);

    // rotate cos_a/sin_a by delta to get next angle (complex multiply)
    const double next_cos = cos_a * cos_delta - sin_a * sin_delta;
    const double next_sin = sin_a * cos_delta + cos_a * sin_delta;
    cos_a = next_cos;
    sin_a = next_sin;
  }

  return points;
}

}  // namespace renee_planner
