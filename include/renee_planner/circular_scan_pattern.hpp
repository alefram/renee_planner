#ifndef RENEE_PLANNER__CIRCULAR_SCAN_PATTERN_HPP_
#define RENEE_PLANNER__CIRCULAR_SCAN_PATTERN_HPP_

#include "renee_planner/scan_pattern.hpp"

namespace renee_planner
{

struct CircularScanPatternParams
{
  double base_radius{0.0};
  double end_effector_radius{0.0};
  int points_per_ring{0};
  double start_angle_rad{0.0};
};

// The original (and default) pattern: base and end effector each move on
// their own circle of constant radius around the machine, in lockstep, so
// the end effector is always `base_radius - end_effector_radius` in front
// of the base along the same ray from machine_center.
class CircularScanPattern : public ScanPattern
{
public:
  explicit CircularScanPattern(const CircularScanPatternParams & params);

  std::vector<ScanPatternPoint> generatePoints(
    const geometry_msgs::msg::Point & machine_center) const override;

private:
  CircularScanPatternParams params_;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__CIRCULAR_SCAN_PATTERN_HPP_
