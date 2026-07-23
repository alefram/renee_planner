#ifndef RENEE_PLANNER__SCAN_PATTERN_HPP_
#define RENEE_PLANNER__SCAN_PATTERN_HPP_

#include <geometry_msgs/msg/point.hpp>

#include <vector>

namespace renee_planner
{

// A single (base, end-effector) planar position pair produced by a
// ScanPattern, in the same frame as machine_center. Z is filled in
// separately by the generator (base_height for the base, each inspection
// viewpoint's height for the end effector) since height tiers are shared
// across every pattern.
struct ScanPatternPoint
{
  double base_x{0.0};
  double base_y{0.0};
  double end_effector_x{0.0};
  double end_effector_y{0.0};
};

// Strategy interface for the planar (x, y) path scanned around a machine.
// To add a new trajectory shape: implement this interface (see
// CircularScanPattern for an example), then teach ScanConfigLoader how to
// build it from a `scan.pattern_type` value in the YAML config. Nothing
// else in renee_planner needs to change -- ScanPlanGenerator only ever
// talks to a ScanConfig's `pattern` through this interface.
class ScanPattern
{
public:
  virtual ~ScanPattern() = default;

  virtual std::vector<ScanPatternPoint> generatePoints(
    const geometry_msgs::msg::Point & machine_center) const = 0;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__SCAN_PATTERN_HPP_
