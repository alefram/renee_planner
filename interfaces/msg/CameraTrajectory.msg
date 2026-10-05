# Ordered camera poses around the machine. header.frame_id is the frame of the
# poses (robot_map); the executor (Behavior Tree) reaches them in order.
std_msgs/Header header
string experiment
# cad | sensor
string mode
string camera_frame
# Seen surface points / all surface points.
float64 coverage_ratio
uint32 surface_points
uint32 uncovered_points
# The YAML written by the generator (config/trajectories/<experiment>.yaml).
string trajectory_file
CameraPose[] poses
