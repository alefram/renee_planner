# One camera pose of a scan trajectory: where the camera's optical frame
# (+Z forward, +X right, +Y down) has to be, in the trajectory's frame.
uint32 id
string name
# Machine section the pose looks at (extraction, transversal, vertical, wrist, ...).
string section
geometry_msgs/Pose pose
# Surface point the optical axis goes through.
geometry_msgs/Point look_at
# Surface sample points this pose sees.
uint32 covered_points
# Position along the loop around the machine (execution order).
float64 arc_length_m
