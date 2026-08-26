# renee_planner

ROS 2 package that creates inspection waypoints around a structure and writes them as YAML.

## Plotting a plan with structure geometry

`plot_scan_waypoints.py` can draw a generic basic form of the inspected structure from the collision primitives in a URDF or Xacro. It supports `box`, `cylinder`, and `sphere` collision geometry, follows fixed joint transforms, and intentionally skips mesh collisions so that the plot stays lightweight.

The model origin must be supplied separately because `machine_center` in a plan can be the structure's geometric center rather than its URDF root frame.

```bash
ros2 run renee_planner plot_scan_waypoints.py /tmp/renee_scan_plan.yaml \
  --structure-model /path/to/structure.urdf \
  --structure-pose X Y Z YAW \
  --save /tmp/renee_scan_plan.png
```

For configurable Xacro models, provide each activation or scale setting as a repeated `--xacro-arg`:

```bash
ros2 run renee_planner plot_scan_waypoints.py /tmp/renee_scan_plan.yaml \
  --structure-model /fnh_pkgs/src/renee_simulation/campetella_sim/models/campetella_CRC/urdf/campetella.urdf.xacro \
  --structure-pose -2.0 -3.0 0.2 0.0 \
  --xacro-arg robot_scale:=0.001 \
  --xacro-arg mode:=navigation
```

If the model contains only mesh collisions, create/use simple primitive collision geometry before plotting it. Running the plot without `--structure-model` remains valid and displays only the plan, robot, floor, and walls.
