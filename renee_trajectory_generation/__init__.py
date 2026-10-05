"""Camera poses (scan trajectories) around the Campetella, generated offline.

Flow (pipeline.py): machine -> surface -> workspace -> candidates ->
visibility -> set_cover -> ordering -> trajectory. These modules do not
import rclpy; the nodes in src/ gather the inputs and call pipeline.generate().
hqp/ is the whole-body HQP model; legacy/ the previous missions (frozen).
"""
