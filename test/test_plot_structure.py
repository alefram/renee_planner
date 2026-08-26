#!/usr/bin/env python3
"""Regression tests for generic URDF primitive loading used by the plot tool."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "plot_scan_waypoints.py"
SPEC = importlib.util.spec_from_file_location("plot_scan_waypoints", MODULE_PATH)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class StructurePrimitiveTest(unittest.TestCase):
    def write_model(self, text):
        directory = tempfile.TemporaryDirectory()
        model = Path(directory.name) / "fixture.urdf"
        model.write_text(text, encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return model

    def test_loads_primitives_with_link_and_structure_transforms(self):
        model = self.write_model("""<robot name='fixture'>
          <link name='base'><collision><origin xyz='1 0 0'/><geometry><box size='2 1 1'/></geometry></collision></link>
          <link name='tool'><collision><geometry><cylinder radius='0.2' length='1.0'/></geometry></collision></link>
          <joint name='fixed' type='fixed'><parent link='base'/><child link='tool'/><origin xyz='0 2 0' rpy='0 0 0'/></joint>
        </robot>""")
        primitives = PLOT.load_structure_primitives(model, (3.0, 4.0, 0.0, 0.0))
        self.assertEqual([item["kind"] for item in primitives], ["box", "cylinder"])
        self.assertAlmostEqual(primitives[0]["transform"][0, 3], 4.0)
        self.assertAlmostEqual(primitives[1]["transform"][0, 3], 3.0)
        self.assertAlmostEqual(primitives[1]["transform"][1, 3], 6.0)

    def test_rejects_model_without_primitive_collisions(self):
        model = self.write_model("""<robot name='fixture'><link name='base'>
          <collision><geometry><mesh filename='missing.stl'/></geometry></collision>
        </link></robot>""")
        with self.assertRaisesRegex(ValueError, "mesh collisions are not rendered"):
            PLOT.load_structure_primitives(model, (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
