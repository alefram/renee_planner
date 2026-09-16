"""Read a static URDF scene and sample actual surfaces without a ROS node."""

from pathlib import Path
import subprocess
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def resolve_path(value, directory):
    """Resolve file/package URIs or paths relative to a configuration file."""
    value = str(value)
    if value.startswith('package://'):
        from ament_index_python.packages import get_package_share_directory
        package, relative = value[len('package://'):].split('/', 1)
        path = Path(get_package_share_directory(package)) / relative
    else:
        path = Path(value.removeprefix('file://')).expanduser()
        if not path.is_absolute():
            path = Path(directory) / path
    if not path.is_file():
        raise ValueError(f'File does not exist: {path}')
    return path.resolve()


def read_urdf(path, arguments=()):
    """Expand Xacro if necessary, resolving mesh paths to absolute filenames."""
    path = Path(path)
    if path.suffix == '.xacro':
        try:
            result = subprocess.run(['xacro', str(path), *arguments], check=True,
                                    text=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError(f'Cannot expand {path}: '
                             f'{getattr(error, "stderr", str(error))}') from error
        root = ET.fromstring(result.stdout)
    else:
        root = ET.parse(path).getroot()
    for mesh in root.iter('mesh'):
        mesh.set('filename', str(resolve_path(mesh.get('filename'), path.parent)))
    return root


def pose_matrix(pose):
    """Convert [x, y, z, roll, pitch, yaw] in metres/radians to SE(3)."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError('Pose must contain six finite values: x y z roll pitch yaw')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _origin(node):
    origin = node.find('origin')
    if origin is None:
        return np.eye(4)
    return pose_matrix([float(v) for key in ('xyz', 'rpy')
                        for v in origin.get(key, '0 0 0').split()])


def _shape(node):
    import trimesh

    geometry = node.find('geometry')
    if geometry is None:
        raise ValueError('URDF visual/collision has no geometry')
    mesh = geometry.find('mesh')
    if mesh is not None:
        result = trimesh.load(mesh.get('filename'), force='mesh', process=False)
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError('Mesh scale must have three positive components')
        result.apply_scale(scale)
        return result
    box = geometry.find('box')
    if box is not None:
        return trimesh.creation.box(extents=np.fromstring(box.get('size'), sep=' '))
    cylinder = geometry.find('cylinder')
    if cylinder is not None:
        return trimesh.creation.cylinder(radius=float(cylinder.get('radius')),
                                         height=float(cylinder.get('length')), sections=32)
    sphere = geometry.find('sphere')
    if sphere is not None:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(sphere.get('radius')))
    raise ValueError('Unsupported URDF geometry')


def load_structure(path, pose, arguments=()):
    """Load transformed collision and visual meshes of a fixed URDF assembly.

    Returns
    -------
    dict
        Local bounds, world visual mesh, world collision meshes and model pose.
        Bounds are conservative unions of collision and visual geometry.

    Raises
    ------
    ValueError
        Moving joints, disconnected/cyclic links or missing geometry.
    """
    import trimesh

    root = read_urdf(path, arguments)
    links = {link.get('name'): link for link in root.findall('link')}
    children, joints = set(), {}
    for joint in root.findall('joint'):
        if joint.get('type') != 'fixed':
            raise ValueError('Campetella must be a static URDF with fixed joints')
        parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
        children.add(child)
        joints.setdefault(parent, []).append((child, _origin(joint)))
    roots = set(links) - children
    if len(roots) != 1:
        raise ValueError('Structure must have exactly one root link')
    visual, collision, visited = [], [], set()

    def visit(name, transform):
        if name in visited or name not in links:
            raise ValueError('Cyclic or invalid URDF link graph')
        visited.add(name)
        for tag, output in [('visual', visual), ('collision', collision)]:
            for node in links[name].findall(tag):
                mesh = _shape(node)
                mesh.apply_transform(transform @ _origin(node))
                if not len(mesh.faces):
                    raise ValueError(f'Empty {tag} geometry in {name}')
                output.append(mesh)
        for child, offset in joints.get(name, []):
            visit(child, transform @ offset)

    visit(next(iter(roots)), np.eye(4))
    if len(visited) != len(links) or not visual or not collision:
        raise ValueError('Structure needs connected visual and collision geometry')
    bounds = trimesh.util.concatenate(visual + collision).bounds
    transform = pose_matrix(pose)
    for mesh in visual + collision:
        mesh.apply_transform(transform)
    return {'bounds': bounds, 'transform': transform,
            'visual': trimesh.util.concatenate(visual), 'collision': collision}


def surface_views(structure, options):
    """Cast inward rays on four lateral faces; return persistent target IDs.

    Heights are measured from the bottom of the model envelope. Ray hits use
    the actual visual mesh; absent hits are reported rather than fabricated.
    """
    bounds, transform = structure['bounds'], structure['transform']
    low, high = bounds
    views, missed = [], []
    count = options['samples_per_face']
    for face, (axis, sign) in enumerate(((0, 1), (1, 1), (0, -1), (1, -1))):
        tangent = 1 - axis
        for band, height in enumerate(options['heights_from_bottom_m']):
            if not 0 < height < high[2] - low[2]:
                raise ValueError('Scan height must lie within the model height')
            fractions = (np.arange(count) + 0.5) / count
            if face in (1, 2):
                fractions = fractions[::-1]
            for sample, fraction in enumerate(fractions):
                target_id = f'face_{face}_band_{band}_sample_{sample}'
                local = (low + high) / 2
                local[axis] = (high[axis] if sign > 0 else low[axis]) + sign
                local[tangent] = low[tangent] + fraction * (high[tangent] - low[tangent])
                local[2] = low[2] + height
                normal = transform[:3, :3][:, axis] * sign
                origin = transform[:3, :3] @ local + transform[:3, 3]
                hits, _, _ = structure['visual'].ray.intersects_location(
                    [origin], [-normal], multiple_hits=True)
                if not len(hits):
                    missed.append(target_id)
                    continue
                target = hits[np.argmin(np.linalg.norm(hits - origin, axis=1))]
                views.append({'id': target_id, 'face': face, 'band': band,
                              'target': target, 'normal': normal})
    if missed:
        warnings.warn(f'{len(missed)} surface rays missed geometry', stacklevel=2)
    return views, missed


def target_visible(mesh, camera, target, tolerance=0.002):
    """Check that the first surface intersection is the requested target."""
    offset = target - camera
    length = np.linalg.norm(offset)
    if length <= tolerance:
        return False
    hits, _, _ = mesh.ray.intersects_location([camera], [offset / length], multiple_hits=True)
    if not len(hits):
        return False
    distances = (hits - camera) @ (offset / length)
    distances = distances[distances >= 0]
    return bool(len(distances) and abs(np.min(distances) - length) <= tolerance)
