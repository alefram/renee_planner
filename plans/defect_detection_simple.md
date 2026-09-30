# Defect detection simple: Nav2 para la base, MoveIt para el brazo

## Objetivo
Por cada target del YAML (`position` + `look_at` de la cámara):
1. elegir la pose de base;
2. Nav2 lleva la base (boat mode);
3. approach lateral solo si hace falta;
4. MoveIt lleva el brazo.

Sin HQP, sin `BasePlanner` y sin sub-fases anidadas.

## Lo que NO cambia
- Launch: `ros2 launch renee_trajectory_generation trajectory_controller.launch.py experiment:=defect_detection_sim`.
- Nodo: `src/trajectory_controller_node.py`, con su interfaz `io` tal cual (`follow_path`, `publish_twist`, `request_ik`, `move_arm`, `apply_collision_objects`, `base_in_map`, `hold_arm`).
- YAML: `config/experiments/defect_detection_sim.yaml`.
- Cleaning, screw_detection, `Mission` y el HQP.

## Cambios

### 1. `renee_trajectory_generation/navigation.py`: tres reglas, un método cada una
- **`LoopPath.boat_path(start_xy, goal_xy)`**: el tramo del loop hacia adelante (CCW) para `FollowPath`. Envuelve `segment()`: nunca giros en sitio y siempre hacia adelante.
- **`footprint_clearance(x, y, yaw, boxes, footprint)` y `clearance_ok(pose, boxes, min_clearance_m=0.30)`**: la distancia mínima (vista cenital) del footprint 1.2×0.7 a las cajas de la Campetella. Extraer la geometría de `BasePlanner.footprint_clear` (`navigation.py:326`) y hacer que `BasePlanner` la reutilice.
- **`lateral_approach_twist(base_xy_yaw, lane_pose, offset_m, boxes, cfg)`**: el approach omnidireccional hacia la máquina con el heading del carril fijo. Usa `gate_twist` (`navigation.py:615`) y devuelve `None` (para) si el siguiente paso rompe `clearance_ok`.
- **`base_candidates(target, loop, boxes, cfg)`**: la muestra del loop más cercana a la proyección XY del target, ±`window_m` a lo largo del loop y fuera de las esquinas (`min_straight_before_stop_m`). Primero todas con offset 0; los offsets laterales `offsets_m` solo después, como fallback. Todo filtrado por `clearance_ok`.
- **`loop_corners_from_boxes(...)`**: el loop se genera desde el límite de 0.30 m (ver *Loop generado desde el límite*).

### 2. `missions/defect_detection.py`: misión lineal sin HQP
`DefectDetectionMission` ya no hereda de `Mission`, cuyo `__init__` construye Pinocchio y el HQP. Pasa a ser una clase propia con `tick(io)` y una lista plana de pasos por target:
1. **choose_base**: para cada `base_candidates(...)`, `io.request_ik` con esa base. El primero con IK válido da `(lane_pose, offset, q_arm)`. Si no hay ninguno, el target queda como `unreachable` y se pasa al siguiente.
2. **stow**: `io.move_arm(travel_arm_q)`, solo si la base tiene que moverse.
3. **back_to_lane**: si el offset actual es > 0, `lateral_approach_twist` hasta offset 0.
4. **drive**: `io.follow_path(loop.boat_path(base, lane_pose))`.
5. **approach**: solo si `offset > 0`, con `lateral_approach_twist`.
6. **reach**: re-IK desde la pose real de la base (seed `q_arm`), después `io.move_arm(q)` (Pilz PTP, con fallback OMPL como ahora) y `io.hold_arm`.
7. **record**: `reached / unreachable / nav_failed / arm_failed` y el error de pose de cámara vía TF.

Reutilizar `ArmMotion` (`arm_motion.py`), `tasks.camera_orientation` y `recorders.save_results`.

### 3. `mission.py` / nodo
- `create_mission` (`mission.py:97`) devuelve la nueva `DefectDetectionMission`.
- En el nodo, que la construcción del robot Pinocchio y el HQP ocurra solo dentro de `Mission`, para que defect detection no necesite el venv.

### 4. `config/experiments/defect_detection_sim.yaml`
- Quitar las secciones del HQP (tasks, levels, gains) y `refine`.
- En `base_placement`, añadir `window_m`. `offsets_m` queda solo como fallback, acotado por `lane_clearance_m - min_clearance_m`.
- En `navigation`, quitar `loop_path.corners` y `loop_path.radius`, y añadir `min_clearance_m`, `lane_clearance_m`, `tracking_margin_m` y `max_radius_m` (ver abajo).
- Mantener `moveit`, `collision_from_description`, `travel_arm_q` y `targets`.

## Loop generado desde el límite (sin `corners`)
Prioridad: la base nunca a menos de `min_clearance_m` (0.30 m) de la Campetella. El loop se calcula a partir de las cajas de colisión que el nodo ya lee (`collision_from_description`) y deja de escribirse a mano.

**Geometría:** el loop es el rectángulo que envuelve (vista cenital) todas las cajas, agrandado en cada lado una distancia `d`:
```
d = lane_clearance_m + footprint_width/2 + tracking_margin_m
```
- Carriles rectos: el lado del footprint que mira a la máquina queda a `lane_clearance_m + tracking_margin_m`.
- Curvas: si el radio es como mucho `d`, el arco queda fuera del círculo de radio `d` alrededor de la esquina de la máquina, y las esquinas del footprint quedan más lejos que su lado interior.
- `LoopPath(corners, radius)` no cambia: las esquinas del rectángulo agrandado con `radius = min(d, max_radius_m)` generan ese contorno.
- Es un rectángulo y no un contorno ajustado porque en boat mode (CCW, solo giros a la izquierda) el loop tiene que ser convexo.

**`navigation.py`:**
- `loop_corners_from_boxes(boxes, clearance_m, footprint, margin_m)` → `(corners CCW, d)`.
- Al construir el loop:
  - `clearance_ok` sobre todas las muestras; si alguna falla, error.
  - Comprobar que el loop cabe dentro de `navigation.workspace`, restando el semiancho del robot y un margen de pared. Si no cabe, el nodo no arranca y dice cuántos cm faltan (ese caso no tiene solución que cumpla el límite).

**Dos distancias:**
- `lane_clearance_m` (p. ej. 0.40): distancia del carril a la máquina.
- `min_clearance_m` (0.30): límite duro.
- El approach lateral solo puede consumir `lane_clearance_m - min_clearance_m` (unos 10 cm con 0.40 / 0.30). Con las dos iguales, no hay approach.

**Garantía en ejecución:**
- En cada `tick`, mientras la base conduce (FollowPath) o hace el approach, `clearance_ok` sobre la pose real (TF).
- Si se viola: cancelar el FollowPath, publicar un twist cero y marcar el target como `clearance_violation`.
- `tracking_margin_m` se ajusta con el error de seguimiento de DWB medido en los bags (`/robot/local_plan`, odometría).

**Chequeo con las cajas de la corrida `20260929_121636`** (footprint 1.2×0.7, límite 0.30 m):
- El loop actual cumple: su mínimo es 0.377 m, en los carros de soporte.
- Loop generado con `lane_clearance_m` 0.35 y `tracking_margin_m` 0.05: esquinas (-0.91,-1.68) (-5.87,-1.68) (-5.87,-5.02) (-0.91,-5.02) y mínimo medido de 0.40 m. Deja una holgura de al menos 0.43 m hasta `workspace`.
- Las paradas antiguas con desplazamiento lateral no cumplen: 0.11, 0.125 y 0.18 m en el lado sur (`extraction`/`transversal_end`), y 0.283 m en dos paradas del carril norte. Los targets de esas paradas son los que hay que revisar con el IK.

**YAML (`navigation`):**
- Se quita `loop_path.corners` y `loop_path.radius`.
- Se añaden `min_clearance_m: 0.30`, `lane_clearance_m: 0.40`, `tracking_margin_m: 0.05` y `max_radius_m`.

## Fuera de alcance, para después
Pausa de SLAM y jump guard, `base_sync`, validity checks a 20 Hz, `arm_between_stops: section` y capture (desactivado de todas formas).

