from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import quaternion

AIRCRAFT_GEOMETRY_BODY: Final = {
    "nose": np.array([40.0, 0.0, 0.0]),
    "tail": np.array([-40.0, 0.0, 0.0]),
    "wing_r": np.array([0.0, 44.0, 0.0]),
    "wing_l": np.array([0.0, -44.0, 0.0]),
    "tail_r": np.array([-40.0, 16.0, 0.0]),
    "tail_l": np.array([-40.0, -16.0, 0.0]),
    "tail_v": np.array([-40.0, 0.0, -16.0]),
}

NED_TO_ENU: Final = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1],
])


def rotate_vector_by_quaternion(pose_q, vector) -> np.ndarray:
    """Rotate a vector using quaternion.
    
    Args:
        pose_q: np.quaternion object or [w, x, y, z] array
        vector: 3D vector as array or list
        
    Returns:
        3D vector rotated by quaternion
    """
    # Convert quaternion input if needed
    if not isinstance(pose_q, quaternion.quaternion):
        quat_array = np.asarray(pose_q, dtype=float)
        if quat_array.shape != (4,):
            raise ValueError("Quaternion must have 4 components (w, x, y, z)")
        pose_q = quaternion.quaternion(*quat_array)
    
    # Normalize quaternion
    pose_q_norm = np.abs(pose_q)
    if pose_q_norm < 1e-12:
        pose_q = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    else:
        pose_q = pose_q / pose_q_norm
    
    # Convert vector to numpy array
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("Vector must have 3 components")
    
    # Create pure quaternion from vector (0, x, y, z)
    vector_q = quaternion.quaternion(0.0, *vector)
    
    # Perform rotation: q * v * q_conjugate
    rotated_q = pose_q * vector_q * pose_q.conjugate()
    
    # Extract vector part (ignore scalar part which should be ~0)
    return np.array([rotated_q.x, rotated_q.y, rotated_q.z])