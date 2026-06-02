from __future__ import annotations

from typing import Optional

import numpy as np


D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def t2p(T):
    return np.asarray(T)[:3, 3]


def t2r(T):
    return np.asarray(T)[:3, :3]


def t2x(T, aw=0.1):
    T = np.asarray(T)
    return T[:3, 3] + float(aw) * T[:3, 0]


def t2y(T, aw=0.1):
    T = np.asarray(T)
    return T[:3, 3] + float(aw) * T[:3, 1]


def t2z(T, aw=0.1):
    T = np.asarray(T)
    return T[:3, 3] + float(aw) * T[:3, 2]


def pr2t(p=(0, 0, 0), R=np.eye(3)):
    p = np.asarray(p, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rpy2r(rpy, unit="rad"):
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    if unit == "deg":
        rpy = np.deg2rad(rpy)
    elif unit != "rad":
        raise ValueError("[rpy2r] Unknown unit:[%s]" % unit)

    roll, pitch, yaw = rpy
    Cphi, Sphi = np.cos(roll), np.sin(roll)
    Cthe, Sthe = np.cos(pitch), np.sin(pitch)
    Cpsi, Spsi = np.cos(yaw), np.sin(yaw)
    return np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi],
    ])


def get_R_from_twopoints(p_fr, p_to):
    p_fr = np.asarray(p_fr, dtype=np.float64).reshape(3)
    p_to = np.asarray(p_to, dtype=np.float64).reshape(3)
    if np.linalg.norm(p_to - p_fr) < 1e-8:
        return np.eye(3)
    p_a = np.array([1e-10, -1e-10, 1.0])
    p_b = (p_to - p_fr) / np.linalg.norm(p_to - p_fr)
    v = np.cross(p_a, p_b)
    if np.linalg.norm(v) == 0:
        return np.eye(3)
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + S + S @ S * (1 - np.dot(p_a, p_b)) / (np.linalg.norm(v) ** 2)


def r2quat(R):
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0

    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def view_in_local(
        p: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        T_wl: Optional[np.ndarray] = None,
    ):
    return _view_transform(p=p, R=R, T=T, T_wl=T_wl, inverse=True)


def view_in_world(
        p: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        T_wl: Optional[np.ndarray] = None,
    ):
    return _view_transform(p=p, R=R, T=T, T_wl=T_wl, inverse=False)


def _view_transform(p=None, R=None, T=None, T_wl=None, inverse=False):
    if sum(x is not None for x in (p, R, T)) != 1:
        raise ValueError("Exactly one of (p, R, T) must be provided.")
    if T_wl is None:
        raise ValueError("T_wl must be provided.")
    T_wl = np.asarray(T_wl)
    if T_wl.shape != (4, 4):
        raise ValueError("T_wl must be (4,4), got %s" % (str(T_wl.shape)))

    R_wl = T_wl[:3, :3]
    t_wl = T_wl[:3, 3]
    if inverse:
        R_use = R_wl.T
        t_use = -R_use @ t_wl
        T_use = pr2t(t_use, R_use)
    else:
        R_use = R_wl
        t_use = t_wl
        T_use = T_wl

    if p is not None:
        p = np.asarray(p)
        return (p.reshape(-1, 3) @ R_use.T + t_use).reshape(p.shape)
    if R is not None:
        R = np.asarray(R)
        return np.einsum("ij,njk->nik", R_use, R.reshape(-1, 3, 3)).reshape(R.shape)

    T = np.asarray(T)
    return (T_use @ T.reshape(-1, 4, 4)).reshape(T.shape)


def np_uv(vec, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(x)
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n


def rpy2r(rpy,unit="rad"):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix.
    
    Parameters:
        rpy (np.array): Array of [roll, pitch, yaw] angles.
        unit (str): Unit of the input angles ('rad' for radians or 'deg' for degrees).
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    if unit == 'rad':
        rpy_rad = rpy
    elif unit == 'deg':
        rpy_rad = np.deg2rad(rpy)
    else:
        raise ValueError("[rpy2r] Unknown unit:[%s]" % unit)
    
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix with a specified rotation order.
    
    Parameters:
        r0 (np.array): Array of [roll, pitch, yaw] in radians.
        order (list): List specifying the order of rotations (default is [0, 1, 2]).
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    c1 = np.cos(r0[0]); c2 = np.cos(r0[1]); c3 = np.cos(r0[2])
    s1 = np.sin(r0[0]); s2 = np.sin(r0[1]); s3 = np.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
    Convert a 3x3 rotation matrix to roll, pitch, and yaw angles.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        unit (str): Unit for output angles ('rad' for radians or 'deg' for degrees).
        
    Returns:
        np.array: Array of [roll, pitch, yaw] angles.
    """
    roll  = np.atan2(R[2, 1], R[2, 2])
    pitch = np.atan2(-R[2, 0], (np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = np.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out
