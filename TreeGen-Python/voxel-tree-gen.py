#!/usr/bin/env python3
"""
PyQt5 + OpenGL procedural voxel tree generator & viewer (Space Colonization)

New in this version
- **Trunk taper controls**: base thickness, tip thickness, and taper power along the main trunk path.
- **Up-then-out growth**: canopy points sampled from a rising-profile canopy (radius increases with Z), with Z-density bias.
- **Branching control**: adjustable branching amount, threshold, and angular spread for multi-child splits.
- Keeps: discrete cross-section stamps; 3D Bresenham raster; Lambert lighting.

Dependencies
    pip install PyQt5 PyOpenGL numpy

Run
    python voxel_tree_app.py

Notes
- Volume is 64^3 (Z up). Colors stored as RGBA8 per voxel.
- Mesher emits face normals so lighting separates faces nicely.
"""
from __future__ import annotations
import sys
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QOpenGLWidget,
    QSplitter,
    QFrame,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QLabel,
)
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtCore import Qt

from OpenGL.GL import *


# ----------------------------- Voxel Volume ----------------------------- #

@dataclass
class VoxelVolume:
    size: Tuple[int, int, int] = (64, 64, 64)  # (X, Y, Z), Z-up

    def __post_init__(self):
        sx, sy, sz = self.size
        self.occ = np.zeros((sx, sy, sz), dtype=np.bool_)      # occupancy
        self.col = np.zeros((sx, sy, sz, 4), dtype=np.uint8)   # RGBA8 per voxel

    def clear(self):
        self.occ.fill(False)
        self.col.fill(0)

    # -- Space Colonization tree generation and voxelization -- #
    def generate_tree_colonization(
        self,
        params: Dict,
        trunk_color: Tuple[int, int, int, int] = (139, 69, 19, 255),  # saddle brown
    ):
        """Generate a semi-realistic tree using Space Colonization and voxelize it.

        Parameters in `params` (voxel units unless noted):
            trunk_height: int – canopy start Z (base of canopy)
            canopy_radius: int – maximum horizontal radius at canopy top
            canopy_height: int – canopy vertical height above trunk_height
            canopy_profile_power: float – radius(z) ∝ (z/h)^power
            z_bias_power: float – bias more attraction points toward canopy top (bigger => more top-heavy)
            point_count: int – number of attraction points
            step: float – growth step length
            r_attr: float – attraction radius
            r_kill: float – kill radius (consume points)
            w_up: float – gravitropism toward +Z
            w_light: float – phototropism weight
            max_iters: int – iteration cap
            base_trunk_thickness: float – trunk thickness at base (1.0..4.0)
            trunk_tip_thickness: float – trunk thickness at canopy (1.0..base)
            trunk_taper_power: float – shapes taper curve from base→tip along trunk path
            branching_amount: float – 0..1, more children when influenced heavily
            branch_threshold: int – number of influences to start spawning extra children
            angle_spread_deg: float – degrees of divergence for split children
        """
        sx, sy, sz = self.size
        cx, cy = sx // 2, sy // 2

        trunk_height = int(params.get('trunk_height', 28))
        canopy_radius = int(params.get('canopy_radius', 16))
        canopy_height = int(params.get('canopy_height', canopy_radius))
        canopy_profile_power = float(params.get('canopy_profile_power', 2.0))
        z_bias_power = float(params.get('z_bias_power', 1.5))
        point_count = int(params.get('point_count', 1000))
        step = float(params.get('step', 3))
        r_attr = float(params.get('r_attr', 18))
        r_kill = float(params.get('r_kill', 6))
        w_up = float(params.get('w_up', 0.6))
        w_light = float(params.get('w_light', 0.25))
        max_iters = int(params.get('max_iters', 600))
        base_trunk_thickness = float(params.get('base_trunk_thickness', 3.5))
        trunk_tip_thickness = float(params.get('trunk_tip_thickness', 1.5))
        trunk_taper_power = float(params.get('trunk_taper_power', 1.4))
        branching_amount = float(params.get('branching_amount', 0.5))
        branch_threshold = int(params.get('branch_threshold', 4))
        angle_spread_deg = float(params.get('angle_spread_deg', 35.0))

        base_trunk_thickness = float(np.clip(base_trunk_thickness, 1.0, 4.0))
        trunk_tip_thickness = float(np.clip(trunk_tip_thickness, 1.0, base_trunk_thickness))
        canopy_radius = int(np.clip(canopy_radius, 3, min(sx, sy) // 2))
        canopy_height = int(np.clip(canopy_height, 4, max(4, sz - 2 - trunk_height)))
        angle_spread = math.radians(np.clip(angle_spread_deg, 0.0, 90.0))

        # Light direction for phototropism (world-space)
        L = np.array([0.5, 0.2, 1.0], dtype=np.float32)
        L = L / (np.linalg.norm(L) + 1e-8)

        # Prepare volume
        self.clear()

        # Sample attraction points in a rising-profile canopy above trunk_height
        # z in [0, canopy_height], radius(z) = canopy_radius * (z/h)^power
        rng = np.random.default_rng(12345)
        pts = []
        for _ in range(point_count):
            # bias z toward top: z_rel = h * u^(1/z_bias_power)
            u = rng.random()
            z_rel = canopy_height * (u ** (1.0 / max(1e-6, z_bias_power)))
            # radial sample inside disk of radius R(z)
            Rz = canopy_radius * ((z_rel / canopy_height) ** max(0.0, canopy_profile_power))
            a = rng.uniform(0.0, 2.0 * math.pi)
            r_disk = math.sqrt(rng.random()) * Rz
            px = cx + 0.5 + r_disk * math.cos(a)
            py = cy + 0.5 + r_disk * math.sin(a)
            pz = trunk_height + 0.5 + z_rel
            if 1 <= px < sx - 1 and 1 <= py < sy - 1 and 1 <= pz < sz - 1:
                pts.append(np.array([px, py, pz], dtype=np.float32))
        pts = np.array(pts, dtype=np.float32)

        # Node graph
        nodes: List[np.ndarray] = [np.array([cx + 0.5, cy + 0.5, 0.5], dtype=np.float32)]  # root at ground center
        parents: List[int] = [-1]

        # Growth loop
        r_attr2 = r_attr * r_attr
        r_kill2 = r_kill * r_kill
        up = np.array([0, 0, 1], dtype=np.float32)

        for _ in range(max_iters):
            if len(pts) == 0:
                break
            N = len(nodes)
            acc = np.zeros((N, 3), dtype=np.float32)
            cnt = np.zeros((N,), dtype=np.int32)

            node_pos = np.vstack(nodes)  # (N,3)
            # naive nearest (OK for small N)
            for i in range(len(pts)):
                dp = node_pos - pts[i]
                d2 = np.sum(dp * dp, axis=1)
                j = int(np.argmin(d2))
                if d2[j] <= r_attr2:
                    v = pts[i] - node_pos[j]
                    n = np.linalg.norm(v)
                    if n > 1e-6:
                        acc[j] += v / n
                        cnt[j] += 1

            new_positions: List[np.ndarray] = []
            new_parents: List[int] = []

            for j in range(N):
                if cnt[j] == 0:
                    continue
                base_dir = acc[j] + w_up * up + w_light * L + 0.05 * rng.normal(0, 1, size=3)
                n = np.linalg.norm(base_dir)
                if n < 1e-6:
                    continue
                dir_main = base_dir / n

                # Determine number of children based on influence count
                excess = max(0, int(cnt[j]) - int(branch_threshold))
                extra = 0
                if branching_amount > 0 and excess > 0:
                    extra = min(2, int(round(branching_amount * (excess / max(1, branch_threshold)))))
                num_children = 1 + extra

                # Build orthonormal basis around dir_main
                perp1 = np.cross(dir_main, up)
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.cross(dir_main, np.array([1.0, 0.0, 0.0], dtype=np.float32))
                perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
                perp2 = np.cross(dir_main, perp1)
                perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)

                child_dirs: List[np.ndarray] = []
                if num_children == 1:
                    child_dirs = [dir_main]
                elif num_children == 2:
                    c = math.cos(angle_spread)
                    s = math.sin(angle_spread)
                    child_dirs = [normalize(dir_main * c + perp1 * s),
                                  normalize(dir_main * c - perp1 * s)]
                else:  # 3 children
                    c = math.cos(angle_spread)
                    s = math.sin(angle_spread)
                    child_dirs = [dir_main,
                                  normalize(dir_main * c + perp1 * s),
                                  normalize(dir_main * c - perp1 * s)]

                for dvec in child_dirs:
                    pos = node_pos[j] + step * dvec
                    pos = np.clip(pos, [1.5, 1.5, 1.5], [sx - 1.5, sy - 1.5, sz - 1.5])
                    new_positions.append(pos.astype(np.float32))
                    new_parents.append(j)

            if not new_positions:
                break

            for k, pos in enumerate(new_positions):
                nodes.append(pos)
                parents.append(new_parents[k])

            # Remove points close to any new node
            newp = np.vstack(new_positions)
            keep_mask = np.ones((len(pts),), dtype=bool)
            for i in range(len(pts)):
                d2 = np.sum((newp - pts[i]) ** 2, axis=1)
                if np.min(d2) <= r_kill2:
                    keep_mask[i] = False
            pts = pts[keep_mask]

        # --- Children lists & subtree area (Leonardo's rule) ---
        n_nodes = len(nodes)
        children: List[List[int]] = [[] for _ in range(n_nodes)]
        for i in range(1, n_nodes):
            p = parents[i]
            children[p].append(i)

        area = np.zeros((n_nodes,), dtype=np.float32)
        for i in range(n_nodes - 1, -1, -1):
            if len(children[i]) == 0:
                area[i] = 1.0
            else:
                ssum = 0.0
                for ch in children[i]:
                    ssum += area[ch]
                area[i] = max(1.0, ssum)
        diam_f = np.sqrt(area)  # continuous diameter proxy, >= 1

        # Identify main trunk path: follow the largest-area child from root
        is_trunk = np.zeros((n_nodes,), dtype=bool)
        trunk_chain: List[int] = [0]
        cur = 0
        while len(children[cur]) > 0:
            ch_list = children[cur]
            best = ch_list[0]
            best_a = area[best]
            for ch in ch_list[1:]:
                if area[ch] > best_a:
                    best, best_a = ch, area[ch]
            is_trunk[best] = True
            trunk_chain.append(best)
            cur = best
        is_trunk[0] = True

        # normalized position along trunk chain
        trunk_s = np.zeros((n_nodes,), dtype=np.float32)
        if len(trunk_chain) > 1:
            for idx, node_idx in enumerate(trunk_chain):
                trunk_s[node_idx] = idx / float(len(trunk_chain) - 1)

        # --- Thickness mapping ---
        # a) base area-driven map for *all* nodes (for branches)
        d_min = float(np.min(diam_f))
        d_max = float(np.max(diam_f))
        denom = max(1e-8, d_max - d_min)
        w_area = (diam_f - d_min) / denom  # 0..1
        t_area = 1.0 + (base_trunk_thickness - 1.0) * w_area
        t_area = np.clip(t_area, 1.0, 4.0)

        # b) trunk override using base/tip and taper power
        t_trunk = np.full((n_nodes,), 1.0, dtype=np.float32)
        if len(trunk_chain) > 0:
            s_pow = np.power(trunk_s, max(0.01, trunk_taper_power))
            t_trunk = (1.0 - s_pow) * base_trunk_thickness + s_pow * trunk_tip_thickness
            t_trunk = np.clip(t_trunk, 1.0, 4.0)

        # combined: use trunk mapping on trunk nodes; area mapping elsewhere
        t_cont = np.where(is_trunk, t_trunk, t_area)

        # Quantize to stamp keys
        levels = np.array([1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0], dtype=np.float32)
        idx = np.argmin(np.abs(t_cont[:, None] - levels[None, :]), axis=1)
        t_key = levels[idx]

        # --- Voxelization via 3D Bresenham + shape stamps ---
        self.clear()
        bark = np.array(trunk_color, dtype=np.uint8)

        STAMPS = {
            1.0:  [(0, 0)],
            1.3:  [(0, 0), (1, 0)],
            1.7:  [(0, 0), (1, 0), (0, 1)],  # L (2x2, 3 vox)
            2.0:  [(0, 0), (1, 0), (0, 1), (1, 1)],  # full 2x2
            2.5:  [(-1, 0), (0, 0), (1, 0), (0, 1)],  # T (bar on X, stem +Y)
            3.0:  [(-1, 0), (0, 0), (1, 0), (0, -1), (0, 1)],  # plus
            3.25: [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)],
            3.5:  [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1), (1, -1)],
            3.75: [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1), (1, -1), (-1, 1)],
            4.0:  [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)],
        }

        def stamp_shape(x: int, y: int, z: int, level: float):
            offs = STAMPS.get(float(level), STAMPS[1.0])
            for ox, oy in offs:
                xi, yi = x + ox, y + oy
                if 0 <= xi < sx and 0 <= yi < sy and 0 <= z < sz:
                    self.occ[xi, yi, z] = True
                    self.col[xi, yi, z] = bark

        for i in range(1, n_nodes):
            p = parents[i]
            a = nodes[p]
            bpt = nodes[i]
            x1, y1, z1 = int(round(a[0])), int(round(a[1])), int(round(a[2]))
            x2, y2, z2 = int(round(bpt[0])), int(round(bpt[1])), int(round(bpt[2]))
            level = float(t_key[i])
            for (vx, vy, vz) in bresenham3d(x1, y1, z1, x2, y2, z2):
                stamp_shape(vx, vy, vz, level)


# ----------------------------- 3D Bresenham ----------------------------- #

def bresenham3d(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int):
    """Yield integer voxels along a 3D line using Bresenham-like stepping.
    Always includes the end point.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    x, y, z = x1, y1, z1
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x2:
            yield (x, y, z)
            if p1 >= 0:
                y += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z += zs
                p2 -= 2 * dx
            x += xs
            p1 += 2 * dy
            p2 += 2 * dz
        yield (x, y, z)
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y2:
            yield (x, y, z)
            if p1 >= 0:
                x += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z += zs
                p2 -= 2 * dy
            y += ys
            p1 += 2 * dx
            p2 += 2 * dz
        yield (x, y, z)
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z2:
            yield (x, y, z)
            if p1 >= 0:
                y += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x += xs
                p2 -= 2 * dz
            z += zs
            p1 += 2 * dy
            p2 += 2 * dx
        yield (x, y, z)


# ----------------------------- Mesher ----------------------------- #

class VoxelMesher:
    """Builds a simple cubic mesh from voxel occupancy & colors.
    Emits per-vertex positions (float32), normals (float32), colors (float32 0..1), and triangle indices (uint32).
    Faces are added only where a voxel face borders empty/out-of-bounds.
    """

    # Face definitions: (normal, vertex offsets in CCW order when facing outward)
    FACES = {
        "+X": (np.array([+1, 0, 0], dtype=np.int32),
               [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        "-X": (np.array([-1, 0, 0], dtype=np.int32),
               [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)]),
        "+Y": (np.array([0, +1, 0], dtype=np.int32),
               [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),  # CCW outward
        "-Y": (np.array([0, -1, 0], dtype=np.int32),
               [(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)]),  # CCW outward
        "+Z": (np.array([0, 0, +1], dtype=np.int32),
               [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        "-Z": (np.array([0, 0, -1], dtype=np.int32),
               [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)]),
    }

    @staticmethod
    def mesh(volume: VoxelVolume):
        occ = volume.occ
        col = volume.col
        sx, sy, sz = volume.size

        vertices = []   # [x,y,z]
        normals = []    # [nx,ny,nz]
        colors = []     # [r,g,b,a] in 0..1
        indices = []

        filled = np.argwhere(occ)
        if filled.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.uint32),
                np.array([sx/2.0, sy/2.0, sz/2.0], dtype=np.float32),
            )

        min_xyz = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        vtx_count = 0
        for x, y, z in filled:
            min_xyz = np.minimum(min_xyz, [x, y, z])
            max_xyz = np.maximum(max_xyz, [x + 1, y + 1, z + 1])

            voxel_color = col[x, y, z].astype(np.float32) / 255.0

            for name, (nrm, offsets) in VoxelMesher.FACES.items():
                nx, ny, nz = x + nrm[0], y + nrm[1], z + nrm[2]
                is_out = not (0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz)
                if is_out or not occ[nx, ny, nz]:
                    for ox, oy, oz in offsets:
                        vx, vy, vz = x + ox, y + oy, z + oz
                        vertices.append((vx, vy, vz))
                        normals.append(tuple(nrm.astype(np.float32)))
                        colors.append(tuple(voxel_color))
                    indices.extend([vtx_count + 0, vtx_count + 1, vtx_count + 2,
                                    vtx_count + 0, vtx_count + 2, vtx_count + 3])
                    vtx_count += 4

        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)

        center = (min_xyz + max_xyz) * 0.5
        return vertices, normals, colors, indices, center.astype(np.float32)


# ----------------------------- GL Utilities ----------------------------- #

def compile_shader(src: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not status:
        log = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)
        raise RuntimeError(f"Shader compile error (type {shader_type}):\n{log}")
    return shader


def link_program(vs: int, fs: int) -> int:
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    status = glGetProgramiv(prog, GL_LINK_STATUS)
    if not status:
        log = glGetProgramInfoLog(prog).decode()
        glDeleteProgram(prog)
        raise RuntimeError(f"Program link error:\n{log}")
    glDetachShader(prog, vs)
    glDetachShader(prog, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(center - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# ----------------------------- GL Widget ----------------------------- #

class GLVoxelWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(900, 900)
        self.setFocusPolicy(Qt.StrongFocus)

        # Mesh data (CPU side)
        self.vertices: Optional[np.ndarray] = None  # (N,3) float32
        self.normals: Optional[np.ndarray] = None   # (N,3) float32
        self.colors: Optional[np.ndarray] = None    # (N,4) float32
        self.indices: Optional[np.ndarray] = None   # (M,) uint32
        self.center = np.array([32.0, 32.0, 32.0], dtype=np.float32)

        # GL resources
        self.program = None
        self.vao = None
        self.vbo_pos = None
        self.vbo_col = None
        self.vbo_nrm = None
        self.ebo = None
        self.u_mvp_loc = None
        self.u_light_dir_loc = None
        self.u_ambient_loc = None

        # Camera controls (orbit)
        self.yaw_deg = 45.0
        self.pitch_deg = 30.0  # [-89, 89]
        self.distance = 120.0
        self.last_mouse_pos = None

        # Projection params
        self.fov = 45.0
        self.near = 0.1
        self.far = 1000.0
        # Simple world-space light direction (normalized at draw)
        self.light_dir = np.array([0.6, 0.7, 1.0], dtype=np.float32)

    # ----------------- Public API ----------------- #
    def set_mesh(self, vertices: np.ndarray, normals: np.ndarray, colors: np.ndarray, indices: np.ndarray, center: np.ndarray):
        self.vertices = vertices.astype(np.float32)
        self.normals = normals.astype(np.float32)
        self.colors = colors.astype(np.float32)
        self.indices = indices.astype(np.uint32)
        self.center = center.astype(np.float32)

        if self.vertices.size:
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            radius = np.linalg.norm(maxs - mins) * 0.5
            radius = max(radius, 1.0)
            self.distance = radius * 2.5
        else:
            self.distance = 50.0

        if self.context() is not None:
            self.makeCurrent()
            self._upload_mesh()
            self.doneCurrent()
        self.update()

    # ----------------- QOpenGLWidget overrides ----------------- #
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glEnable(GL_MULTISAMPLE)
        glClearColor(0.08, 0.09, 0.11, 1.0)

        vs_src = """
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec4 aColor;
        layout(location=2) in vec3 aNormal;
        uniform mat4 uMVP;
        out vec4 vColor;
        out vec3 vNormal;
        void main(){
            vColor = aColor;
            vNormal = aNormal;
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        fs_src = """
        #version 330 core
        in vec4 vColor;
        in vec3 vNormal;
        uniform vec3 uLightDir;
        uniform float uAmbient;
        out vec4 FragColor;
        void main(){
            float ndl = max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
            float shade = clamp(uAmbient + ndl, 0.0, 1.0);
            FragColor = vec4(vColor.rgb * shade, vColor.a);
        }
        """
        vs = compile_shader(vs_src, GL_VERTEX_SHADER)
        fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
        self.program = link_program(vs, fs)
        self.u_mvp_loc = glGetUniformLocation(self.program, b"uMVP")
        self.u_light_dir_loc = glGetUniformLocation(self.program, b"uLightDir")
        self.u_ambient_loc = glGetUniformLocation(self.program, b"uAmbient")

        self.vao = glGenVertexArrays(1)
        self.vbo_pos = glGenBuffers(1)
        self.vbo_col = glGenBuffers(1)
        self.vbo_nrm = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        # Positions
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Colors
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Normals
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_nrm)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        # EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

        glBindVertexArray(0)

        if self.vertices is not None:
            self._upload_mesh()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.indices is None or self.indices.size == 0:
            return

        width = max(1, self.width())
        height = max(1, self.height())
        proj = perspective(self.fov, width / float(height), self.near, self.far)

        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(np.clip(self.pitch_deg, -89.0, 89.0))
        cx, cy, cz = self.center
        ex = cx + self.distance * math.cos(pitch) * math.cos(yaw)
        ey = cy + self.distance * math.cos(pitch) * math.sin(yaw)
        ez = cz + self.distance * math.sin(pitch)
        eye = np.array([ex, ey, ez], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        view = look_at(eye, self.center, up)

        mvp = proj @ view

        glUseProgram(self.program)
        glUniformMatrix4fv(self.u_mvp_loc, 1, GL_FALSE, mvp.astype(np.float32).T)
        ld = normalize(self.light_dir.astype(np.float32))
        glUniform3f(self.u_light_dir_loc, float(ld[0]), float(ld[1]), float(ld[2]))
        glUniform1f(self.u_ambient_loc, 0.35)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, int(self.indices.size), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)

    def _upload_mesh(self):
        glBindVertexArray(self.vao)

        if self.vertices is None or self.indices is None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_nrm)
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindVertexArray(0)
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_col)
        glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_nrm)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    # Interaction
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.last_mouse_pos is not None and (event.buttons() & Qt.LeftButton):
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.yaw_deg = (self.yaw_deg - dx * 0.4) % 360.0  # natural direction
            self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.3, -89.0, 89.0))
            self.last_mouse_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y() / 120.0
        factor = math.pow(0.9, delta)
        self.distance = float(np.clip(self.distance * factor, 2.0, 2000.0))
        self.update()
        super().wheelEvent(event)


# ----------------------------- Control Panel ----------------------------- #

class ControlPanel(QFrame):
    generate_clicked = QtCore.pyqtSignal(object)  # dict of params

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        # Colonization params
        self.trunk_spin = QSpinBox();      self.trunk_spin.setRange(1, 63);  self.trunk_spin.setValue(28)
        self.canopy_spin = QSpinBox();     self.canopy_spin.setRange(3, 31); self.canopy_spin.setValue(16)
        self.canh_spin = QSpinBox();       self.canh_spin.setRange(4, 63);   self.canh_spin.setValue(16)
        self.prof_spin = QDoubleSpinBox(); self.prof_spin.setRange(0.2, 6.0);self.prof_spin.setSingleStep(0.1); self.prof_spin.setValue(2.0)
        self.zbias_spin = QDoubleSpinBox();self.zbias_spin.setRange(0.2, 6.0);self.zbias_spin.setSingleStep(0.1);self.zbias_spin.setValue(1.5)
        self.points_spin = QSpinBox();     self.points_spin.setRange(50, 50000); self.points_spin.setValue(1000)
        self.step_spin = QDoubleSpinBox(); self.step_spin.setRange(0.5, 20.0); self.step_spin.setSingleStep(0.5); self.step_spin.setValue(3.0)
        self.attr_spin = QDoubleSpinBox(); self.attr_spin.setRange(2.0, 80.0); self.attr_spin.setSingleStep(1.0); self.attr_spin.setValue(18.0)
        self.kill_spin = QDoubleSpinBox(); self.kill_spin.setRange(1.0, 40.0); self.kill_spin.setSingleStep(1.0); self.kill_spin.setValue(6.0)
        self.up_spin = QDoubleSpinBox();   self.up_spin.setRange(0.0, 3.0);   self.up_spin.setSingleStep(0.1);   self.up_spin.setValue(0.6)
        self.light_spin = QDoubleSpinBox();self.light_spin.setRange(0.0, 3.0);self.light_spin.setSingleStep(0.1);self.light_spin.setValue(0.25)
        self.iters_spin = QSpinBox();      self.iters_spin.setRange(1, 5000); self.iters_spin.setValue(600)

        # Trunk taper
        self.base_thick_spin = QDoubleSpinBox(); self.base_thick_spin.setRange(1.0, 4.0); self.base_thick_spin.setSingleStep(0.05); self.base_thick_spin.setValue(3.5)
        self.tip_thick_spin  = QDoubleSpinBox(); self.tip_thick_spin.setRange(1.0, 4.0);  self.tip_thick_spin.setSingleStep(0.05);  self.tip_thick_spin.setValue(1.5)
        self.taper_pow_spin  = QDoubleSpinBox(); self.taper_pow_spin.setRange(0.2, 6.0); self.taper_pow_spin.setSingleStep(0.1);  self.taper_pow_spin.setValue(1.4)

        # Branching
        self.branch_amt_spin = QDoubleSpinBox(); self.branch_amt_spin.setRange(0.0, 1.0); self.branch_amt_spin.setSingleStep(0.05); self.branch_amt_spin.setValue(0.5)
        self.branch_thr_spin = QSpinBox();       self.branch_thr_spin.setRange(1, 32);     self.branch_thr_spin.setValue(4)
        self.spread_deg_spin = QDoubleSpinBox(); self.spread_deg_spin.setRange(0.0, 90.0); self.spread_deg_spin.setSingleStep(1.0);  self.spread_deg_spin.setValue(35.0)

        self.generate_btn = QPushButton("Generate Tree")
        self.generate_btn.clicked.connect(self._emit_generate)

        form = QFormLayout()
        form.addRow(QLabel("<b>Space Colonization</b>"))
        form.addRow("Trunk Height (Z)", self.trunk_spin)
        form.addRow("Canopy Radius", self.canopy_spin)
        form.addRow("Canopy Height", self.canh_spin)
        form.addRow("Canopy Profile Power", self.prof_spin)
        form.addRow("Top Z Bias Power", self.zbias_spin)
        form.addRow("Attraction Points", self.points_spin)
        form.addRow("Step Length", self.step_spin)
        form.addRow("Attract Radius", self.attr_spin)
        form.addRow("Kill Radius", self.kill_spin)
        form.addRow("Up Tropism", self.up_spin)
        form.addRow("Light Tropism", self.light_spin)
        form.addRow("Max Iterations", self.iters_spin)

        form.addRow(QLabel("<b>Trunk Taper</b>"))
        form.addRow("Base Thickness (1–4)", self.base_thick_spin)
        form.addRow("Tip Thickness (1–4)", self.tip_thick_spin)
        form.addRow("Taper Power", self.taper_pow_spin)

        form.addRow(QLabel("<b>Branching</b>"))
        form.addRow("Branching Amount (0–1)", self.branch_amt_spin)
        form.addRow("Branch Threshold", self.branch_thr_spin)
        form.addRow("Angle Spread (deg)", self.spread_deg_spin)
        form.addRow(self.generate_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch(1)

    def _emit_generate(self):
        params = dict(
            trunk_height=self.trunk_spin.value(),
            canopy_radius=self.canopy_spin.value(),
            canopy_height=self.canh_spin.value(),
            canopy_profile_power=self.prof_spin.value(),
            z_bias_power=self.zbias_spin.value(),
            point_count=self.points_spin.value(),
            step=self.step_spin.value(),
            r_attr=self.attr_spin.value(),
            r_kill=self.kill_spin.value(),
            w_up=self.up_spin.value(),
            w_light=self.light_spin.value(),
            max_iters=self.iters_spin.value(),
            base_trunk_thickness=self.base_thick_spin.value(),
            trunk_tip_thickness=self.tip_thick_spin.value(),
            trunk_taper_power=self.taper_pow_spin.value(),
            branching_amount=self.branch_amt_spin.value(),
            branch_threshold=self.branch_thr_spin.value(),
            angle_spread_deg=self.spread_deg_spin.value(),
        )
        self.generate_clicked.emit(params)


# ----------------------------- Main Window ----------------------------- #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procedural Voxel Tree (Space Colonization)")
        self.resize(1200, 920)

        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 3)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)

        self.volume = VoxelVolume(size=(64, 64, 64))
        self.mesher = VoxelMesher()

        self.glw = GLVoxelWidget()
        self.controls = ControlPanel()
        self.controls.generate_clicked.connect(self.on_generate)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 920])

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.addWidget(splitter)
        self.setCentralWidget(container)

        # Initial content
        self.on_generate({
            'trunk_height': 28,
            'canopy_radius': 16,
            'canopy_height': 16,
            'canopy_profile_power': 2.0,
            'z_bias_power': 1.5,
            'point_count': 1000,
            'step': 3.0,
            'r_attr': 18.0,
            'r_kill': 6.0,
            'w_up': 0.6,
            'w_light': 0.25,
            'max_iters': 600,
            'base_trunk_thickness': 3.5,
            'trunk_tip_thickness': 1.5,
            'trunk_taper_power': 1.4,
            'branching_amount': 0.5,
            'branch_threshold': 4,
            'angle_spread_deg': 35.0,
        })

    @QtCore.pyqtSlot(object)
    def on_generate(self, params: Dict):
        # Generate voxel content from Space Colonization
        self.volume.generate_tree_colonization(params)
        # Mesh it
        vertices, normals, colors, indices, center = self.mesher.mesh(self.volume)
        # Send to GL
        self.glw.set_mesh(vertices, normals, colors, indices, center)


# ----------------------------- Entrypoint ----------------------------- #

def main():
    app = QApplication(sys.argv)

    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
