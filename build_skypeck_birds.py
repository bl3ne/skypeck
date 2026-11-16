"""
High-detail anatomical 3D bird character generator for Skypeck in Blender 5.3+.
Builds 4 real-bird-inspired articulated character rigs in local pivot coordinates:
  1. sky_jay        - Blue Jay (Cyanocitta cristata) & Eurasian Kingfisher
  2. origami_glider - Wandering Albatross (Diomedea exulans) & Red-crowned Crane (Origami)
  3. crimson_falcon - Peregrine Falcon (Falco peregrinus) & Golden Eagle
  4. emerald_owl    - Barn Owl (Tyto alba) & Great Horned Owl (Bubo virginianus)

Exports:
  - /home/bl3ne/.gemini/antigravity/scratch/skypeck/skypeck_birds.glb
  - /home/bl3ne/.gemini/antigravity/scratch/skypeck/skypeck_birds.blend
"""

import math
import os
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler


def to_blender(x, y, z):
    """Convert Three.js Y-up (x, y, z) to Blender Z-up (x, -z, y)."""
    return Vector((float(x), float(-z), float(y)))


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(name, hex_color, roughness=0.5, metalness=0.0, emissive_hex=None, emissive_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r = ((hex_color >> 16) & 0xFF) / 255.0
    g = ((hex_color >> 8) & 0xFF) / 255.0
    b = (hex_color & 0xFF) / 255.0
    # sRGB to linear approximation
    rl, gl, bl = r ** 2.2, g ** 2.2, b ** 2.2
    bsdf.inputs["Base Color"].default_value = (rl, gl, bl, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metalness
    if emissive_hex is not None and emissive_strength > 0.0:
        er = (((emissive_hex >> 16) & 0xFF) / 255.0) ** 2.2
        eg = (((emissive_hex >> 8) & 0xFF) / 255.0) ** 2.2
        eb = ((emissive_hex & 0xFF) / 255.0) ** 2.2
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (er, eg, eb, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emissive_strength
    mat.use_backface_culling = False
    return mat


def ensure_mat_slot(obj, mat):
    for idx, m in enumerate(obj.data.materials):
        if m == mat:
            return idx
    obj.data.materials.append(mat)
    return len(obj.data.materials) - 1


def create_part_object(name):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def finalize_bmesh(bm, obj):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = False
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def transform_pt(pt, origin=(0, 0, 0), rot_xyz=(0, 0, 0), scale=(1, 1, 1)):
    x = pt[0] * scale[0]
    y = pt[1] * scale[1]
    z = pt[2] * scale[2]
    rx, ry, rz = rot_xyz
    # Rotate Z (roll), then X (pitch), then Y (yaw) in Three.js space
    if rz != 0.0:
        cz, sz = math.cos(rz), math.sin(rz)
        x, y = x * cz - y * sz, x * sz + y * cz
    if rx != 0.0:
        cx, sx = math.cos(rx), math.sin(rx)
        y, z = y * cx - z * sx, y * sx + z * cx
    if ry != 0.0:
        cy, sy = math.cos(ry), math.sin(ry)
        x, z = x * cy + z * sy, -x * sy + z * cy
    return (x + origin[0], y + origin[1], z + origin[2])


def add_lofted_rings(bm, rings, mat_idx_fn, cap_start=True, cap_end=True):
    """
    rings: list of rings; each ring is a list of (x, y, z) in Three.js space.
    mat_idx_fn(ring_idx, seg_idx, num_rings, num_segs) -> int material slot index.
    """
    v_rings = []
    for r in rings:
        v_rings.append([bm.verts.new(to_blender(p[0], p[1], p[2])) for p in r])
    n_rings = len(v_rings)
    n_segs = len(v_rings[0])
    for ri in range(n_rings - 1):
        r0 = v_rings[ri]
        r1 = v_rings[ri + 1]
        for si in range(n_segs):
            s_next = (si + 1) % n_segs
            try:
                f = bm.faces.new((r0[si], r0[s_next], r1[s_next], r1[si]))
                f.material_index = mat_idx_fn(ri, si, n_rings, n_segs)
            except ValueError:
                pass
    if cap_start and n_segs >= 3:
        center = Vector((0, 0, 0))
        for v in v_rings[0]:
            center += v.co
        center /= n_segs
        cv = bm.verts.new(center)
        for si in range(n_segs):
            s_next = (si + 1) % n_segs
            try:
                f = bm.faces.new((v_rings[0][s_next], v_rings[0][si], cv))
                f.material_index = mat_idx_fn(0, si, n_rings, n_segs)
            except ValueError:
                pass
    if cap_end and n_segs >= 3:
        center = Vector((0, 0, 0))
        for v in v_rings[-1]:
            center += v.co
        center /= n_segs
        cv = bm.verts.new(center)
        for si in range(n_segs):
            s_next = (si + 1) % n_segs
            try:
                f = bm.faces.new((v_rings[-1][si], v_rings[-1][s_next], cv))
                f.material_index = mat_idx_fn(n_rings - 2, si, n_rings, n_segs)
            except ValueError:
                pass


def make_avian_cross_section(z, rx, ry_dorsal, ry_ventral, y_offset=0.0, keel_sharpness=0.25, n_segs=16):
    """
    Generate a single anatomical bird cross-section ring at longitudinal coordinate z:
    - Rounded dorsal back (top, theta near 0)
    - Full pectoral sides
    - Pronounced sternum keel (bottom, theta near pi)
    """
    pts = []
    for i in range(n_segs):
        theta = (i / n_segs) * math.tau  # 0 = top (+Y), pi/2 = +X, pi = bottom (-Y)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        x = sin_t * rx
        if cos_t >= 0:
            # Dorsal arch
            y = (cos_t ** 0.85) * ry_dorsal
        else:
            # Ventral breast with sternum keel pinch
            keel = (abs(cos_t) ** 1.6) * keel_sharpness * ry_ventral
            y = -((-cos_t) ** 0.9) * ry_ventral - keel
        pts.append((x, y + y_offset, z))
    return pts


def add_feather_3d(
    bm,
    origin=(0, 0, 0),
    length=0.75,
    width=0.22,
    thickness=0.024,
    rot_xyz=(0, 0, 0),
    camber=0.04,
    asymmetry=0.35,
    mat_base=0,
    mat_bar=None,
    mat_tip=None,
    tip_sharpness=0.65
):
    """
    Build an anatomical 3D feather extending along +Z in local feather space,
    then transformed by rot_xyz and origin in Three.js space.
    Includes central rachis ridge, asymmetric leading/trailing vanes, camber arch,
    subterminal bar, and tapered tip.
    """
    steps = [0.0, 0.18, 0.45, 0.72, 0.88, 1.0]
    w_profile = [0.38, 0.88, 1.0, 0.92, 0.58 * (1.0 - 0.35 * tip_sharpness), 0.0]
    rings = []
    for idx, t in enumerate(steps):
        z = t * length
        cy = math.sin(t * math.pi) * camber
        w = width * w_profile[idx]
        th = thickness * (1.0 - 0.65 * t)
        if idx == len(steps) - 1:
            # Tip vertex ring (collapsed slight width)
            w = width * 0.04 * (1.0 - tip_sharpness * 0.8)
            th = thickness * 0.15
        w_lead = w * (0.5 - asymmetry * 0.22)
        w_trail = w * (0.5 + asymmetry * 0.22)
        # 6-point diamond-airfoil cross-section around rachis
        local_ring = [
            (0.0, cy + th * 0.65, z),          # 0: dorsal rachis ridge
            (w_trail, cy - th * 0.08, z),      # 1: outer/trailing vane edge
            (0.0, cy - th * 0.50, z),          # 2: ventral rachis groove
            (-w_lead, cy - th * 0.05, z),      # 3: inner/leading vane edge
        ]
        rings.append([transform_pt(p, origin=origin, rot_xyz=rot_xyz) for p in local_ring])

    def pick_mat(ri, si, nr, ns):
        if mat_tip is not None and ri >= nr - 2:
            return mat_tip
        if mat_bar is not None and ri == nr - 3:
            return mat_bar
        return mat_base

    add_lofted_rings(bm, rings, pick_mat, cap_start=True, cap_end=True)


def add_wing_airfoil_patagium(
    bm,
    side,
    span,
    root_chord,
    tip_chord,
    root_thick,
    tip_thick,
    sweep_z,
    dihedral_y,
    mat_upper,
    mat_lower,
    mat_leading,
    origin=(0, 0, 0)
):
    """
    Create a cambered avian airfoil wing core extending along side * +X.
    """
    n_span = 6
    n_profile = 10
    rings = []
    for si in range(n_span):
        u = si / (n_span - 1)
        x = origin[0] + side * (u * span)
        y_off = origin[1] + math.sin(u * math.pi * 0.6) * dihedral_y
        z_off = origin[2] + (u ** 1.15) * sweep_z
        chord = root_chord * (1.0 - u) + tip_chord * u
        thick = root_thick * (1.0 - u * 0.65) + tip_thick * (u * 0.65)
        ring = []
        for pi in range(n_profile):
            ang = (pi / n_profile) * math.tau
            # Airfoil parameterization: rounded leading edge (-Z), sharp trailing edge (+Z)
            zc = -math.cos(ang) * 0.5  # -0.5 (leading) to +0.5 (trailing)
            taper = (0.5 - zc) ** 0.65 if zc < 0.5 else 0.05
            yc = math.sin(ang) * thick * taper
            # Add positive avian camber arch
            camber_arch = (0.25 - zc * zc) * thick * 0.55
            ring.append((x, y_off + yc + camber_arch, z_off + zc * chord))
        if side < 0:
            ring.reverse()
        rings.append(ring)

    def mat_fn(ri, si, nr, ns):
        # Leading edge vs dorsal vs ventral
        if si in (0, ns - 1):
            return mat_leading
        if si < ns // 2:
            return mat_upper
        return mat_lower

    add_lofted_rings(bm, rings, mat_fn, cap_start=True, cap_end=True)


def add_tucked_legs(bm, y_base, z_base, spread_x, scale, mat_leg, mat_claw):
    """Add tucked avian legs and curved talons under the belly."""
    for side in (-1, 1):
        bx = side * spread_x
        # Tarsus tucked backward
        tarsus_rings = [
            make_avian_cross_section(z_base - 0.12 * scale, 0.055 * scale, 0.055 * scale, 0.055 * scale, y_base + 0.04 * scale, 0.0, 8),
            make_avian_cross_section(z_base + 0.16 * scale, 0.038 * scale, 0.038 * scale, 0.038 * scale, y_base - 0.05 * scale, 0.0, 8),
        ]
        tarsus_rings = [[(p[0] + bx, p[1], p[2]) for p in r] for r in tarsus_rings]
        add_lofted_rings(bm, tarsus_rings, lambda *_: mat_leg, cap_start=True, cap_end=True)
        # 3 curled toes + talons
        for ti in (-1, 0, 1):
            tx = bx + ti * 0.035 * scale
            claw_rings = [
                [(tx + math.sin(a) * 0.022 * scale, y_base - 0.06 * scale + math.cos(a) * 0.022 * scale, z_base + 0.16 * scale) for a in [k * math.tau / 6 for k in range(6)]],
                [(tx + math.sin(a) * 0.014 * scale, y_base - 0.11 * scale + math.cos(a) * 0.014 * scale, z_base + 0.25 * scale) for a in [k * math.tau / 6 for k in range(6)]],
                [(tx + math.sin(a) * 0.004 * scale, y_base - 0.08 * scale + math.cos(a) * 0.004 * scale, z_base + 0.31 * scale) for a in [k * math.tau / 6 for k in range(6)]],
            ]
            add_lofted_rings(bm, claw_rings, lambda ri, *_: mat_leg if ri == 0 else mat_claw, cap_start=True, cap_end=True)


def add_eye_with_glint(bm, side, pos, radius, mat_ring, mat_iris, mat_pupil, mat_glint):
    """Build a layered anatomical bird eye with orbital ring, iris, pupil, and specular glint."""
    ex, ey, ez = pos[0] * side, pos[1], pos[2]
    n_seg = 12
    # Orbital ring + eyeball dome protruding slightly along side * X and -Z
    rings = []
    for ri, (frac, ext, m) in enumerate([(1.18, 0.0, mat_ring), (1.0, 0.035, mat_iris), (0.58, 0.052, mat_pupil), (0.15, 0.060, mat_pupil)]):
        r = radius * frac
        ring = []
        for si in range(n_seg):
            ang = (si / n_seg) * math.tau
            dy = math.cos(ang) * r
            dz = math.sin(ang) * r
            dx = side * (ext + r * 0.28)
            ring.append((ex + dx, ey + dy, ez + dz - ext * 0.45))
        if side < 0:
            ring.reverse()
        rings.append(ring)

    def eye_mat(ri, *_):
        if ri == 0:
            return mat_ring
        if ri == 1:
            return mat_iris
        return mat_pupil

    add_lofted_rings(bm, rings, eye_mat, cap_start=False, cap_end=True)

    # Specular glint prism
    gx = ex + side * (radius * 0.36 + 0.058)
    gy = ey + radius * 0.28
    gz = ez - radius * 0.26
    gs = radius * 0.22
    glint_rings = [
        [(gx, gy + gs, gz - gs), (gx + side * gs * 0.5, gy, gz - gs), (gx, gy - gs, gz - gs), (gx - side * gs * 0.2, gy, gz - gs)],
        [(gx, gy + gs * 0.5, gz + gs), (gx + side * gs * 0.5, gy, gz + gs), (gx, gy - gs * 0.5, gz + gs), (gx - side * gs * 0.2, gy, gz + gs)],
    ]
    add_lofted_rings(bm, glint_rings, lambda *_: mat_glint, cap_start=True, cap_end=True)


# ===========================================================================
# 1. SKY JAY — North American Blue Jay (Cyanocitta cristata) & Kingfisher
# ===========================================================================
def build_sky_jay():
    m_cobalt = make_material("jay_cobalt", 0x1E6BFF, roughness=0.46)
    m_cerulean = make_material("jay_cerulean", 0x38BDF8, roughness=0.42)
    m_ultramarine = make_material("jay_ultramarine", 0x0A369D, roughness=0.52)
    m_ivory = make_material("jay_ivory", 0xF1F5F9, roughness=0.55)
    m_collar = make_material("jay_collar", 0x0F172A, roughness=0.48)
    m_beak = make_material("jay_beak", 0xFFB800, roughness=0.32, metalness=0.08)
    m_beak_dark = make_material("jay_beak_dark", 0xB45309, roughness=0.38)
    m_eye = make_material("jay_eye", 0x090D16, roughness=0.12)
    m_glint = make_material("jay_glint", 0xFFFFFF, roughness=0.05, emissive_hex=0xFFFFFF, emissive_strength=0.6)
    m_leg = make_material("jay_leg", 0x334155, roughness=0.55)

    # --- BODY ---
    obj_body = create_part_object("sky_jay__body")
    bm = bmesh.new()
    i_cobalt = ensure_mat_slot(obj_body, m_cobalt)
    i_cerulean = ensure_mat_slot(obj_body, m_cerulean)
    i_ivory = ensure_mat_slot(obj_body, m_ivory)
    i_collar = ensure_mat_slot(obj_body, m_collar)
    i_leg = ensure_mat_slot(obj_body, m_leg)
    i_ultra = ensure_mat_slot(obj_body, m_ultramarine)

    # 11 longitudinal torso stations from neck (-1.08) to tail base (+1.06)
    stations = [
        (-1.08, 0.24, 0.22, 0.20, 0.14),
        (-0.86, 0.38, 0.32, 0.34, 0.10),
        (-0.58, 0.52, 0.42, 0.48, 0.04),
        (-0.26, 0.58, 0.46, 0.52, 0.00),
        (0.06,  0.54, 0.44, 0.46, 0.01),
        (0.38,  0.46, 0.38, 0.36, 0.03),
        (0.68,  0.34, 0.28, 0.25, 0.05),
        (0.92,  0.22, 0.18, 0.15, 0.06),
        (1.08,  0.13, 0.10, 0.09, 0.06),
    ]
    rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.22, n_segs=20) for (z, rx, ryd, ryv, y0) in stations]

    def body_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        # Black Blue Jay necklace collar around neck/upper breast (ri == 1)
        if ri == 1 and cos_a < 0.35:
            return i_collar
        # Dorsal cobalt & cerulean mantle
        if cos_a > 0.15:
            return i_cerulean if ri in (2, 3, 4) and abs(math.sin(ang)) < 0.65 else i_cobalt
        # Lateral flank transition
        if cos_a > -0.20:
            return i_cerulean if ri < 5 else i_cobalt
        # Ventral ivory breast & belly
        return i_ivory

    add_lofted_rings(bm, rings, body_mat, cap_start=True, cap_end=True)

    # Layered dorsal scapular & mantle feathers on back
    for row in range(3):
        z_row = -0.42 + row * 0.34
        for side in (-1, 1):
            for col in range(2):
                fx = side * (0.16 + col * 0.16)
                fy = 0.39 - col * 0.06 - row * 0.03
                add_feather_3d(
                    bm,
                    origin=(fx, fy, z_row),
                    length=0.44,
                    width=0.18,
                    thickness=0.022,
                    rot_xyz=(-0.08, side * 0.12, -side * 0.25),
                    camber=0.025,
                    mat_base=i_cerulean if col == 0 else i_cobalt,
                    mat_bar=i_collar,
                    mat_tip=i_ultra
                )

    add_tucked_legs(bm, y_base=-0.34, z_base=0.42, spread_x=0.18, scale=1.0, mat_leg=i_leg, mat_claw=i_collar)
    finalize_bmesh(bm, obj_body)

    # --- HEAD (local to headGroup at (0, 0.24, -1.10)) ---
    obj_head = create_part_object("sky_jay__head")
    bm = bmesh.new()
    h_cobalt = ensure_mat_slot(obj_head, m_cobalt)
    h_cerulean = ensure_mat_slot(obj_head, m_cerulean)
    h_ivory = ensure_mat_slot(obj_head, m_ivory)
    h_collar = ensure_mat_slot(obj_head, m_collar)
    h_beak = ensure_mat_slot(obj_head, m_beak)
    h_beak_dark = ensure_mat_slot(obj_head, m_beak_dark)
    h_eye = ensure_mat_slot(obj_head, m_eye)
    h_glint = ensure_mat_slot(obj_head, m_glint)

    head_stations = [
        (0.32,  0.26, 0.24, 0.20, -0.02),
        (0.14,  0.35, 0.33, 0.26, 0.02),
        (-0.06, 0.37, 0.36, 0.27, 0.03),
        (-0.26, 0.31, 0.29, 0.22, 0.01),
        (-0.42, 0.19, 0.17, 0.14, -0.02),
    ]
    h_rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.08, n_segs=18) for (z, rx, ryd, ryv, y0) in head_stations]

    def head_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        sin_a = abs(math.sin(ang))
        # Black bridle/lore line near beak & eyes
        if ri >= 2 and abs(cos_a) < 0.28 and sin_a > 0.75:
            return h_collar
        # Crown is cobalt/cerulean
        if cos_a > 0.22:
            return h_cobalt if ri < 2 else h_cerulean
        # Cheek & throat are white ivory
        return h_ivory

    add_lofted_rings(bm, h_rings, head_mat, cap_start=True, cap_end=True)

    # Prominent Blue Jay occipital crest (7 sculpted feathers)
    crest_specs = [
        (0.0,   0.33, -0.06, 0.56, 0.14, 0.42, 0.0),
        (-0.09, 0.31, -0.02, 0.50, 0.13, 0.36, -0.14),
        (0.09,  0.31, -0.02, 0.50, 0.13, 0.36, 0.14),
        (-0.17, 0.27,  0.04, 0.42, 0.11, 0.28, -0.26),
        (0.17,  0.27,  0.04, 0.42, 0.11, 0.28, 0.26),
        (0.0,   0.36, -0.16, 0.48, 0.12, 0.48, 0.0),
        (0.0,   0.26,  0.10, 0.44, 0.13, 0.25, 0.0),
    ]
    for (cx, cy, cz, clen, cwid, pitch, yaw) in crest_specs:
        add_feather_3d(
            bm,
            origin=(cx, cy, cz),
            length=clen,
            width=cwid,
            thickness=0.024,
            rot_xyz=(pitch, yaw, 0.0),
            camber=0.04,
            mat_base=h_cobalt,
            mat_bar=h_cerulean,
            mat_tip=h_collar,
            tip_sharpness=0.85
        )

    # Upper maxilla + lower mandible beak (pointing -Z)
    maxilla_rings = [
        [(0.0, 0.09, -0.36), (0.14, 0.0, -0.36), (0.0, -0.02, -0.36), (-0.14, 0.0, -0.36)],
        [(0.0, 0.06, -0.64), (0.09, -0.01, -0.64), (0.0, -0.03, -0.64), (-0.09, -0.01, -0.64)],
        [(0.0, 0.01, -0.88), (0.03, -0.03, -0.88), (0.0, -0.045, -0.88), (-0.03, -0.03, -0.88)],
        [(0.0, -0.035, -0.98), (0.008, -0.045, -0.98), (0.0, -0.055, -0.98), (-0.008, -0.045, -0.98)],
    ]
    add_lofted_rings(bm, maxilla_rings, lambda ri, si, *_: h_beak_dark if si == 0 else h_beak, cap_start=True, cap_end=True)

    mandible_rings = [
        [(0.0, -0.01, -0.36), (0.12, -0.02, -0.36), (0.0, -0.09, -0.36), (-0.12, -0.02, -0.36)],
        [(0.0, -0.02, -0.62), (0.07, -0.03, -0.62), (0.0, -0.07, -0.62), (-0.07, -0.03, -0.62)],
        [(0.0, -0.035, -0.86), (0.015, -0.04, -0.86), (0.0, -0.05, -0.86), (-0.015, -0.04, -0.86)],
    ]
    add_lofted_rings(bm, mandible_rings, lambda *_: h_beak_dark, cap_start=True, cap_end=True)

    for side in (-1, 1):
        add_eye_with_glint(bm, side, (0.28, 0.07, -0.18), 0.082, h_collar, h_eye, h_eye, h_glint)

    finalize_bmesh(bm, obj_head)

    # --- WINGS (inner & outer L/R) ---
    for side, tag in [(-1, "L"), (1, "R")]:
        # INNER WING (local to leftWingPivot / rightWingPivot at (side*0.36, 0.16, -0.22))
        obj_wi = create_part_object(f"sky_jay__wing_inner_{tag}")
        bm = bmesh.new()
        wi_cobalt = ensure_mat_slot(obj_wi, m_cobalt)
        wi_cerulean = ensure_mat_slot(obj_wi, m_cerulean)
        wi_ultra = ensure_mat_slot(obj_wi, m_ultramarine)
        wi_ivory = ensure_mat_slot(obj_wi, m_ivory)
        wi_collar = ensure_mat_slot(obj_wi, m_collar)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=1.02,
            root_chord=0.68,
            tip_chord=0.52,
            root_thick=0.14,
            tip_thick=0.09,
            sweep_z=0.04,
            dihedral_y=0.03,
            mat_upper=wi_cerulean,
            mat_lower=wi_ivory,
            mat_leading=wi_cobalt,
            origin=(0.0, 0.0, -0.08)
        )

        # Greater secondary coverts (white-tipped like real Blue Jay!)
        for ci in range(7):
            u = (ci + 0.5) / 7.0
            cx = side * (0.10 + u * 0.88)
            add_feather_3d(
                bm,
                origin=(cx, 0.045, -0.06 + u * 0.03),
                length=0.44,
                width=0.16,
                thickness=0.018,
                rot_xyz=(-0.03, side * 0.04, 0.0),
                camber=0.025,
                asymmetry=0.2 * side,
                mat_base=wi_cerulean,
                mat_bar=wi_collar,
                mat_tip=wi_ivory,
                tip_sharpness=0.35
            )

        # 8 Secondary flight feathers (remiges secondariae) with black bars & white tips
        for si in range(8):
            u = (si + 0.4) / 8.0
            sx = side * (0.08 + u * 0.92)
            slen = 0.72 - u * 0.06
            add_feather_3d(
                bm,
                origin=(sx, 0.0, 0.06 + u * 0.04),
                length=slen,
                width=0.18,
                thickness=0.020,
                rot_xyz=(-0.02, side * (0.03 + u * 0.06), 0.0),
                camber=0.035,
                asymmetry=0.32 * side,
                mat_base=wi_cobalt,
                mat_bar=wi_collar,
                mat_tip=wi_ivory,
                tip_sharpness=0.45
            )

        finalize_bmesh(bm, obj_wi)

        # OUTER WING (local to outerWingL / outerWingR at (side*1.34, 0, -0.02))
        obj_wo = create_part_object(f"sky_jay__wing_outer_{tag}")
        bm = bmesh.new()
        wo_cobalt = ensure_mat_slot(obj_wo, m_cobalt)
        wo_cerulean = ensure_mat_slot(obj_wo, m_cerulean)
        wo_ultra = ensure_mat_slot(obj_wo, m_ultramarine)
        wo_ivory = ensure_mat_slot(obj_wo, m_ivory)
        wo_collar = ensure_mat_slot(obj_wo, m_collar)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=0.72,
            root_chord=0.46,
            tip_chord=0.18,
            root_thick=0.085,
            tip_thick=0.035,
            sweep_z=0.16,
            dihedral_y=0.01,
            mat_upper=wo_cobalt,
            mat_lower=wo_ivory,
            mat_leading=wo_ultra,
            origin=(0.0, 0.0, -0.04)
        )

        # Alula (3 thumb feathers at wrist leading edge)
        for ai in range(3):
            add_feather_3d(
                bm,
                origin=(side * (0.06 + ai * 0.09), 0.03, -0.18 + ai * 0.03),
                length=0.28 - ai * 0.03,
                width=0.09,
                thickness=0.015,
                rot_xyz=(0.02, side * (0.75 - ai * 0.12), 0.0),
                camber=0.02,
                mat_base=wo_cerulean,
                mat_tip=wo_ultra,
                tip_sharpness=0.8
            )

        # 8 Slotted Primary Remiges fanning from inner primary to outer wingtip
        for pi in range(8):
            v = pi / 7.0  # 0 = innermost primary, 1 = outermost leading primary
            px = side * (0.08 + v * 0.58)
            pz = 0.06 - v * 0.08
            plen = 0.76 + math.sin(v * math.pi * 0.75) * 0.34
            yaw = side * (0.14 + v * 1.02)
            pitch = -0.02 + v * 0.05
            add_feather_3d(
                bm,
                origin=(px, 0.0, pz),
                length=plen,
                width=0.155 - v * 0.035,
                thickness=0.018,
                rot_xyz=(pitch, yaw, 0.0),
                camber=0.03,
                asymmetry=0.45 * side,
                mat_base=wo_cobalt if pi < 5 else wo_ultra,
                mat_bar=wo_collar,
                mat_tip=wo_ivory if pi < 4 else wo_ultra,
                tip_sharpness=0.82
            )

        finalize_bmesh(bm, obj_wo)

    # --- TAIL FEATHERS (7 articulated rectrices, local to each tailFeather pivot) ---
    for ti in range(7):
        obj_tf = create_part_object(f"sky_jay__tail_{ti}")
        bm = bmesh.new()
        tf_cobalt = ensure_mat_slot(obj_tf, m_cobalt)
        tf_cerulean = ensure_mat_slot(obj_tf, m_cerulean)
        tf_ivory = ensure_mat_slot(obj_tf, m_ivory)
        tf_collar = ensure_mat_slot(obj_tf, m_collar)
        tf_ultra = ensure_mat_slot(obj_tf, m_ultramarine)

        factor = (ti / 6.0) * 2.0 - 1.0  # -1..+1
        is_central = abs(factor) < 0.2
        is_outer = abs(factor) > 0.6
        t_len = 1.16 - abs(factor) * 0.22
        add_feather_3d(
            bm,
            origin=(0.0, 0.0, 0.0),
            length=t_len,
            width=0.22,
            thickness=0.022,
            rot_xyz=(0.0, 0.0, factor * 0.08),
            camber=0.03,
            asymmetry=0.25 * factor,
            mat_base=tf_cobalt if ti % 2 == 0 else tf_cerulean,
            mat_bar=tf_collar,
            mat_tip=tf_ultra if is_central else (tf_ivory if is_outer else tf_collar),
            tip_sharpness=0.55
        )
        finalize_bmesh(bm, obj_tf)


# ===========================================================================
# 2. ORIGAMI GLIDER — Wandering Albatross & Red-crowned Crane (Washi + Gold)
# ===========================================================================
def build_origami_glider():
    m_washi = make_material("origami_washi", 0xF8FAFC, roughness=0.82, metalness=0.02)
    m_cream = make_material("origami_cream", 0xFEF3C7, roughness=0.76, metalness=0.05)
    m_gold = make_material("origami_gold", 0xF59E0B, roughness=0.28, metalness=0.72, emissive_hex=0xF59E0B, emissive_strength=0.12)
    m_ink = make_material("origami_ink", 0x0F172A, roughness=0.52, metalness=0.15)
    m_crimson = make_material("origami_crimson", 0xDC2626, roughness=0.38, emissive_hex=0xDC2626, emissive_strength=0.2)

    # --- BODY ---
    obj_body = create_part_object("origami_glider__body")
    bm = bmesh.new()
    i_washi = ensure_mat_slot(obj_body, m_washi)
    i_cream = ensure_mat_slot(obj_body, m_cream)
    i_gold = ensure_mat_slot(obj_body, m_gold)
    i_ink = ensure_mat_slot(obj_body, m_ink)

    # Faceted 8-gon origami crane/albatross hull with crisp diamond folds
    hull_stations = [
        (-1.05, 0.14, 0.18, 0.24, 0.14),
        (-0.72, 0.32, 0.30, 0.42, 0.04),
        (-0.28, 0.48, 0.44, 0.54, -0.02),
        (0.24,  0.42, 0.36, 0.46, 0.00),
        (0.68,  0.26, 0.24, 0.28, 0.04),
        (1.02,  0.10, 0.12, 0.14, 0.08),
    ]
    hull_rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.45, n_segs=8) for (z, rx, ryd, ryv, y0) in hull_stations]

    def origami_body_mat(ri, si, nr, ns):
        if si in (0, 4):
            return i_gold
        if ri == 0 and si in (1, 7):
            return i_ink
        return i_washi if si in (1, 2, 6, 7) else i_cream

    add_lofted_rings(bm, hull_rings, origami_body_mat, cap_start=True, cap_end=True)

    # Origami layered dorsal diamond pleats & gilded spine
    for p_idx in range(3):
        pz = -0.52 + p_idx * 0.38
        add_feather_3d(
            bm,
            origin=(0.0, 0.38 - p_idx * 0.04, pz),
            length=0.48,
            width=0.36 - p_idx * 0.04,
            thickness=0.025,
            rot_xyz=(-0.06, 0.0, 0.0),
            camber=0.0,
            mat_base=i_washi,
            mat_bar=i_cream,
            mat_tip=i_gold,
            tip_sharpness=0.95
        )
    finalize_bmesh(bm, obj_body)

    # --- HEAD (local to headGroup at (0, 0.22, -0.95)) ---
    obj_head = create_part_object("origami_glider__head")
    bm = bmesh.new()
    h_washi = ensure_mat_slot(obj_head, m_washi)
    h_gold = ensure_mat_slot(obj_head, m_gold)
    h_ink = ensure_mat_slot(obj_head, m_ink)
    h_crimson = ensure_mat_slot(obj_head, m_crimson)

    # Faceted Red-crowned Crane head & long folded origami bill
    crane_rings = [
        [(0.0, 0.22, 0.18), (0.18, 0.04, 0.18), (0.12, -0.14, 0.18), (0.0, -0.20, 0.18), (-0.12, -0.14, 0.18), (-0.18, 0.04, 0.18)],
        [(0.0, 0.32, -0.06), (0.22, 0.06, -0.06), (0.14, -0.12, -0.06), (0.0, -0.18, -0.06), (-0.14, -0.12, -0.06), (-0.22, 0.06, -0.06)],
        [(0.0, 0.20, -0.34), (0.13, 0.02, -0.34), (0.08, -0.08, -0.34), (0.0, -0.11, -0.34), (-0.08, -0.08, -0.34), (-0.13, 0.02, -0.34)],
        [(0.0, 0.06, -0.68), (0.05, 0.0, -0.68), (0.03, -0.04, -0.68), (0.0, -0.06, -0.68), (-0.03, -0.04, -0.68), (-0.05, 0.0, -0.68)],
        [(0.0, 0.01, -1.04), (0.01, -0.01, -1.04), (0.008, -0.02, -1.04), (0.0, -0.025, -1.04), (-0.008, -0.02, -1.04), (-0.01, -0.01, -1.04)],
    ]

    def crane_head_mat(ri, si, *_):
        if ri == 1 and si in (0, 5):
            return h_crimson  # Tancho red crown!
        if ri >= 2:
            return h_gold if si in (0, 5) else h_washi
        if si in (1, 4):
            return h_ink
        return h_washi

    add_lofted_rings(bm, crane_rings, crane_head_mat, cap_start=True, cap_end=True)

    # Origami swept paper cresthorn
    add_feather_3d(
        bm,
        origin=(0.0, 0.28, -0.05),
        length=0.52,
        width=0.14,
        thickness=0.02,
        rot_xyz=(0.38, 0.0, 0.0),
        camber=0.0,
        mat_base=h_crimson,
        mat_tip=h_gold,
        tip_sharpness=0.95
    )
    finalize_bmesh(bm, obj_head)

    # --- WINGS (Albatross high-aspect-ratio pleated origami wings) ---
    for side, tag in [(-1, "L"), (1, "R")]:
        obj_wi = create_part_object(f"origami_glider__wing_inner_{tag}")
        bm = bmesh.new()
        wi_washi = ensure_mat_slot(obj_wi, m_washi)
        wi_cream = ensure_mat_slot(obj_wi, m_cream)
        wi_gold = ensure_mat_slot(obj_wi, m_gold)
        wi_ink = ensure_mat_slot(obj_wi, m_ink)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=1.46,
            root_chord=0.76,
            tip_chord=0.48,
            root_thick=0.075,
            tip_thick=0.045,
            sweep_z=0.14,
            dihedral_y=0.05,
            mat_upper=wi_washi,
            mat_lower=wi_cream,
            mat_leading=wi_gold,
            origin=(0.0, 0.0, -0.12)
        )

        # 9 Pleated origami secondary panels with black crane trailing tips & gold ribs
        for si in range(9):
            u = (si + 0.35) / 9.0
            sx = side * (0.10 + u * 1.32)
            add_feather_3d(
                bm,
                origin=(sx, 0.01, 0.06 + u * 0.12),
                length=0.68 - u * 0.10,
                width=0.17,
                thickness=0.016,
                rot_xyz=(-0.01, side * (0.04 + u * 0.08), side * 0.06),
                camber=0.0,
                mat_base=wi_washi if si % 2 == 0 else wi_cream,
                mat_bar=wi_gold,
                mat_tip=wi_ink,
                tip_sharpness=0.92
            )
        finalize_bmesh(bm, obj_wi)

        obj_wo = create_part_object(f"origami_glider__wing_outer_{tag}")
        bm = bmesh.new()
        wo_washi = ensure_mat_slot(obj_wo, m_washi)
        wo_cream = ensure_mat_slot(obj_wo, m_cream)
        wo_gold = ensure_mat_slot(obj_wo, m_gold)
        wo_ink = ensure_mat_slot(obj_wo, m_ink)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=1.28,
            root_chord=0.46,
            tip_chord=0.14,
            root_thick=0.045,
            tip_thick=0.018,
            sweep_z=0.42,
            dihedral_y=-0.02,
            mat_upper=wo_washi,
            mat_lower=wo_cream,
            mat_leading=wo_gold,
            origin=(0.0, 0.0, -0.04)
        )

        for pi in range(7):
            v = pi / 6.0
            px = side * (0.08 + v * 1.12)
            pz = 0.04 + v * 0.32
            add_feather_3d(
                bm,
                origin=(px, 0.0, pz),
                length=0.82 - v * 0.18,
                width=0.15 - v * 0.03,
                thickness=0.014,
                rot_xyz=(0.01, side * (0.18 + v * 0.72), 0.0),
                camber=0.0,
                mat_base=wo_washi,
                mat_bar=wo_ink,
                mat_tip=wo_gold,
                tip_sharpness=0.96
            )
        finalize_bmesh(bm, obj_wo)

    # --- TAIL (5 articulated origami crane tail pleats) ---
    for ti in range(5):
        obj_tf = create_part_object(f"origami_glider__tail_{ti}")
        bm = bmesh.new()
        tf_washi = ensure_mat_slot(obj_tf, m_washi)
        tf_cream = ensure_mat_slot(obj_tf, m_cream)
        tf_gold = ensure_mat_slot(obj_tf, m_gold)
        tf_ink = ensure_mat_slot(obj_tf, m_ink)

        factor = (ti / 4.0) * 2.0 - 1.0
        add_feather_3d(
            bm,
            origin=(0.0, 0.0, 0.0),
            length=1.02 + abs(factor) * 0.14,
            width=0.24,
            thickness=0.018,
            rot_xyz=(0.04, 0.0, factor * 0.16),
            camber=0.0,
            mat_base=tf_washi if ti % 2 == 0 else tf_cream,
            mat_bar=tf_ink,
            mat_tip=tf_gold,
            tip_sharpness=0.94
        )
        finalize_bmesh(bm, obj_tf)


# ===========================================================================
# 3. CRIMSON FALCON — Peregrine Falcon (Falco peregrinus) & Golden Eagle
# ===========================================================================
def build_crimson_falcon():
    m_crimson = make_material("falcon_crimson", 0xDC2626, roughness=0.42)
    m_burgundy = make_material("falcon_burgundy", 0x7F1D1D, roughness=0.48)
    m_flame = make_material("falcon_flame", 0xF97316, roughness=0.38)
    m_solar = make_material("falcon_solar", 0xFACC15, roughness=0.25, metalness=0.15, emissive_hex=0xFACC15, emissive_strength=0.48)
    m_obsidian = make_material("falcon_obsidian", 0x18181B, roughness=0.35)
    m_buff = make_material("falcon_buff", 0xFEF08A, roughness=0.50)
    m_eye = make_material("falcon_eye", 0xFDE047, roughness=0.1, emissive_hex=0xFACC15, emissive_strength=0.8)

    # --- BODY ---
    obj_body = create_part_object("crimson_falcon__body")
    bm = bmesh.new()
    i_crimson = ensure_mat_slot(obj_body, m_crimson)
    i_burgundy = ensure_mat_slot(obj_body, m_burgundy)
    i_flame = ensure_mat_slot(obj_body, m_flame)
    i_solar = ensure_mat_slot(obj_body, m_solar)
    i_obsidian = ensure_mat_slot(obj_body, m_obsidian)
    i_buff = ensure_mat_slot(obj_body, m_buff)

    # Muscular teardrop high-speed stooping raptor body
    stations = [
        (-1.14, 0.25, 0.22, 0.22, 0.10),
        (-0.88, 0.44, 0.36, 0.42, 0.04),
        (-0.52, 0.60, 0.48, 0.56, -0.02),
        (-0.14, 0.56, 0.45, 0.50, -0.01),
        (0.26,  0.46, 0.37, 0.38, 0.01),
        (0.64,  0.34, 0.27, 0.26, 0.02),
        (0.96,  0.22, 0.17, 0.16, 0.02),
        (1.20,  0.12, 0.09, 0.09, 0.02),
    ]
    rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.34, n_segs=20) for (z, rx, ryd, ryv, y0) in stations]

    def falcon_body_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        if cos_a > 0.20:
            return i_burgundy if ri in (0, 3, 5) else i_crimson
        if cos_a > -0.25:
            return i_flame
        # Barred raptor breast
        return i_buff if ri % 2 == 0 else i_flame

    add_lofted_rings(bm, rings, falcon_body_mat, cap_start=True, cap_end=True)

    # Raptor dorsal armor plumes
    for row in range(4):
        z_row = -0.50 + row * 0.34
        for side in (-1, 1):
            add_feather_3d(
                bm,
                origin=(side * 0.20, 0.41 - row * 0.04, z_row),
                length=0.48,
                width=0.20,
                thickness=0.022,
                rot_xyz=(-0.06, side * 0.10, -side * 0.22),
                camber=0.02,
                mat_base=i_crimson,
                mat_bar=i_flame,
                mat_tip=i_solar,
                tip_sharpness=0.82
            )

    add_tucked_legs(bm, y_base=-0.38, z_base=0.44, spread_x=0.20, scale=1.12, mat_leg=i_solar, mat_claw=i_obsidian)
    finalize_bmesh(bm, obj_body)

    # --- HEAD (local to headGroup at (0, 0.16, -1.16)) ---
    obj_head = create_part_object("crimson_falcon__head")
    bm = bmesh.new()
    h_crimson = ensure_mat_slot(obj_head, m_crimson)
    h_burgundy = ensure_mat_slot(obj_head, m_burgundy)
    h_flame = ensure_mat_slot(obj_head, m_flame)
    h_solar = ensure_mat_slot(obj_head, m_solar)
    h_obsidian = ensure_mat_slot(obj_head, m_obsidian)
    h_buff = ensure_mat_slot(obj_head, m_buff)
    h_eye = ensure_mat_slot(obj_head, m_eye)

    head_stations = [
        (0.28,  0.28, 0.24, 0.22, -0.01),
        (0.08,  0.36, 0.31, 0.26, 0.02),
        (-0.14, 0.35, 0.30, 0.24, 0.02),
        (-0.34, 0.26, 0.22, 0.18, 0.00),
    ]
    h_rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.12, n_segs=18) for (z, rx, ryd, ryv, y0) in head_stations]

    def falcon_head_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        # Peregrine dark malar tear-stripe on cheek
        if ri in (1, 2) and -0.35 < cos_a < 0.15:
            return h_obsidian
        if cos_a > 0.15:
            return h_burgundy if ri == 1 else h_crimson
        return h_buff

    add_lofted_rings(bm, h_rings, falcon_head_mat, cap_start=True, cap_end=True)

    # Fierce supraorbital raptor brow ridges + swept nape plumes
    for side in (-1, 1):
        add_feather_3d(
            bm,
            origin=(side * 0.22, 0.18, -0.30),
            length=0.38,
            width=0.11,
            thickness=0.028,
            rot_xyz=(0.12, side * 0.18, -side * 0.35),
            camber=0.02,
            mat_base=h_flame,
            mat_tip=h_solar,
            tip_sharpness=0.88
        )
        add_eye_with_glint(bm, side, (0.25, 0.06, -0.18), 0.078, h_obsidian, h_eye, h_obsidian, h_buff)

    # Golden cere + strongly hooked Peregrine raptor beak with tomial tooth
    hook_rings = [
        [(0.0, 0.11, -0.30), (0.14, 0.01, -0.30), (0.0, -0.05, -0.30), (-0.14, 0.01, -0.30)],
        [(0.0, 0.09, -0.46), (0.11, 0.0, -0.46), (0.0, -0.05, -0.46), (-0.11, 0.0, -0.46)],
        [(0.0, 0.04, -0.66), (0.065, -0.03, -0.66), (0.0, -0.07, -0.66), (-0.065, -0.03, -0.66)],
        [(0.0, -0.06, -0.78), (0.028, -0.09, -0.78), (0.0, -0.13, -0.78), (-0.028, -0.09, -0.78)],
        [(0.0, -0.16, -0.81), (0.008, -0.17, -0.81), (0.0, -0.19, -0.81), (-0.008, -0.17, -0.81)],
    ]
    add_lofted_rings(bm, hook_rings, lambda ri, *_: h_solar if ri <= 1 else h_obsidian, cap_start=True, cap_end=True)
    finalize_bmesh(bm, obj_head)

    # --- WINGS (Swept falcate Peregrine wings) ---
    for side, tag in [(-1, "L"), (1, "R")]:
        obj_wi = create_part_object(f"crimson_falcon__wing_inner_{tag}")
        bm = bmesh.new()
        wi_crimson = ensure_mat_slot(obj_wi, m_crimson)
        wi_burgundy = ensure_mat_slot(obj_wi, m_burgundy)
        wi_flame = ensure_mat_slot(obj_wi, m_flame)
        wi_solar = ensure_mat_slot(obj_wi, m_solar)
        wi_obsidian = ensure_mat_slot(obj_wi, m_obsidian)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=1.24,
            root_chord=0.72,
            tip_chord=0.48,
            root_thick=0.13,
            tip_thick=0.08,
            sweep_z=0.38,
            dihedral_y=0.04,
            mat_upper=wi_crimson,
            mat_lower=wi_flame,
            mat_leading=wi_burgundy,
            origin=(0.0, 0.0, -0.06)
        )

        for ci in range(7):
            u = (ci + 0.4) / 7.0
            cx = side * (0.10 + u * 1.06)
            cz = -0.04 + u * 0.34
            add_feather_3d(
                bm,
                origin=(cx, 0.04, cz),
                length=0.46,
                width=0.16,
                thickness=0.018,
                rot_xyz=(-0.02, side * 0.14, 0.0),
                camber=0.025,
                mat_base=wi_burgundy,
                mat_bar=wi_crimson,
                mat_tip=wi_flame,
                tip_sharpness=0.78
            )

        for si in range(8):
            u = (si + 0.35) / 8.0
            sx = side * (0.08 + u * 1.10)
            sz = 0.08 + u * 0.36
            add_feather_3d(
                bm,
                origin=(sx, 0.0, sz),
                length=0.68 - u * 0.08,
                width=0.17,
                thickness=0.019,
                rot_xyz=(-0.02, side * (0.12 + u * 0.10), 0.0),
                camber=0.03,
                asymmetry=0.35 * side,
                mat_base=wi_crimson,
                mat_bar=wi_obsidian,
                mat_tip=wi_solar,
                tip_sharpness=0.82
            )
        finalize_bmesh(bm, obj_wi)

        obj_wo = create_part_object(f"crimson_falcon__wing_outer_{tag}")
        bm = bmesh.new()
        wo_crimson = ensure_mat_slot(obj_wo, m_crimson)
        wo_flame = ensure_mat_slot(obj_wo, m_flame)
        wo_solar = ensure_mat_slot(obj_wo, m_solar)
        wo_obsidian = ensure_mat_slot(obj_wo, m_obsidian)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=0.92,
            root_chord=0.46,
            tip_chord=0.14,
            root_thick=0.078,
            tip_thick=0.028,
            sweep_z=0.52,
            dihedral_y=-0.02,
            mat_upper=wo_crimson,
            mat_lower=wo_flame,
            mat_leading=wo_solar,
            origin=(0.0, 0.0, -0.02)
        )

        # Razor alula blades
        for ai in range(2):
            add_feather_3d(
                bm,
                origin=(side * (0.08 + ai * 0.12), 0.025, -0.12 + ai * 0.06),
                length=0.34 - ai * 0.05,
                width=0.09,
                thickness=0.016,
                rot_xyz=(0.0, side * 0.62, 0.0),
                camber=0.02,
                mat_base=wo_flame,
                mat_tip=wo_solar,
                tip_sharpness=0.92
            )

        # 7 Scythe-like primary feathers
        for pi in range(7):
            v = pi / 6.0
            px = side * (0.08 + v * 0.78)
            pz = 0.06 + v * 0.42
            plen = 0.82 + math.sin(v * math.pi * 0.8) * 0.38
            add_feather_3d(
                bm,
                origin=(px, 0.0, pz),
                length=plen,
                width=0.15 - v * 0.03,
                thickness=0.018,
                rot_xyz=(0.02, side * (0.28 + v * 0.78), 0.0),
                camber=0.028,
                asymmetry=0.45 * side,
                mat_base=wo_crimson if pi < 4 else wo_flame,
                mat_bar=wo_obsidian,
                mat_tip=wo_solar,
                tip_sharpness=0.92
            )
        finalize_bmesh(bm, obj_wo)

    # --- TAIL (7 articulated forked raptor/kite rectrices) ---
    for ti in range(7):
        obj_tf = create_part_object(f"crimson_falcon__tail_{ti}")
        bm = bmesh.new()
        tf_crimson = ensure_mat_slot(obj_tf, m_crimson)
        tf_flame = ensure_mat_slot(obj_tf, m_flame)
        tf_solar = ensure_mat_slot(obj_tf, m_solar)
        tf_obsidian = ensure_mat_slot(obj_tf, m_obsidian)

        factor = (ti / 6.0) * 2.0 - 1.0
        # Forked swallow-raptor profile: outer feathers longest
        t_len = 0.78 + (abs(factor) ** 1.35) * 0.52
        add_feather_3d(
            bm,
            origin=(0.0, 0.0, 0.0),
            length=t_len,
            width=0.19,
            thickness=0.020,
            rot_xyz=(0.0, factor * 0.08, factor * 0.10),
            camber=0.02,
            asymmetry=0.35 * factor,
            mat_base=tf_crimson if abs(factor) > 0.4 else tf_flame,
            mat_bar=tf_obsidian,
            mat_tip=tf_solar,
            tip_sharpness=0.88
        )
        finalize_bmesh(bm, obj_tf)


# ===========================================================================
# 4. EMERALD OWL SPIRIT — Barn Owl (Tyto alba) & Great Horned Owl
# ===========================================================================
def build_emerald_owl():
    m_emerald = make_material("owl_emerald", 0x10B981, roughness=0.55)
    m_forest = make_material("owl_forest", 0x065F46, roughness=0.60)
    m_mint = make_material("owl_mint", 0x6EE7B7, roughness=0.52)
    m_disc = make_material("owl_disc", 0xECFDF5, roughness=0.58)
    m_cyan = make_material("owl_cyan", 0x00F0FF, roughness=0.20, emissive_hex=0x00F0FF, emissive_strength=0.75)
    m_beak = make_material("owl_beak", 0x0F172A, roughness=0.35)
    m_pupil = make_material("owl_pupil", 0x020617, roughness=0.08)

    # --- BODY ---
    obj_body = create_part_object("emerald_owl__body")
    bm = bmesh.new()
    i_emerald = ensure_mat_slot(obj_body, m_emerald)
    i_forest = ensure_mat_slot(obj_body, m_forest)
    i_mint = ensure_mat_slot(obj_body, m_mint)
    i_disc = ensure_mat_slot(obj_body, m_disc)
    i_cyan = ensure_mat_slot(obj_body, m_cyan)
    i_beak = ensure_mat_slot(obj_body, m_beak)

    # Plush, broad-chested nocturnal owl body
    stations = [
        (-0.92, 0.34, 0.30, 0.28, 0.14),
        (-0.64, 0.56, 0.46, 0.50, 0.06),
        (-0.28, 0.68, 0.54, 0.60, 0.00),
        (0.10,  0.64, 0.50, 0.54, -0.01),
        (0.46,  0.52, 0.40, 0.42, 0.00),
        (0.76,  0.34, 0.26, 0.26, 0.01),
        (0.98,  0.18, 0.14, 0.14, 0.01),
    ]
    rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.14, n_segs=20) for (z, rx, ryd, ryv, y0) in stations]

    def owl_body_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        if cos_a > 0.18:
            return i_forest if (ri + si) % 3 == 0 else i_emerald
        if cos_a > -0.22:
            return i_emerald if ri % 2 == 0 else i_mint
        # Speckled mint-cream breast with bioluminescent chevrons
        if ri in (1, 3) and si % 2 == 0:
            return i_disc
        return i_mint

    add_lofted_rings(bm, rings, owl_body_mat, cap_start=True, cap_end=True)

    # Bioluminescent breast & dorsal rune feathers
    for row in range(3):
        for side in (-1, 1):
            add_feather_3d(
                bm,
                origin=(side * (0.20 + row * 0.06), -0.34 + row * 0.04, -0.45 + row * 0.28),
                length=0.34,
                width=0.16,
                thickness=0.020,
                rot_xyz=(0.08, side * 0.08, side * 0.15),
                camber=0.02,
                mat_base=i_mint,
                mat_bar=i_forest,
                mat_tip=i_cyan,
                tip_sharpness=0.5
            )

    add_tucked_legs(bm, y_base=-0.42, z_base=0.36, spread_x=0.22, scale=1.15, mat_leg=i_disc, mat_claw=i_beak)
    finalize_bmesh(bm, obj_body)

    # --- HEAD (local to headGroup at (0, 0.34, -0.82)) ---
    obj_head = create_part_object("emerald_owl__head")
    bm = bmesh.new()
    h_emerald = ensure_mat_slot(obj_head, m_emerald)
    h_forest = ensure_mat_slot(obj_head, m_forest)
    h_mint = ensure_mat_slot(obj_head, m_mint)
    h_disc = ensure_mat_slot(obj_head, m_disc)
    h_cyan = ensure_mat_slot(obj_head, m_cyan)
    h_beak = ensure_mat_slot(obj_head, m_beak)
    h_pupil = ensure_mat_slot(obj_head, m_pupil)

    # Broad rounded owl cranium
    owl_head_stations = [
        (0.34,  0.34, 0.30, 0.26, -0.02),
        (0.12,  0.48, 0.42, 0.36, 0.02),
        (-0.12, 0.52, 0.44, 0.38, 0.03),
        (-0.34, 0.44, 0.38, 0.32, 0.01),
        (-0.46, 0.28, 0.24, 0.20, -0.01),
    ]
    oh_rings = [make_avian_cross_section(z, rx, ryd, ryv, y0, keel_sharpness=0.05, n_segs=20) for (z, rx, ryd, ryv, y0) in owl_head_stations]

    def owl_head_mat(ri, si, nr, ns):
        ang = (si / ns) * math.tau
        cos_a = math.cos(ang)
        if ri >= 3:
            return h_disc if abs(cos_a) < 0.75 else h_mint
        return h_forest if cos_a > 0.35 and si % 2 == 0 else h_emerald

    add_lofted_rings(bm, oh_rings, owl_head_mat, cap_start=True, cap_end=True)

    # Sculpted Heart-Shaped Facial Disc Rim (Tyto alba style) on front of face (z ~ -0.40)
    n_disc = 24
    disc_outer = []
    disc_inner = []
    disc_center = []
    for i in range(n_disc):
        t = (i / n_disc) * math.tau
        # Parametric heart-disc contour
        hx = 0.44 * math.sin(t) * (1.0 + 0.12 * math.cos(t))
        hy = 0.36 * math.cos(t) - 0.08 * math.cos(2 * t) + 0.03
        disc_outer.append((hx * 1.08, hy * 1.08, -0.35))
        disc_inner.append((hx * 0.88, hy * 0.88, -0.43))
        disc_center.append((hx * 0.25, hy * 0.25 - 0.04, -0.39))
    add_lofted_rings(bm, [disc_outer, disc_inner, disc_center], lambda ri, *_: h_forest if ri == 0 else h_disc, cap_start=False, cap_end=True)

    # Great Horned Owl Plumicorns (2 x 4 layered ear-tuft feathers with glowing cyan tips)
    for side in (-1, 1):
        for pf in range(4):
            u = pf / 3.0
            add_feather_3d(
                bm,
                origin=(side * (0.26 + u * 0.12), 0.32 - u * 0.04, -0.18 + u * 0.08),
                length=0.54 - u * 0.07,
                width=0.14 - u * 0.02,
                thickness=0.024,
                rot_xyz=(0.55 - u * 0.10, side * (0.42 + u * 0.15), -side * 0.35),
                camber=0.03,
                mat_base=h_forest if pf % 2 == 0 else h_emerald,
                mat_bar=h_mint,
                mat_tip=h_cyan,
                tip_sharpness=0.88
            )

        # Forward-facing tubular owl eyes at z = -0.43
        eye_x = side * 0.21
        eye_y = 0.06
        eye_rings = [
            [(eye_x + math.sin(a) * 0.165, eye_y + math.cos(a) * 0.165, -0.38) for a in [k * math.tau / 12 for k in range(12)]],
            [(eye_x + math.sin(a) * 0.145, eye_y + math.cos(a) * 0.145, -0.45) for a in [k * math.tau / 12 for k in range(12)]],
            [(eye_x + math.sin(a) * 0.115, eye_y + math.cos(a) * 0.115, -0.48) for a in [k * math.tau / 12 for k in range(12)]],
            [(eye_x + math.sin(a) * 0.055, eye_y + math.cos(a) * 0.055, -0.50) for a in [k * math.tau / 12 for k in range(12)]],
        ]
        add_lofted_rings(
            bm,
            eye_rings,
            lambda ri, *_: h_forest if ri == 0 else (h_cyan if ri == 1 else h_pupil),
            cap_start=False,
            cap_end=True
        )

    # Downward-hooked owl beak nestled between facial discs
    owl_beak_rings = [
        [(0.0, 0.02, -0.40), (0.08, -0.04, -0.40), (0.0, -0.10, -0.40), (-0.08, -0.04, -0.40)],
        [(0.0, -0.02, -0.54), (0.045, -0.07, -0.54), (0.0, -0.13, -0.54), (-0.045, -0.07, -0.54)],
        [(0.0, -0.14, -0.60), (0.01, -0.16, -0.60), (0.0, -0.18, -0.60), (-0.01, -0.16, -0.60)],
    ]
    add_lofted_rings(bm, owl_beak_rings, lambda *_: h_beak, cap_start=True, cap_end=True)
    finalize_bmesh(bm, obj_head)

    # --- WINGS (Broad, rounded, silent-flight owl wings with comb leading edge & cyan tips) ---
    for side, tag in [(-1, "L"), (1, "R")]:
        obj_wi = create_part_object(f"emerald_owl__wing_inner_{tag}")
        bm = bmesh.new()
        wi_emerald = ensure_mat_slot(obj_wi, m_emerald)
        wi_forest = ensure_mat_slot(obj_wi, m_forest)
        wi_mint = ensure_mat_slot(obj_wi, m_mint)
        wi_disc = ensure_mat_slot(obj_wi, m_disc)
        wi_cyan = ensure_mat_slot(obj_wi, m_cyan)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=1.28,
            root_chord=0.86,
            tip_chord=0.68,
            root_thick=0.15,
            tip_thick=0.10,
            sweep_z=0.06,
            dihedral_y=0.04,
            mat_upper=wi_emerald,
            mat_lower=wi_disc,
            mat_leading=wi_forest,
            origin=(0.0, 0.0, -0.10)
        )

        # Scalloped owl covert rows
        for row in range(2):
            for ci in range(7):
                u = (ci + 0.4) / 7.0
                cx = side * (0.10 + u * 1.12)
                add_feather_3d(
                    bm,
                    origin=(cx, 0.05 - row * 0.018, -0.14 + row * 0.16),
                    length=0.44,
                    width=0.20,
                    thickness=0.020,
                    rot_xyz=(-0.02, side * 0.03, 0.0),
                    camber=0.03,
                    mat_base=wi_forest if (ci + row) % 2 == 0 else wi_emerald,
                    mat_bar=wi_mint,
                    mat_tip=wi_cyan if row == 1 and ci % 2 == 0 else wi_mint,
                    tip_sharpness=0.25
                )

        # 9 Broad, rounded secondary owl remiges
        for si in range(9):
            u = (si + 0.35) / 9.0
            sx = side * (0.08 + u * 1.16)
            add_feather_3d(
                bm,
                origin=(sx, 0.0, 0.08 + u * 0.04),
                length=0.76 - u * 0.04,
                width=0.22,
                thickness=0.022,
                rot_xyz=(-0.01, side * (0.02 + u * 0.05), 0.0),
                camber=0.04,
                asymmetry=0.25 * side,
                mat_base=wi_emerald if si % 2 == 0 else wi_forest,
                mat_bar=wi_mint,
                mat_tip=wi_cyan,
                tip_sharpness=0.22
            )
        finalize_bmesh(bm, obj_wi)

        obj_wo = create_part_object(f"emerald_owl__wing_outer_{tag}")
        bm = bmesh.new()
        wo_emerald = ensure_mat_slot(obj_wo, m_emerald)
        wo_forest = ensure_mat_slot(obj_wo, m_forest)
        wo_mint = ensure_mat_slot(obj_wo, m_mint)
        wo_disc = ensure_mat_slot(obj_wo, m_disc)
        wo_cyan = ensure_mat_slot(obj_wo, m_cyan)

        add_wing_airfoil_patagium(
            bm,
            side=side,
            span=0.82,
            root_chord=0.62,
            tip_chord=0.28,
            root_thick=0.095,
            tip_thick=0.042,
            sweep_z=0.14,
            dihedral_y=0.02,
            mat_upper=wo_emerald,
            mat_lower=wo_disc,
            mat_leading=wo_forest,
            origin=(0.0, 0.0, -0.06)
        )

        # Plush owl alula
        for ai in range(3):
            add_feather_3d(
                bm,
                origin=(side * (0.06 + ai * 0.10), 0.03, -0.22 + ai * 0.04),
                length=0.32 - ai * 0.03,
                width=0.12,
                thickness=0.018,
                rot_xyz=(0.02, side * 0.65, 0.0),
                camber=0.03,
                mat_base=wo_forest,
                mat_tip=wo_cyan,
                tip_sharpness=0.35
            )

        # 7 Rounded silent-flight primary feathers with glowing cyan tips
        for pi in range(7):
            v = pi / 6.0
            px = side * (0.08 + v * 0.68)
            pz = 0.08 - v * 0.06
            plen = 0.80 + math.sin(v * math.pi * 0.85) * 0.26
            add_feather_3d(
                bm,
                origin=(px, 0.0, pz),
                length=plen,
                width=0.20 - v * 0.03,
                thickness=0.020,
                rot_xyz=(0.01, side * (0.12 + v * 0.88), 0.0),
                camber=0.038,
                asymmetry=0.38 * side,
                mat_base=wo_forest if pi % 2 == 0 else wo_emerald,
                mat_bar=wo_mint,
                mat_tip=wo_cyan,
                tip_sharpness=0.32
            )
        finalize_bmesh(bm, obj_wo)

    # --- TAIL (7 broad rounded owl rectrices) ---
    for ti in range(7):
        obj_tf = create_part_object(f"emerald_owl__tail_{ti}")
        bm = bmesh.new()
        tf_emerald = ensure_mat_slot(obj_tf, m_emerald)
        tf_forest = ensure_mat_slot(obj_tf, m_forest)
        tf_mint = ensure_mat_slot(obj_tf, m_mint)
        tf_cyan = ensure_mat_slot(obj_tf, m_cyan)

        factor = (ti / 6.0) * 2.0 - 1.0
        t_len = 0.96 - (factor * factor) * 0.16
        add_feather_3d(
            bm,
            origin=(0.0, 0.0, 0.0),
            length=t_len,
            width=0.26,
            thickness=0.022,
            rot_xyz=(0.0, 0.0, factor * 0.06),
            camber=0.035,
            asymmetry=0.22 * factor,
            mat_base=tf_emerald if ti % 2 == 0 else tf_forest,
            mat_bar=tf_mint,
            mat_tip=tf_cyan,
            tip_sharpness=0.25
        )
        finalize_bmesh(bm, obj_tf)


def main():
    clear_scene()
    build_sky_jay()
    build_origami_glider()
    build_crimson_falcon()
    build_emerald_owl()

    out_dir = "/home/bl3ne/.gemini/antigravity/scratch/skypeck"
    blend_path = os.path.join(out_dir, "skypeck_birds.blend")
    glb_path = os.path.join(out_dir, "skypeck_birds.glb")

    total_verts = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == "MESH")
    total_faces = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    print(f"[Skypeck Birds] Built {len(bpy.data.objects)} articulated bird parts ({total_verts} verts, {total_faces} faces)")

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_yup=True,
        export_apply=True,
        export_materials="EXPORT",
    )
    print(f"[Skypeck Birds] Exported {glb_path} ({os.path.getsize(glb_path)} bytes)")


if __name__ == "__main__":
    main()
