"""
Procedural high-detail 3D world generator for Skypeck (Race One) in Blender 5.3+.
Generates both skypeck_world.glb and skypeck_world.blend with exact 1:1 Three.js Y-up coordinates.
"""

import math
import os
import random
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler


# ---------------------------------------------------------------------------
# Coordinate & Terrain Math (1:1 parity with index.html WorldGenerator)
# ---------------------------------------------------------------------------
def to_blender(x, y, z):
    """Convert Three.js Y-up (x, y, z) to Blender Z-up (x, -z, y)."""
    return Vector((x, -z, y))


def hash_2d(x, z):
    s = math.sin(x * 127.1 + z * 311.7) * 43758.5453123
    return s - math.floor(s)


def value_noise(x, z):
    xi = math.floor(x)
    zi = math.floor(z)
    xf = x - xi
    zf = z - zi
    u = xf * xf * (3.0 - 2.0 * xf)
    v = zf * zf * (3.0 - 2.0 * zf)
    a = hash_2d(xi, zi)
    b = hash_2d(xi + 1, zi)
    c = hash_2d(xi, zi + 1)
    d = hash_2d(xi + 1, zi + 1)
    return (a * (1.0 - u) + b * u) * (1.0 - v) + (c * (1.0 - u) + d * u) * v


def fbm(x, z, octaves=3):
    val = 0.0
    amp = 0.5
    freq = 1.0
    for _ in range(octaves):
        val += amp * (value_noise(x * freq, z * freq) * 2.0 - 1.0)
        freq *= 2.07
        amp *= 0.5
    return val


def smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def dist_to_segment(px, pz, ax, az, bx, bz):
    dx = bx - ax
    dz = bz - az
    len_sq = dx * dx + dz * dz
    if len_sq == 0.0:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / len_sq))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def get_biome_weights(x, z):
    bay_dist = math.hypot((x + 65.0) * 0.9, (z + 255.0) * 1.1)
    bay_weight = 1.0 - smoothstep(210.0, 325.0, bay_dist)

    chasm_dist = math.hypot(x - 380.0, z + 140.0)
    chasm_weight = 1.0 - smoothstep(75.0, 138.0, chasm_dist)

    d_can1 = dist_to_segment(x, z, 388.0, 10.0, 335.0, 215.0)
    d_can2 = dist_to_segment(x, z, 335.0, 215.0, 215.0, 315.0)
    canyon_track_dist = min(d_can1, d_can2)
    canyon_weight = 1.0 - smoothstep(65.0, 125.0, canyon_track_dist)

    return bay_weight, bay_dist, chasm_weight, chasm_dist, canyon_weight, canyon_track_dist


def get_terrain_height(x, z):
    bay_weight, _, chasm_weight, chasm_dist, canyon_weight, canyon_track_dist = get_biome_weights(x, z)

    h = 21.0 + fbm(x * 0.008, z * 0.008) * 16.0 + fbm(x * 0.024, z * 0.024) * 4.0

    start_hill_dist = math.hypot(x + 40.0, z - 355.0)
    start_hill = (1.0 - smoothstep(25.0, 130.0, start_hill_dist)) * 16.0
    h += start_hill

    if bay_weight > 0.001:
        bay_h = -10.5 + fbm(x * 0.015, z * 0.015) * 3.5
        islands = [
            (-245.0, -185.0, 52.0, 11.0),
            (-75.0, -290.0, 62.0, 14.0),
            (115.0, -275.0, 54.0, 12.0),
            (-30.0, -145.0, 44.0, 8.0),
        ]
        for ix, iz, ir, ih in islands:
            d = math.hypot(x - ix, z - iz)
            rise = (1.0 - smoothstep(ir * 0.3, ir, d)) * (ih + 10.5)
            bay_h = max(bay_h, -10.5 + rise)
        h = h * (1.0 - bay_weight) + bay_h * bay_weight

    if chasm_weight > 0.001:
        rim = (1.0 - smoothstep(78.0, 128.0, chasm_dist)) * smoothstep(52.0, 92.0, chasm_dist) * 28.0
        pit = (1.0 - smoothstep(42.0, 82.0, chasm_dist)) * -52.0
        chasm_target = max(-18.5, h + rim + pit)
        h = h * (1.0 - chasm_weight * 0.92) + chasm_target * (chasm_weight * 0.92)

    if canyon_weight > 0.001:
        floor_factor = 1.0 - smoothstep(22.0, 38.0, canyon_track_dist)
        wall_factor = (1.0 - smoothstep(48.0, 88.0, canyon_track_dist)) * smoothstep(24.0, 52.0, canyon_track_dist)
        canyon_h = 5.5 + wall_factor * 56.0 + fbm(x * 0.04, z * 0.04) * 5.5
        blend = canyon_weight * max(floor_factor, wall_factor * 0.95)
        h = h * (1.0 - blend) + canyon_h * blend

    outer_r = math.hypot(x, z)
    if outer_r > 500.0:
        h += smoothstep(500.0, 650.0, outer_r) * 45.0

    # Extra rocky strata micro-relief away from start hill for high-resolution sculpt detail
    if start_hill_dist > 45.0 and bay_weight < 0.35:
        strata = math.sin(h * 0.45 + fbm(x * 0.05, z * 0.05, 2) * 2.0) * 0.45
        h += strata * smoothstep(28.0, 55.0, h)

    return h


RAW_COURSE_POINTS = [
    (-40.0, 40.0, 355.0),
    (-175.0, 42.0, 330.0),
    (-305.0, 36.0, 235.0),
    (-375.0, 30.0, 105.0),
    (-370.0, 22.0, -65.0),
    (-290.0, 18.0, -230.0),
    (-140.0, 26.0, -345.0),
    (35.0, 22.0, -375.0),
    (210.0, 28.0, -335.0),
    (345.0, 36.0, -245.0),
    (395.0, 16.0, -155.0),
    (365.0, 2.0, -95.0),
    (395.0, 22.0, -30.0),
    (385.0, 20.0, 105.0),
    (325.0, 19.0, 225.0),
    (220.0, 22.0, 315.0),
    (95.0, 31.0, 355.0),
]


def min_course_dist_xz(x, z):
    best = 9999.0
    n = len(RAW_COURSE_POINTS)
    for i in range(n):
        ax, _, az = RAW_COURSE_POINTS[i]
        bx, _, bz = RAW_COURSE_POINTS[(i + 1) % n]
        best = min(best, dist_to_segment(x, z, ax, az, bx, bz))
    return best


# ---------------------------------------------------------------------------
# Blender Scene & Material Helpers
# ---------------------------------------------------------------------------
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def hex_to_rgba(hex_str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    # Convert sRGB hex to linear RGB for Blender Principled BSDF
    def srgb_to_lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (srgb_to_lin(r), srgb_to_lin(g), srgb_to_lin(b), 1.0)


def make_pbr_material(name, hex_color, roughness=0.75, metallic=0.0, emission_hex=None, emission_strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = hex_to_rgba(hex_color)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if emission_hex and emission_strength > 0.0:
            em_rgba = hex_to_rgba(emission_hex)
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = em_rgba
                bsdf.inputs["Emission Strength"].default_value = emission_strength
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = em_rgba
    return mat


def create_object_from_bmesh(name, bm, materials, collection, smooth=False):
    mesh = bpy.data.meshes.new(name + "_Mesh")
    for m in materials:
        mesh.materials.append(m)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for p in mesh.polygons:
            p.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def bm_add_transformed(bm, sub_bm, mat_idx, matrix):
    """Merge geometry from sub_bm into bm with transform matrix and assigned material_index."""
    vert_map = {}
    for v in sub_bm.verts:
        nv = bm.verts.new(matrix @ v.co)
        vert_map[v] = nv
    for f in sub_bm.faces:
        try:
            nf = bm.faces.new([vert_map[v] for v in f.verts])
            nf.material_index = mat_idx
            nf.smooth = f.smooth
        except ValueError:
            pass
    sub_bm.free()


def make_box_bm(sx, sy, sz):
    sb = bmesh.new()
    bmesh.ops.create_cube(sb, size=1.0)
    bmesh.ops.scale(sb, vec=(sx, sy, sz), verts=sb.verts)
    return sb


def make_cone_bm(r1, r2, height, segs=8):
    sb = bmesh.new()
    bmesh.ops.create_cone(
        sb,
        cap_ends=True,
        cap_tris=False,
        segments=segs,
        radius1=r1,
        radius2=r2,
        depth=height,
    )
    return sb


def make_icosphere_bm(radius, subdiv=1, sx=1.0, sy=1.0, sz=1.0):
    sb = bmesh.new()
    bmesh.ops.create_icosphere(sb, subdivisions=subdiv, radius=radius)
    if sx != 1.0 or sy != 1.0 or sz != 1.0:
        bmesh.ops.scale(sb, vec=(sx, sy, sz), verts=sb.verts)
    return sb


def make_rock_bm(radius, seed=0, sx=1.0, sy=1.0, sz=1.0):
    rng = random.Random(seed)
    sb = make_icosphere_bm(radius, subdiv=2, sx=sx, sy=sy, sz=sz)
    for v in sb.verts:
        n = 0.78 + rng.random() * 0.44
        v.co *= n
    return sb


def matrix_at_three(x, y, z, rot_x=0.0, rot_y=0.0, rot_z=0.0, scale=1.0):
    pos = to_blender(x, y, z)
    # Note: Three.js Euler (rot_x, rot_y, rot_z) mapped to Blender Z-up
    eul = Euler((rot_x, -rot_z, rot_y), "XYZ")
    m = Matrix.Translation(pos) @ eul.to_matrix().to_4x4() @ Matrix.Scale(scale, 4)
    return m


# ---------------------------------------------------------------------------
# World Building Modules
# ---------------------------------------------------------------------------
def build_terrain(collection, mats):
    """Build 200x200 sculpted terrain (40,000 verts / 79,202 tris) with slope/biome PBR mats."""
    bm = bmesh.new()
    grid_n = 200
    size = 1320.0
    half = size * 0.5
    step = size / (grid_n - 1)

    verts_grid = [[None] * grid_n for _ in range(grid_n)]
    for iz in range(grid_n):
        z = -half + iz * step
        for ix in range(grid_n):
            x = -half + ix * step
            y = get_terrain_height(x, z)
            verts_grid[iz][ix] = bm.verts.new(to_blender(x, y, z))

    bm.verts.ensure_lookup_table()

    # Create properly wound quads pointing +Z in Blender (+Y in Three.js)
    for iz in range(grid_n - 1):
        for ix in range(grid_n - 1):
            v00 = verts_grid[iz][ix]
            v10 = verts_grid[iz][ix + 1]
            v11 = verts_grid[iz + 1][ix + 1]
            v01 = verts_grid[iz + 1][ix]
            bm.faces.new((v00, v01, v11, v10))

    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.normal_update()

    # Material index assignment:
    # 0: Mat_Meadow_Grass, 1: Mat_Forest_Moss, 2: Mat_Tropical_Sand,
    # 3: Mat_Cliff_Granite, 4: Mat_Canyon_Sandstone, 5: Mat_Alpine_Snow
    slope_cos_33 = math.cos(math.radians(33.0))

    for face in bm.faces:
        c_bl = face.calc_center_median()
        x = c_bl.x
        y = c_bl.z
        z = -c_bl.y
        nz = abs(face.normal.z)

        bay_weight, _, chasm_weight, _, canyon_weight, canyon_dist = get_biome_weights(x, z)

        if y > 55.0 and canyon_weight < 0.35 and nz > 0.72:
            face.material_index = 5  # Alpine snow
        elif nz < slope_cos_33 or (chasm_weight > 0.3 and y < 22.0) or (y > 57.0 and canyon_weight < 0.35):
            face.material_index = 3  # Cliff granite (>33 deg slope or chasm cliff)
        elif canyon_weight > 0.28:
            face.material_index = 3 if canyon_dist > 26.0 else 4  # Canyon granite wall vs sandstone floor
        elif bay_weight > 0.2 and y < 5.6:
            face.material_index = 2  # Tropical sand
        elif fbm(x * 0.018, z * 0.018, 2) > 0.12 and bay_weight < 0.35:
            face.material_index = 1  # Forest moss pockets
        else:
            face.material_index = 0  # Meadow grass

    obj = create_object_from_bmesh("Terrain_HighRes", bm, mats, collection, smooth=True)
    return obj


def build_zone1_sanctuary_and_forest(collection, mats):
    """Start Sanctuary shrine, Megalith Arches, 160 varied trees, rocks & mushrooms."""
    # 1. Start_Sanctuary behind/flanking the start hill perch (-40, y, 355)
    bm_sanc = bmesh.new()
    sx, sz = -20.0, 358.0
    sy = get_terrain_height(sx, sz)
    ang_course = math.atan2(-135.0, -25.0)  # rough heading along start tangent

    def place_sanc(sub_bm, mat_idx, lx, ly, lz, rx=0.0, ry=0.0, rz=0.0):
        # Local frame rotated around Y by ang_course relative to (sx, sy, sz)
        ca, sa = math.cos(ang_course), math.sin(ang_course)
        wx = sx + lx * ca + lz * sa
        wz = sz - lx * sa + lz * ca
        wy = sy + ly
        m = matrix_at_three(wx, wy, wz, rx, ry + ang_course, rz)
        bm_add_transformed(bm_sanc, sub_bm, mat_idx, m)

    # Stepped stone dais & archway flanking start
    place_sanc(make_box_bm(18.0, 1.2, 12.0), 0, 0.0, 0.2, 0.0)
    place_sanc(make_box_bm(14.5, 0.8, 9.5), 0, 0.0, 1.0, 0.0)

    # Twin carved pillars + lintel portal
    for side in (-1.0, 1.0):
        place_sanc(make_box_bm(2.2, 2.2, 2.2), 0, side * 6.5, 2.4, 0.0)
        place_sanc(make_cone_bm(0.95, 0.78, 10.5, 8), 0, side * 6.5, 8.2, 0.0)
        place_sanc(make_box_bm(2.3, 1.0, 2.3), 0, side * 6.5, 13.6, 0.0)
        # Hanging banner in royal blue & gold
        place_sanc(make_box_bm(0.18, 5.2, 1.4), 2, side * 5.0, 9.8, 0.0)
        place_sanc(make_icosphere_bm(0.65, 1), 3, side * 6.5, 14.8, 0.0)
        # Flanking wooden torii/brazier posts
        place_sanc(make_cone_bm(0.35, 0.28, 4.5, 6), 1, side * 9.5, 2.8, -5.5)
        place_sanc(make_icosphere_bm(0.55, 1), 3, side * 9.5, 5.3, -5.5)

    # Massive curved timber + stone lintel spanning the Start Sanctuary gate
    place_sanc(make_box_bm(16.5, 1.5, 2.0), 1, 0.0, 14.6, 0.0)
    place_sanc(make_box_bm(19.0, 0.9, 2.6), 0, 0.0, 15.6, 0.0)

    create_object_from_bmesh("Start_Sanctuary", bm_sanc, mats["sanctuary"], collection, smooth=False)

    # 2. Ancient Megalith Arches along Zone 1 forest ridge
    bm_mega = bmesh.new()
    arch_spots = [
        (-125.0, 340.0, 0.3),
        (-245.0, 280.0, 0.85),
        (-335.0, 175.0, 1.2),
        (-380.0, 45.0, 1.55),
    ]
    for ax, az, arot in arch_spots:
        ay = get_terrain_height(ax, az)
        ca, sa = math.cos(arot), math.sin(arot)
        for side in (-1.0, 1.0):
            px = ax + ca * side * 14.0
            pz = az - sa * side * 14.0
            py = get_terrain_height(px, pz)
            bm_add_transformed(bm_mega, make_rock_bm(3.2, int(abs(px + pz)), 1.0, 1.1, 3.8), 0, matrix_at_three(px, py + 7.5, pz, 0.0, arot, side * 0.12))
            bm_add_transformed(bm_mega, make_rock_bm(2.5, int(abs(px)), 1.2, 1.2, 2.2), 1, matrix_at_three(px, py + 15.5, pz, 0.0, arot, -side * 0.2))
        bm_add_transformed(bm_mega, make_rock_bm(3.0, int(abs(ax)), 5.8, 1.1, 1.2), 0, matrix_at_three(ax, ay + 20.5, az, 0.0, arot, 0.0))

    create_object_from_bmesh("Megalith_Arches", bm_mega, mats["megalith"], collection, smooth=False)

    # 3. 160 Multi-Variant Forest Trees (Oaks, Layered Pines, Birches)
    bm_trees = bmesh.new()
    rng = random.Random(4201)
    tree_count = 0
    attempts = 0
    while tree_count < 160 and attempts < 1400:
        attempts += 1
        ang = rng.uniform(0.0, math.pi * 2.0)
        rad = rng.uniform(35.0, 420.0)
        tx = -205.0 + math.cos(ang) * rad * 0.96
        tz = 215.0 + math.sin(ang) * rad * 0.78

        if math.hypot(tx + 40.0, tz - 355.0) < 28.0:
            continue
        if min_course_dist_xz(tx, tz) < 22.0:
            continue

        bay_w, _, chasm_w, _, canyon_w, _ = get_biome_weights(tx, tz)
        if bay_w > 0.35 or chasm_w > 0.3 or canyon_w > 0.3:
            continue

        ty = get_terrain_height(tx, tz)
        if ty < 6.0 or ty > 56.0:
            continue

        sc = rng.uniform(0.85, 1.55)
        variant = tree_count % 3

        if variant == 0:
            # Lush multi-sphere Oak
            bm_add_transformed(bm_trees, make_cone_bm(0.85 * sc, 0.42 * sc, 6.5 * sc, 7), 0, matrix_at_three(tx, ty + 3.0 * sc, tz))
            for ox, oy, oz, r, midx in [
                (0.0, 7.6, 0.0, 3.3, 2),
                (1.7, 6.4, 1.1, 2.4, 3),
                (-1.8, 6.2, -1.0, 2.3, 2),
                (0.4, 9.6, -0.5, 2.1, 3),
            ]:
                bm_add_transformed(bm_trees, make_icosphere_bm(r * sc, 1), midx, matrix_at_three(tx + ox * sc, ty + oy * sc, tz + oz * sc))
        elif variant == 1:
            # Layered Alpine Pine
            bm_add_transformed(bm_trees, make_cone_bm(0.62 * sc, 0.25 * sc, 5.5 * sc, 6), 0, matrix_at_three(tx, ty + 2.6 * sc, tz))
            for lvl, (rad_c, h_c, y_c) in enumerate([(3.2, 4.2, 5.2), (2.5, 3.8, 7.5), (1.7, 3.4, 9.6), (0.95, 2.6, 11.4)]):
                bm_add_transformed(
                    bm_trees,
                    make_cone_bm(rad_c * sc, 0.1 * sc, h_c * sc, 7),
                    4 if lvl % 2 == 0 else 2,
                    matrix_at_three(tx, ty + y_c * sc, tz, 0.0, lvl * 0.4, 0.0),
                )
        else:
            # Slender White-Barked Birch with Golden-Green Crown
            bm_add_transformed(bm_trees, make_cone_bm(0.46 * sc, 0.22 * sc, 7.4 * sc, 6), 1, matrix_at_three(tx, ty + 3.5 * sc, tz, 0.05, rng.uniform(0, 6.28), 0.0))
            for ox, oy, oz, r in [(0.0, 8.2, 0.0, 2.5), (1.3, 7.0, -0.9, 1.9), (-1.2, 7.4, 0.9, 1.8)]:
                bm_add_transformed(bm_trees, make_icosphere_bm(r * sc, 1), 5, matrix_at_three(tx + ox * sc, ty + oy * sc, tz + oz * sc))

        tree_count += 1

    create_object_from_bmesh("Forest_Trees", bm_trees, mats["trees"], collection, smooth=False)

    # 4. Zone 1 Forest Props: Amanita Mushrooms, Yellow Chanterelles, Mossy Rocks, Fallen Logs
    bm_props = bmesh.new()
    for i in range(95):
        px = -350.0 + ((i * 89.3) % 390.0)
        pz = 55.0 + ((i * 141.7) % 355.0)
        if math.hypot(px + 40.0, pz - 355.0) < 12.0:
            continue
        bay_w, _, chasm_w, _, canyon_w, _ = get_biome_weights(px, pz)
        if bay_w > 0.35 or chasm_w > 0.35 or canyon_w > 0.35:
            continue
        py = get_terrain_height(px, pz)
        if py < 6.0:
            continue
        sc = 0.85 + (i % 4) * 0.25

        if i % 5 == 0:
            # Mossy boulder
            bm_add_transformed(bm_props, make_rock_bm(1.6 * sc, i, 1.3, 0.8, 1.1), 4, matrix_at_three(px, py + 0.3 * sc, pz))
        elif i % 5 == 1:
            # Fallen timber log
            bm_add_transformed(bm_props, make_cone_bm(0.48 * sc, 0.42 * sc, 5.2 * sc, 7), 5, matrix_at_three(px, py + 0.35 * sc, pz, math.pi / 2.0, (i * 0.8), 0.0))
        elif i % 3 == 0:
            # Yellow chanterelle
            bm_add_transformed(bm_props, make_cone_bm(0.24 * sc, 0.16 * sc, 0.9 * sc, 6), 0, matrix_at_three(px, py + 0.42 * sc, pz))
            bm_add_transformed(bm_props, make_cone_bm(0.62 * sc, 0.08 * sc, 0.52 * sc, 7), 3, matrix_at_three(px, py + 0.95 * sc, pz))
        else:
            # Red Amanita mushroom with raised white dots
            bm_add_transformed(bm_props, make_cone_bm(0.42 * sc, 0.28 * sc, 1.45 * sc, 7), 0, matrix_at_three(px, py + 0.7 * sc, pz))
            bm_add_transformed(bm_props, make_icosphere_bm(0.95 * sc, 1, 1.0, 0.65, 1.0), 1, matrix_at_three(px, py + 1.48 * sc, pz))
            for s in range(6):
                sa = (s / 6.0) * math.pi * 2.0
                rr = 0.08 * sc if s == 0 else 0.56 * sc
                dy = 2.08 * sc if s == 0 else 1.76 * sc
                bm_add_transformed(bm_props, make_icosphere_bm(0.15 * sc, 0), 2, matrix_at_three(px + math.cos(sa) * rr, py + dy, pz + math.sin(sa) * rr))

    create_object_from_bmesh("Forest_Props", bm_props, mats["forest_props"], collection, smooth=False)


def build_zone2_tropical_bay(collection, mats):
    """Coastal Lighthouse, Wooden Docks, 5 Sailboats, 3 Sea-Stack Arches, 45 Coconut Palms, Corals & Clouds."""
    # 1. Lighthouse on main tropical island (-75, y, -290)
    bm_lh = bmesh.new()
    lx, lz = -75.0, -290.0
    ly = get_terrain_height(lx, lz)

    # Octagonal stone base plinth
    bm_add_transformed(bm_lh, make_cone_bm(6.8, 6.2, 3.6, 8), 0, matrix_at_three(lx, ly + 1.2, lz))
    # Striped tower segments (white/red alternating)
    for seg in range(5):
        r_bot = 4.2 - seg * 0.35
        r_top = 4.2 - (seg + 1) * 0.35
        h_seg = 4.4
        y_seg = ly + 3.0 + seg * h_seg + h_seg * 0.5
        bm_add_transformed(bm_lh, make_cone_bm(r_bot, r_top, h_seg, 12), 0 if seg % 2 == 0 else 1, matrix_at_three(lx, y_seg, lz))

    # Gallery deck & railing posts
    top_y = ly + 25.0
    bm_add_transformed(bm_lh, make_cone_bm(3.6, 3.6, 0.9, 12), 2, matrix_at_three(lx, top_y, lz))
    for p in range(10):
        pa = (p / 10.0) * math.pi * 2.0
        bm_add_transformed(bm_lh, make_box_bm(0.18, 1.3, 0.18), 2, matrix_at_three(lx + math.cos(pa) * 3.35, top_y + 1.0, lz + math.sin(pa) * 3.35))
    # Lantern glass & emissive beacon core + conical red roof
    bm_add_transformed(bm_lh, make_cone_bm(2.2, 2.2, 3.2, 10), 3, matrix_at_three(lx, top_y + 2.0, lz))
    bm_add_transformed(bm_lh, make_cone_bm(3.1, 0.15, 3.4, 12), 1, matrix_at_three(lx, top_y + 5.1, lz))

    create_object_from_bmesh("Lighthouse", bm_lh, mats["lighthouse"], collection, smooth=False)

    # 2. Wooden Docks & 5 Sailboats
    bm_boats = bmesh.new()
    dock_specs = [
        (-75.0, -242.0, 0.0),
        (-212.0, -185.0, 1.1),
        (88.0, -252.0, -0.8),
    ]
    for dx, dz, drot in dock_specs:
        ca, sa = math.cos(drot), math.sin(drot)
        for p in range(7):
            dist = p * 3.0
            px = dx + sa * dist
            pz = dz + ca * dist
            bm_add_transformed(bm_boats, make_box_bm(4.6, 0.42, 2.5), 0, matrix_at_three(px, 1.3, pz, 0.0, drot, 0.0))
            for side in (-1.0, 1.0):
                bm_add_transformed(bm_boats, make_cone_bm(0.28, 0.25, 5.5, 6), 0, matrix_at_three(px + ca * side * 2.0, -0.5, pz - sa * side * 2.0))

    boat_spots = [
        (-160.0, -210.0, 0.4),
        (-15.0, -240.0, -0.7),
        (55.0, -320.0, 1.2),
        (-260.0, -280.0, 0.9),
        (165.0, -215.0, -1.1),
    ]
    for bx, bz, brot in boat_spots:
        ca, sa = math.cos(brot), math.sin(brot)
        # Sculpted wooden boat hull + cabin + mast + mainsail + jib
        bm_add_transformed(bm_boats, make_icosphere_bm(2.2, 1, 1.0, 0.48, 2.8), 0, matrix_at_three(bx, 0.35, bz, 0.0, brot, 0.0))
        bm_add_transformed(bm_boats, make_box_bm(2.2, 1.1, 3.2), 1, matrix_at_three(bx, 1.1, bz, 0.0, brot, 0.0))
        bm_add_transformed(bm_boats, make_cone_bm(0.2, 0.1, 11.5, 6), 0, matrix_at_three(bx, 6.2, bz, 0.0, brot, 0.0))
        # Triangular billowing mainsail
        bm_add_transformed(bm_boats, make_cone_bm(2.6, 0.15, 8.8, 4), 1, matrix_at_three(bx + sa * 2.1, 6.4, bz + ca * 2.1, 0.0, brot, 0.0))
        # Forward jib sail
        bm_add_transformed(bm_boats, make_cone_bm(1.6, 0.1, 6.8, 4), 2, matrix_at_three(bx - sa * 2.2, 5.5, bz - ca * 2.2, 0.0, brot, 0.0))

    create_object_from_bmesh("Tropical_Bay_Docks_Boats", bm_boats, mats["boats"], collection, smooth=False)

    # 3. 3 Dramatic Sea-Stack Rock Arches in the Bay
    bm_sea_arches = bmesh.new()
    sea_arches = [
        (-315.0, -165.0, 0.9),
        (-135.0, -355.0, -0.25),
        (145.0, -350.0, 0.55),
    ]
    for idx, (ax, az, arot) in enumerate(sea_arches):
        ca, sa = math.cos(arot), math.sin(arot)
        for side in (-1.0, 1.0):
            px = ax + ca * side * 16.0
            pz = az - sa * side * 16.0
            bm_add_transformed(bm_sea_arches, make_rock_bm(5.8, idx * 10 + int(side + 2), 1.0, 1.0, 3.6), 0, matrix_at_three(px, 6.0, pz, 0.0, arot, side * 0.18))
            bm_add_transformed(bm_sea_arches, make_rock_bm(4.6, idx * 20 + int(side + 2), 1.2, 1.1, 2.2), 1, matrix_at_three(px * 0.99 + ax * 0.01, 18.0, pz * 0.99 + az * 0.01, 0.0, arot, -side * 0.25))
        bm_add_transformed(bm_sea_arches, make_rock_bm(4.5, idx * 33, 4.6, 1.2, 1.2), 0, matrix_at_three(ax, 25.0, az, 0.0, arot, 0.0))

    create_object_from_bmesh("Sea_Stack_Arch", bm_sea_arches, mats["sea_arch"], collection, smooth=False)

    # 4. 45 Sculpted Coconut Palms & Submerged Corals
    bm_palms = bmesh.new()
    islands = [
        (-245.0, -185.0, 44.0),
        (-75.0, -290.0, 52.0),
        (115.0, -275.0, 46.0),
        (-30.0, -145.0, 36.0),
    ]
    rng = random.Random(7702)
    palm_idx = 0
    for isl_i, (ix, iz, ir) in enumerate(islands):
        count_on_isl = 12 if isl_i != 3 else 9
        for p in range(count_on_isl):
            pa = (p / count_on_isl) * math.pi * 2.0 + isl_i * 0.5
            pr = rng.uniform(7.0, ir * 0.65)
            px = ix + math.cos(pa) * pr
            pz = iz + math.sin(pa) * pr
            if isl_i == 1 and math.hypot(px + 75.0, pz + 290.0) < 12.0:
                continue  # keep clear of Lighthouse plinth
            py = get_terrain_height(px, pz)
            if py < 0.8:
                continue
            sc = rng.uniform(0.85, 1.35)
            lean = pa + rng.uniform(-0.4, 0.4)
            cx, cy, cz = px, py, pz
            for s in range(6):
                seg_h = 1.7 * sc
                cx += math.cos(lean) * s * 0.22 * sc
                cz += math.sin(lean) * s * 0.22 * sc
                bm_add_transformed(
                    bm_palms,
                    make_cone_bm((0.46 - s * 0.04) * sc, (0.38 - s * 0.04) * sc, seg_h, 6),
                    0,
                    matrix_at_three(cx, cy + seg_h * 0.5, cz, math.sin(lean) * s * 0.07, 0.0, -math.cos(lean) * s * 0.07),
                )
                cy += seg_h * 0.92
            # Coconuts under crown
            for c_i in range(3):
                ca = (c_i / 3.0) * math.pi * 2.0
                bm_add_transformed(bm_palms, make_icosphere_bm(0.32 * sc, 0), 2, matrix_at_three(cx + math.cos(ca) * 0.45 * sc, cy - 0.35 * sc, cz + math.sin(ca) * 0.45 * sc))
            # 8 arched radial palm fronds
            for f in range(8):
                fa = (f / 8.0) * math.pi * 2.0
                bm_add_transformed(
                    bm_palms,
                    make_cone_bm(0.85 * sc, 0.05 * sc, 4.6 * sc, 4),
                    1,
                    matrix_at_three(cx + math.cos(fa) * 2.0 * sc, cy - 0.25 * sc, cz + math.sin(fa) * 2.0 * sc, 0.0, -fa, -math.pi * 0.38),
                )
            palm_idx += 1

    # Submerged coral reefs near islands
    for c_i in range(36):
        cx = -280.0 + ((c_i * 71.0) % 420.0)
        cz = -360.0 + ((c_i * 53.0) % 210.0)
        cy = get_terrain_height(cx, cz)
        if -7.0 < cy < -0.8:
            col_idx = 3 if c_i % 2 == 0 else 4
            for br in range(3):
                ba = (br / 3.0) * math.pi * 2.0
                bm_add_transformed(bm_palms, make_cone_bm(0.7, 0.3, 2.4, 6), col_idx, matrix_at_three(cx + math.cos(ba) * 0.6, cy + 0.9, cz + math.sin(ba) * 0.6, 0.3, ba, 0.25))

    create_object_from_bmesh("Tropical_Palms_Corals", bm_palms, mats["palms"], collection, smooth=False)

    # 5. Volumetric Soft Clouds floating over Tropical Bay
    bm_clouds = bmesh.new()
    cloud_coords = [
        (-320.0, 24.0, -135.0, 1.35),
        (-185.0, 20.0, -295.0, 1.55),
        (-45.0, 25.0, -345.0, 1.45),
        (110.0, 24.0, -335.0, 1.30),
        (-140.0, 42.0, -210.0, 1.70),
        (40.0, 46.0, -240.0, 1.60),
        (230.0, 52.0, -195.0, 1.45),
    ]
    for clx, cly, clz, cls in cloud_coords:
        for px, py, pz, pr in [(0, 0, 0, 7.5), (5.5, -1.0, 1.2, 5.6), (-5.8, -0.8, -1.0, 5.8), (2.2, 2.5, -2.2, 5.0), (-2.5, 1.6, 2.8, 4.8)]:
            bm_add_transformed(bm_clouds, make_icosphere_bm(pr * cls, 2), 0, matrix_at_three(clx + px * cls, cly + py * cls, clz + pz * cls))

    create_object_from_bmesh("World_Clouds", bm_clouds, mats["cloud"], collection, smooth=True)


def build_zone3_waterfall_chasm(collection, mats):
    """Suspension Bridge spanning the Circular Chasm + 5 Stone Aqueduct Outfalls + Hexagonal Basalt Columns."""
    cx, cz = 380.0, -140.0

    # 1. Chasm_Bridge spanning rim-to-rim high above dive path (y ~ 54m)
    bm_bridge = bmesh.new()
    span_half = 86.0
    deck_y = 54.0
    # Bridge runs diagonally (NE-SW orientation so it frames the circular chasm cleanly)
    b_ang = 0.55
    ca, sa = math.cos(b_ang), math.sin(b_ang)

    def place_br(sub_bm, mat_idx, lx, ly, lz, rx=0.0, ry=0.0, rz=0.0):
        wx = cx + lx * ca + lz * sa
        wz = cz - lx * sa + lz * ca
        wy = deck_y + ly
        bm_add_transformed(bm_bridge, sub_bm, mat_idx, matrix_at_three(wx, wy, wz, rx, ry + b_ang, rz))

    # Stone cliff anchor towers at both ends
    for side in (-1.0, 1.0):
        end_x = side * span_half
        end_ground = get_terrain_height(cx + end_x * ca, cz - end_x * sa)
        tower_h = max(12.0, (deck_y - end_ground) + 12.0)
        for z_rail in (-3.2, 3.2):
            place_br(make_box_bm(3.2, tower_h, 2.6), 0, end_x, 8.0 - tower_h * 0.5, z_rail)
        place_br(make_box_bm(3.0, 2.0, 9.2), 0, end_x, 9.2, 0.0)

    # Catenary wooden planks + side rope rails + vertical suspenders
    plank_count = 44
    for i in range(plank_count + 1):
        u = (i / plank_count) * 2.0 - 1.0  # -1..+1
        px = u * (span_half - 2.0)
        sag = -6.5 * (1.0 - u * u)
        place_br(make_box_bm(2.8, 0.35, 5.6), 1, px, sag, 0.0)

        # Rail posts & catenary main cables
        cable_y = 8.5 - 11.5 * (1.0 - u * u)
        for z_rail in (-2.9, 2.9):
            place_br(make_box_bm(0.24, 2.2, 0.24), 1, px, sag + 1.1, z_rail)
            if i % 2 == 0:
                susp_h = max(0.6, cable_y - sag)
                place_br(make_box_bm(0.14, susp_h, 0.14), 2, px, sag + susp_h * 0.5, z_rail)

        if i < plank_count:
            u_next = ((i + 1) / plank_count) * 2.0 - 1.0
            px_next = u_next * (span_half - 2.0)
            c_y_next = 8.5 - 11.5 * (1.0 - u_next * u_next)
            dx = px_next - px
            dy = c_y_next - cable_y
            seg_len = math.hypot(dx, dy)
            slope_ang = math.atan2(dy, dx)
            for z_rail in (-2.9, 2.9):
                place_br(make_box_bm(seg_len + 0.15, 0.24, 0.24), 2, (px + px_next) * 0.5, (cable_y + c_y_next) * 0.5, z_rail, 0.0, 0.0, slope_ang)

    create_object_from_bmesh("Chasm_Bridge", bm_bridge, mats["bridge"], collection, smooth=False)

    # 2. 5 Stone Aqueduct Outfalls on rim + Hexagonal Basalt Columns in deep lagoon
    bm_chasm = bmesh.new()
    fall_angles = [0.2, 1.15, 2.35, 3.65, 5.1]
    for a in fall_angles:
        rx = cx + math.cos(a) * 62.0
        rz = cz + math.sin(a) * 62.0
        ry = get_terrain_height(rx, rz)
        look_yaw = math.atan2(cx - rx, cz - rz)
        ca2, sa2 = math.cos(look_yaw), math.sin(look_yaw)
        # Stone sluice channel + twin pilasters at waterfall crest
        bm_add_transformed(bm_chasm, make_box_bm(16.0, 4.5, 14.0), 0, matrix_at_three(rx, max(35.0, ry), rz, 0.0, look_yaw, 0.0))
        for side in (-1.0, 1.0):
            bm_add_transformed(bm_chasm, make_box_bm(3.2, 10.5, 12.0), 0, matrix_at_three(rx + ca2 * side * 9.5, max(38.0, ry + 3.0), rz - sa2 * side * 9.5, 0.0, look_yaw, 0.0))

    # Hexagonal basalt clusters rising from bottom pool (y = -17.5)
    for idx in range(28):
        ba = (idx / 28.0) * math.pi * 2.0
        br = 46.0 + (idx % 3) * 6.5
        bx = cx + math.cos(ba) * br
        bz = cz + math.sin(ba) * br
        bh = 9.0 + (idx % 5) * 4.5
        bm_add_transformed(bm_chasm, make_cone_bm(2.8, 2.6, bh, 6), 1, matrix_at_three(bx, -18.0 + bh * 0.5, bz, 0.0, ba, 0.0))

    create_object_from_bmesh("Chasm_Aqueducts_Basalt", bm_chasm, mats["chasm"], collection, smooth=False)


def build_zone4_crystal_canyon(collection, mats):
    """5 Rock Arches, 32 Purple Crystal & Geode Clusters, 24 Giant Bioluminescent Blue Mushrooms & Lantern Flora."""
    bm_arches = bmesh.new()
    bm_crystals = bmesh.new()
    bm_mushrooms = bmesh.new()
    bm_lanterns = bmesh.new()

    # Canyon centerline interpolation points
    canyon_segs = [
        (388.0, 10.0),
        (372.0, 105.0),
        (335.0, 215.0),
        (280.0, 275.0),
        (215.0, 315.0),
    ]

    def sample_canyon(u):
        idx_f = max(0.0, min(len(canyon_segs) - 1.0001, u * (len(canyon_segs) - 1)))
        i0 = int(idx_f)
        frac = idx_f - i0
        ax, az = canyon_segs[i0]
        bx, bz = canyon_segs[i0 + 1]
        px = ax + (bx - ax) * frac
        pz = az + (bz - az) * frac
        tx = bx - ax
        tz = bz - az
        tlen = max(0.001, math.hypot(tx, tz))
        return px, pz, tx / tlen, tz / tlen

    # 1. 5 Natural Canyon Rock Arches spanning high across canyon walls
    for i in range(5):
        u = 0.1 + i * 0.2
        px, pz, tx, tz = sample_canyon(u)
        yaw = math.atan2(tx, tz)
        nx, nz = -tz, tx
        # Spanning rib built from faceted boulders arching from y=22 up to y=64
        steps = 9
        for k in range(steps):
            s = (k / (steps - 1)) * 2.0 - 1.0  # -1..+1
            arch_x = px + nx * s * 38.0
            arch_z = pz + nz * s * 38.0
            arch_y = 62.0 - 34.0 * (s * s)
            bm_add_transformed(bm_arches, make_rock_bm(6.5 - abs(s) * 1.2, i * 31 + k, 1.3, 1.0, 1.1), 0 if k % 2 == 0 else 1, matrix_at_three(arch_x, arch_y, arch_z, 0.0, yaw, s * 0.45))

    create_object_from_bmesh("Canyon_Rock_Arches", bm_arches, mats["canyon_arches"], collection, smooth=False)

    # 2. 32 Glowing Purple Crystal Clusters & Wall Geodes
    for i in range(32):
        u = i / 31.0
        px, pz, tx, tz = sample_canyon(u)
        nx, nz = -tz, tx
        side = 1.0 if i % 2 == 0 else -1.0
        on_wall = (i % 3 == 0)
        offset = (28.0 if on_wall else 15.0 + (i % 3) * 3.5) * side
        cx = px + nx * offset
        cz = pz + nz * offset
        cy = get_terrain_height(cx, cz)

        # Dark geode rock base
        bm_add_transformed(bm_crystals, make_rock_bm(2.2, i * 17, 1.2, 0.8, 1.2), 1, matrix_at_three(cx, cy + 0.3, cz))

        spikes = 6
        for s in range(spikes):
            sh = 4.6 + (s % 3) * 2.6
            sr = 0.72 + (s % 2) * 0.35
            sx_off = (s - 2.5) * 0.65
            sz_off = ((s * 3) % 3 - 1) * 0.65
            tilt_z = -side * (0.45 if on_wall else 0.18) + (s - 2.5) * 0.16
            bm_add_transformed(
                bm_crystals,
                make_cone_bm(sr, 0.04, sh, 6),
                0,
                matrix_at_three(cx + sx_off, cy + sh * 0.42, cz + sz_off, ((s % 2) - 0.5) * 0.35, s * 0.5, tilt_z),
            )

    create_object_from_bmesh("Crystal_Formation", bm_crystals, mats["crystals"], collection, smooth=False)

    # 3. 24 Giant Bioluminescent Blue Mushrooms (sculpted stem, gills, spotted glowing cap)
    for i in range(24):
        u = (i + 0.5) / 24.0
        px, pz, tx, tz = sample_canyon(u)
        nx, nz = -tz, tx
        side = -1.0 if i % 2 == 0 else 1.0
        offset = (14.5 + (i % 3) * 3.8) * side
        mx = px + nx * offset
        mz = pz + nz * offset
        my = get_terrain_height(mx, mz)
        sc = 0.95 + (i % 3) * 0.32

        # Curved stem
        bm_add_transformed(bm_mushrooms, make_cone_bm(1.18 * sc, 0.65 * sc, 5.4 * sc, 8), 0, matrix_at_three(mx, my + 2.5 * sc, mz, 0.06, i * 0.7, side * 0.08))
        # Underside glowing gill disc
        bm_add_transformed(bm_mushrooms, make_cone_bm(2.55 * sc, 0.65 * sc, 0.85 * sc, 10), 2, matrix_at_three(mx, my + 4.7 * sc, mz))
        # Electric blue dome cap
        bm_add_transformed(bm_mushrooms, make_icosphere_bm(2.65 * sc, 2, 1.0, 0.48, 1.0), 1, matrix_at_three(mx, my + 5.2 * sc, mz))
        # Bioluminescent sky-blue spots embedded on cap
        for sp in range(8):
            sa = (sp / 8.0) * math.pi * 2.0
            rr = 0.1 * sc if sp == 0 else 1.6 * sc
            dy = 6.45 * sc if sp == 0 else 5.85 * sc
            bm_add_transformed(bm_mushrooms, make_icosphere_bm(0.32 * sc, 0), 2, matrix_at_three(mx + math.cos(sa) * rr, my + dy, mz + math.sin(sa) * rr))

    create_object_from_bmesh("Giant_Bio_Mushroom", bm_mushrooms, mats["bio_mushrooms"], collection, smooth=False)

    # 4. Hanging Pink Lantern Flora & Canyon Floor Pebbles
    for i in range(28):
        u = i / 27.0
        px, pz, tx, tz = sample_canyon(u)
        nx, nz = -tz, tx
        side = 1.0 if i % 2 == 0 else -1.0
        lx = px + nx * side * 12.2
        lz = pz + nz * side * 12.2
        ly = get_terrain_height(lx, lz)
        bm_add_transformed(bm_lanterns, make_cone_bm(0.18, 0.11, 2.8, 5), 0, matrix_at_three(lx, ly + 1.3, lz, 0.0, 0.0, side * 0.25))
        bm_add_transformed(bm_lanterns, make_icosphere_bm(0.58, 0), 1, matrix_at_three(lx - side * 0.42, ly + 2.6, lz))

    create_object_from_bmesh("Lantern_Flora", bm_lanterns, mats["lanterns"], collection, smooth=False)


def main():
    clear_scene()
    scene = bpy.context.scene
    col = scene.collection

    # Shared PBR Materials
    mats = {
        "terrain": [
            make_pbr_material("Mat_Meadow_Grass", "#68B82E", roughness=0.85),
            make_pbr_material("Mat_Forest_Moss", "#488228", roughness=0.88),
            make_pbr_material("Mat_Tropical_Sand", "#EED6A7", roughness=0.78),
            make_pbr_material("Mat_Cliff_Granite", "#525B6C", roughness=0.82),
            make_pbr_material("Mat_Canyon_Sandstone", "#9E7B66", roughness=0.86),
            make_pbr_material("Mat_Alpine_Snow", "#F1F5F9", roughness=0.55),
        ],
        "sanctuary": [
            make_pbr_material("Mat_Sanc_Stone", "#CBD5E1", roughness=0.78),
            make_pbr_material("Mat_Sanc_Timber", "#6B4226", roughness=0.82),
            make_pbr_material("Mat_Sanc_Banner", "#1E6BFF", roughness=0.65),
            make_pbr_material("Mat_Sanc_Gold", "#FFD700", roughness=0.35, metallic=0.6, emission_hex="#FFD700", emission_strength=0.4),
        ],
        "megalith": [
            make_pbr_material("Mat_Megalith_Stone", "#64748B", roughness=0.88),
            make_pbr_material("Mat_Megalith_Moss", "#4A7C23", roughness=0.9),
        ],
        "trees": [
            make_pbr_material("Mat_Tree_Trunk", "#5C3A21", roughness=0.9),
            make_pbr_material("Mat_Birch_Bark", "#E2E8F0", roughness=0.8),
            make_pbr_material("Mat_Oak_Leaf", "#4E9F3D", roughness=0.75),
            make_pbr_material("Mat_Teal_Leaf", "#3B8EA5", roughness=0.75),
            make_pbr_material("Mat_Pine_Leaf", "#2D6A4F", roughness=0.78),
            make_pbr_material("Mat_Birch_Leaf", "#8AD849", roughness=0.72),
        ],
        "forest_props": [
            make_pbr_material("Mat_Mushroom_Stem", "#F8FAFC", roughness=0.72),
            make_pbr_material("Mat_Amanita_Cap", "#E62E2E", roughness=0.52),
            make_pbr_material("Mat_Amanita_Spot", "#FFFFFF", roughness=0.45),
            make_pbr_material("Mat_Yellow_Cap", "#FACC15", roughness=0.52),
            make_pbr_material("Mat_Moss_Rock", "#526356", roughness=0.88),
            make_pbr_material("Mat_Fallen_Log", "#614126", roughness=0.9),
        ],
        "lighthouse": [
            make_pbr_material("Mat_LH_White", "#F8FAFC", roughness=0.65),
            make_pbr_material("Mat_LH_Red", "#DC2626", roughness=0.6),
            make_pbr_material("Mat_LH_Iron", "#1E293B", roughness=0.45, metallic=0.7),
            make_pbr_material("Mat_LH_Beacon", "#FEF08A", roughness=0.2, emission_hex="#FACC15", emission_strength=2.2),
        ],
        "boats": [
            make_pbr_material("Mat_Boat_Wood", "#7C4D2B", roughness=0.78),
            make_pbr_material("Mat_Sail_White", "#F8FAFC", roughness=0.68),
            make_pbr_material("Mat_Sail_Accent", "#F97316", roughness=0.62),
        ],
        "sea_arch": [
            make_pbr_material("Mat_Sea_Arch_Rock", "#78716C", roughness=0.85),
            make_pbr_material("Mat_Sea_Arch_Sand", "#D6C098", roughness=0.82),
        ],
        "palms": [
            make_pbr_material("Mat_Palm_Trunk", "#8D5B32", roughness=0.85),
            make_pbr_material("Mat_Palm_Leaf", "#4CC43A", roughness=0.7),
            make_pbr_material("Mat_Coconut", "#4A2C14", roughness=0.82),
            make_pbr_material("Mat_Coral_Pink", "#FB7185", roughness=0.65),
            make_pbr_material("Mat_Coral_Teal", "#2DD4BF", roughness=0.65),
        ],
        "cloud": [
            make_pbr_material("Mat_Cloud_White", "#FFFFFF", roughness=0.95),
        ],
        "bridge": [
            make_pbr_material("Mat_Bridge_Stone", "#64748B", roughness=0.85),
            make_pbr_material("Mat_Bridge_Plank", "#854D27", roughness=0.82),
            make_pbr_material("Mat_Bridge_Rope", "#D6B67C", roughness=0.9),
        ],
        "chasm": [
            make_pbr_material("Mat_Aqueduct_Stone", "#5A6275", roughness=0.85),
            make_pbr_material("Mat_Basalt_Pillar", "#334155", roughness=0.8),
        ],
        "canyon_arches": [
            make_pbr_material("Mat_Canyon_Arch_Dark", "#5A6275", roughness=0.86),
            make_pbr_material("Mat_Canyon_Arch_Warm", "#7C6A63", roughness=0.85),
        ],
        "crystals": [
            make_pbr_material("Mat_Crystal_Purple", "#D966FF", roughness=0.22, metallic=0.15, emission_hex="#9900FF", emission_strength=1.5),
            make_pbr_material("Mat_Geode_Rock", "#3B4252", roughness=0.9),
        ],
        "bio_mushrooms": [
            make_pbr_material("Mat_Bio_Stem", "#CBEAFE", roughness=0.65),
            make_pbr_material("Mat_Bio_Cap", "#1A75FF", roughness=0.42, emission_hex="#0B3BA8", emission_strength=0.65),
            make_pbr_material("Mat_Bio_Glow", "#80BFFF", roughness=0.25, emission_hex="#80BFFF", emission_strength=1.8),
        ],
        "lanterns": [
            make_pbr_material("Mat_Lantern_Stem", "#2A4D3B", roughness=0.8),
            make_pbr_material("Mat_Lantern_Bulb", "#FF66CC", roughness=0.3, emission_hex="#FF2299", emission_strength=1.8),
        ],
    }

    build_terrain(col, mats["terrain"])
    build_zone1_sanctuary_and_forest(col, mats)
    build_zone2_tropical_bay(col, mats)
    build_zone3_waterfall_chasm(col, mats)
    build_zone4_crystal_canyon(col, mats)

    # Export directory (supports env var override so Snap Blender writes to non-hidden path cleanly)
    out_dir = os.environ.get("SKYPECK_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    glb_path = os.path.join(out_dir, "skypeck_world.glb")
    blend_path = os.path.join(out_dir, "skypeck_world.blend")

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_yup=True,
        export_apply=True,
        export_materials="EXPORT",
    )

    terrain_obj = bpy.data.objects.get("Terrain_HighRes")
    print(f"EXPORTED_GLB={glb_path} size={os.path.getsize(glb_path)}")
    print(f"EXPORTED_BLEND={blend_path} size={os.path.getsize(blend_path)}")
    print(f"OBJECTS={[o.name for o in bpy.data.objects]}")
    if terrain_obj:
        print(f"TERRAIN_VERTS={len(terrain_obj.data.vertices)} TERRAIN_POLYS={len(terrain_obj.data.polygons)}")


if __name__ == "__main__":
    main()
