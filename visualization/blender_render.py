"""Photorealistic wormhole embedding render with Blender (Cycles).

Run headless from the repository root:

    blender --background --python visualization/blender_render.py -- --b0 1.0 --out figures/wormhole_blender.png

Builds the revolved Morris-Thorne embedding surface as a mesh, assigns a glossy
material, lights the scene and renders with Cycles.  Requires Blender's bundled
Python (the ``bpy`` module); it is a no-op import elsewhere.
"""

from __future__ import annotations

import sys
import math


def _embedding_profile(b0, r_max, n):
    """(radius, height) samples for the two-sheet Morris-Thorne surface."""
    pts = []
    z = 0.0
    prev = None
    for i in range(n):
        r = b0 * (1 + 1e-3) + (r_max - b0) * i / (n - 1)
        val = r * r / (b0 * b0) - 1.0
        dz = 1.0 / math.sqrt(val) if val > 1e-6 else 0.0
        if prev:
            z += 0.5 * (dz + prev[1]) * (r - prev[0])
        prev = (r, dz)
        pts.append((r, z))
    lower = [(r, -zz) for (r, zz) in reversed(pts)]
    return lower + pts


def build_and_render(b0=1.0, out="figures/wormhole_blender.png", n_r=80, n_phi=160):
    import bpy  # only available inside Blender

    # clean scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    prof = _embedding_profile(b0, 5 * b0, n_r)
    verts, faces = [], []
    rows = len(prof)
    for (R, Z) in prof:
        for j in range(n_phi):
            ph = 2 * math.pi * j / (n_phi - 1)
            verts.append((R * math.cos(ph), R * math.sin(ph), Z))
    for i in range(rows - 1):
        for j in range(n_phi - 1):
            a = i * n_phi + j
            faces.append((a, a + 1, a + n_phi + 1, a + n_phi))

    mesh = bpy.data.meshes.new("wormhole")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("wormhole", mesh)
    bpy.context.collection.objects.link(obj)

    # smooth + glossy blue material
    for p in mesh.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("whmat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.10, 0.45, 1.0, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.4
    bsdf.inputs["Roughness"].default_value = 0.25
    obj.data.materials.append(mat)

    # camera + light
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (9, -9, 6)
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("key", type="AREA")
    light_data.energy = 2000
    light = bpy.data.objects.new("key", light_data)
    light.location = (8, -4, 10)
    bpy.context.collection.objects.link(light)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    bpy.ops.render.render(write_still=True)
    return out


def _parse_args(argv):
    b0, out = 1.0, "figures/wormhole_blender.png"
    if "--" in argv:
        rest = argv[argv.index("--") + 1:]
        for i, a in enumerate(rest):
            if a == "--b0":
                b0 = float(rest[i + 1])
            elif a == "--out":
                out = rest[i + 1]
    return b0, out


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("This script must be run inside Blender:\n"
              "  blender --background --python visualization/blender_render.py -- --b0 1.0")
        sys.exit(0)
    b0, out = _parse_args(sys.argv)
    print("rendered:", build_and_render(b0=b0, out=out))
