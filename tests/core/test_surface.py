from infinigen.assets.materials.dev import BasicBSDF
from infinigen.assets.utils.decorate import read_center
from infinigen.core import surface
from infinigen.core.util import blender as butil


def test_assign_material_basic():
    cube = butil.spawn_cube()

    surface.assign_material(cube, BasicBSDF()())
    assert len(cube.material_slots) == 1


def test_assign_material_strmask():
    cube = butil.spawn_cube()

    face_centers = read_center(cube)
    mask = face_centers[:, 0] > 0.1
    surface.write_attr_data(cube, "right", mask, domain="FACE")

    surface.assign_material(cube, BasicBSDF()())
    surface.assign_material(cube, BasicBSDF()(), selection="right")

    material_index = surface.read_attr_data(cube, "material_index", domain="FACE")
    assert (material_index == 0).sum() == 5
    assert (material_index == 1).sum() == 1


def test_assign_material_negstr():
    cube = butil.spawn_cube()

    face_centers = read_center(cube)
    mask = face_centers[:, 0] > 0.1
    surface.write_attr_data(cube, "right", mask, domain="FACE")

    surface.assign_material(cube, BasicBSDF()())
    surface.assign_material(cube, BasicBSDF()(), selection="!right")

    material_index = surface.read_attr_data(cube, "material_index", domain="FACE")
    assert (material_index == 0).sum() == 1
    assert (material_index == 1).sum() == 5


def test_assign_material_nparray():
    cube = butil.spawn_cube()

    face_centers = read_center(cube)
    mask = face_centers[:, 0] > 0.1
    surface.write_attr_data(cube, "right", mask, domain="FACE")

    surface.assign_material(cube, BasicBSDF()())
    surface.assign_material(cube, BasicBSDF()(), selection=mask)

    material_index = surface.read_attr_data(cube, "material_index", domain="FACE")
    assert (material_index == 0).sum() == 5
    assert (material_index == 1).sum() == 1


def test_assign_material_multi():
    cube = butil.spawn_cube()

    face_centers = read_center(cube)
    mask = face_centers[:, 0] > 0.1
    surface.write_attr_data(cube, "right", mask, domain="FACE")

    surface.assign_material(cube, BasicBSDF()())
    surface.assign_material(cube, BasicBSDF()(), selection="right")
