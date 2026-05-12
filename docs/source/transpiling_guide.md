# Transpiling V1 Geonodes to V2

## 1. Find Transpilable Nodegroups

V1 nodegroups decorated with `@node_utils.to_nodegroup` can be transpiled. Factory classes cannot - rewrite their assembly logic manually.

```bash
# Find decorated nodegroups in a v1 file
grep -n "to_nodegroup" src/infinigen/assets/objects/lamp/lamp.py
```

## 2. Run Transpiler

```bash
uv run python scripts/transpile_v1_to_v2.py \
    infinigen.assets.objects.lamp.lamp.nodegroup_bulb \
    infinigen.assets.objects.lamp.lamp.nodegroup_lamp_head \
    infinigen.assets.objects.lamp.lamp.nodegroup_lamp_geometry \
    infinigen_v2.generators.objects.lamp \
    --format
```

Multiple nodegroups can be transpiled at once. Output goes to the module path specified last.

## 3. Fix Common Issues

**Function order**: `@pf.nodes.node_function` executes at import. Define dependencies first (bottom-up order).

**sample_curve**: Remove `use_all_curves=True` if causing errors - it conflicts with curve_index input.

**Mesh results**: Some nodes return named tuples. Use `.mesh` attribute:
```python
sphere = pf.nodes.geo.mesh_uv_sphere(radius=0.1)
geo = pf.nodes.geo.set_material(sphere.mesh, mat)  # not sphere
```

## 4. Write Distribution Function

```python
def lamp_distribution(
    rng: pf.RNG,
    height: float | None = None,  # optional overrides
) -> pf.MeshObject:
    # Sample params (match v1 ranges)
    stand_radius = pf.random.uniform(rng, 0.005, 0.015)
    if height is None:
        height = pf.random.uniform(rng, 1.0, 1.5)

    # Create materials
    vec = pf.nodes.shader.geometry().position
    metal_mat = pf.Material(surface=brushed_metal_distribution(rng, vec))

    # Call transpiled geometry
    result = lamp_geometry(
        stand_radius=stand_radius,
        metal_material=metal_mat,
        ...
    )

    # Convert to object
    obj = pf.nodes.to_mesh_object(result.geometry)
    return obj
```

## 5. Variant Distributions

Create variants by calling base with overrides:

```python
def desk_lamp_distribution(rng: pf.RNG) -> pf.MeshObject:
    height = pf.random.uniform(rng, 0.25, 0.4)
    return lamp_distribution(rng, height=height)
```

## 6. Add to Manifest

```json
{
  "category": "Object",
  "name": "infinigen_v2.generators.objects.lamp.lamp_distribution"
}
```

## 7. Test Render

```bash
uv run infinigen_v2 lamp_distribution object_demo render_cycles \
    --seed 1 --output outputs/lamp/1 --quiet
```

## Material Patterns

**Wrap shader in material**:
```python
mat = pf.Material(surface=shader_distribution(rng, vec))
```

**Material choice**:
```python
def stem_material_distribution(rng, vec):
    func = pf.control.choice(rng, [
        (wood_grain_distribution, 1.0),
        (brushed_metal_distribution, 1.0),
    ])
    return func(rng, vec)
```

**Shared materials**: Put reusable shaders in `shaders/materials/` (e.g., `plastic_grayscale_distribution`, `lamp_bulb_nonemissive`).

## Placement

```python
from mathutils import Vector

# On top of table (centered)
place_bbox_touching(lamp, table, relation=Vector((0.5, 0.5, 1)), gap=0)

# Beside sofa (random side)
side = pf.control.choice(rng, [(-1, 1.0), (1, 1.0)])
place_bbox_touching(table, sofa, relation=Vector((0.5, side, 0)))
```

### Todo

**Shelves & Storage**
- [ ] `shelves/cell_shelf.py` (10 nodegroups)
- [ ] `shelves/triangle_shelf.py` (9 nodegroups)
- [ ] `shelves/cabinet.py` (7 nodegroups)
- [ ] `shelves/doors.py` (7 nodegroups)
- [ ] `shelves/simple_bookcase.py` (6 nodegroups)
- [ ] `shelves/large_shelf.py` (6 nodegroups)
- [ ] `shelves/drawers.py` (4 nodegroups)
- [ ] `shelves/simple_desk.py` (2 nodegroups)

**Appliances**
- [ ] `appliances/oven.py` (9 nodegroups)
- [ ] `appliances/dishwasher.py` (7 nodegroups)
- [ ] `appliances/beverage_fridge.py` (7 nodegroups)
- [ ] `appliances/microwave.py` (5 nodegroups)
- [ ] `appliances/toaster.py` (4 nodegroups)

**Tables**
- [x] `tables/table_utils.py` (10 nodegroups) - partial, simple_table in v2
- [ ] `tables/lofting.py` (6 nodegroups)
- [ ] `tables/table_top.py` (2 nodegroups)
- [ ] `tables/legs/single_stand.py`
- [ ] `tables/legs/wheeled.py`
- [ ] `tables/legs/square.py`
- [ ] `tables/legs/straight.py`

**Lighting**
- [x] `lamp/lamp.py` (5 nodegroups)
- [x] `lamp/ceiling_lights.py`
- [ ] `lamp/ceiling_classic_lamp.py`

**Other Indoor**
- [x] `windows/window.py` (5 nodegroups)
- [ ] `table_decorations/sink.py` (3 nodegroups)
- [x] `seating/sofa.py` (3 nodegroups)
- [ ] `organizer/plate_rack.py` (3 nodegroups)
- [ ] `organizer/basket.py` (2 nodegroups)
- [ ] `wall_decorations/range_hood.py`
- [ ] `wall_decorations/skirting_board.py`
- [ ] `elements/doors/bar_handle.py`
