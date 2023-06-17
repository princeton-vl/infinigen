
## Part Definition

a Part is comprised of:
- a blender mesh object
- a line curve that runs centrally through it (specified as an np.array of points in order)
- metadata for rigging / animating that part 

a GeonodesPartFactory produces a Part using a geometry nodes group of a particular format. For the nodegroup's input:
- The geonodes will be applied to a single vert as its input, unless you override the base_obj method
- Any nodegroup inputs / 'genes' should be specified by overriding the GeonodesParts.params() method with rng calls
As output, it should have output sockets for:
- output a 'Geometry' field, which will be included in the final creature
- output a 'Skeleton Curve' field, which will be used to populate the skeleton curve
- optionally: output any number of other extra geometry bits
    - these will be parented to the part but not remeshed etc
    - use this for eyes / teeth you want to remain separate
- optionally: output some attributes over its 'Geometry' field
    - use this to specifiy skin rigidity, or hair related parameters

## Part Conventions

+X is the long axis, +Z is the up axis, +Y is the left side of a creature, or the leading edge of a leg or wing
parity 1 is right side, -1 is left side

Length/Yaw/Rad is a 3D polar coordinate for addressing the surface of a Part
- Length is a 0-1 scalar specifying the percentage of the length along the skeleton to travel
- Yaw is a [-1,1] scalar specifying the yaw tangentially to the skeleton curve to shoot a ray. 0=downwards, +-0.5=left/right, +-1=upwards
- Rad is a 0-1 (or more than 1 in some cases) percentage scalar. The code will raycast using Length/Yaw to the surface of the part, then move 'Rad' percentage of the way along that ray towards the surface

Parts are produced by a PartFactory, which contains a dict of params. This dict may be interpolated to produce intermediate parts. We will even attempt to interpolate the matching keys of two dicts from different PartFactory types. To maximize the quality of these interpolations, the param dict should contain these keys, whenever possible:
- 'length_rad1_rad2': The overall length , and starting / ending approx radius of the part
- 'aspect' - what percent of the height is the width
- 'angles_deg' - R^3 