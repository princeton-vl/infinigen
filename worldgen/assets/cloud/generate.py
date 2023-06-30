from assets.utils.object import new_cube
from surfaces import surface
from nodes.node_wrangler import Nodes

        self.cloud_types = [Cumulonimbus, ] if self.cloudy else [Cumulus, Stratocumulus, Altocumulus, ]
            Altocumulus: [16, 64], }
        scale_resolution = 4
        self.resolutions = {k: (scale_resolution * u, scale_resolution * v) for k, (u, v) in
            self.resolutions.items()}

        self.min_distance = 256 if self.cloudy else 64
        self.dome_radius = 1024 if self.cloudy else 256
        self.dome_threshold = 32 if self.cloudy else 0
        self.density_range = [1e-5, 1e-4] if self.cloudy else [1e-4, 2e-4]

        self.max_scale = max([t.MAX_EXPECTED_SCALE for t in self.cloud_types])
        self.density = max([t.PLACEHOLDER_DENSITY for t in self.cloud_types])

    def spawn_locations(self):
        obj = new_cube()
        surface.add_geomod(obj, self.geo_dome, apply=True,
                           input_args=[self.dome_radius, self.dome_threshold, self.density_range,
                               self.min_distance])
        butil.delete(obj)
    def create_asset(self, distance, **kwargs):
        resolution = max(1 - distance / self.max_distance, 0)
        new_cloud = cloud_type("Cloud", self.ref_cloud)
        new_cloud = new_cloud.make_cloud(marching_cubes=False, resolution=resolution, )
    @staticmethod
    def geo_dome(nw, dome_radius, dome_threshold, density_range, min_distance, ):
        ico_sphere = nw.new_node('GeometryNodeMeshIcoSphere',
                                 input_kwargs={'Radius': dome_radius, 'Subdivisions': 8, }, )

        transform = nw.new_node(Nodes.Transform,
                                input_kwargs={'Geometry': ico_sphere, 'Scale': (1.2, 1.4, 1.0), }, )
        position = nw.new_node(Nodes.InputPosition)
        separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position, }, )

        less_than = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz.outputs["Z"], 1: dome_threshold, },
                                attrs={'operation': 'LESS_THAN', }, )

        delete_geometry = nw.new_node('GeometryNodeDeleteGeometry',
                                      input_kwargs={'Geometry': transform, 'Selection': less_than, }, )

        distribute_points_on_faces = nw.new_node(Nodes.DistributePointsOnFaces, input_kwargs={
            'Seed': np.random.randint(1e5), }, attrs={'distribute_method': 'POISSON', }, )

        combine_xyz = nw.new_node(Nodes.CombineXYZ,
                                  input_kwargs={'Z': nw.uniform(32, np.random.randint(64, 1e5)), }, )

        set_position = nw.new_node(Nodes.SetPosition, input_kwargs={
            'Offset': combine_xyz, }, )

        verts = nw.new_node(Nodes.PointsToVertices, input_kwargs={'Points': set_position, }, )

        nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': verts, }, )


class CumulonimbusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Cumulonimbus]
        super(CumulonimbusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulonimbus]


class CumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Cumulus]
        super(CumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Cumulus]


class StratocumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Stratocumulus]
        super(StratocumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Stratocumulus]


class AltocumulusFactory(CloudFactory):
    def __init__(self, factory_seed, coarse=False, max_distance=300, steps=128, ):
        self.cloud_types = [Altocumulus]
        super(AltocumulusFactory, self).__init__(factory_seed, coarse, max_distance, steps)
        self.cloud_types = [Altocumulus]
