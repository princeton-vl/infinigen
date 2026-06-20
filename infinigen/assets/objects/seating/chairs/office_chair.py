# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import bpy
import gin
import numpy as np
from numpy.random import choice, uniform
from pydantic import ConfigDict, Field

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.seating.chairs.seats.curvy_seats import (
    generate_curvy_seats,
)
from infinigen.assets.objects.tables.cocktail_table import geometry_create_legs
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.random import weighted_sample


def geometry_assemble_chair(nw: NodeWrangler, **kwargs):
    generateseat = nw.new_node(
        generate_curvy_seats().name,
        input_kwargs={
            "Width": kwargs["Top Profile Width"],
            "Front Relative Width": kwargs["Top Front Relative Width"],
            "Front Bent": kwargs["Top Front Bent"],
            "Seat Bent": kwargs["Top Seat Bent"],
            "Mid Bent": kwargs["Top Mid Bent"],
            "Mid Relative Width": kwargs["Top Mid Relative Width"],
            "Back Bent": kwargs["Top Back Bent"],
            "Back Relative Width": kwargs["Top Back Relative Width"],
            "Mid Pos": kwargs["Top Mid Pos"],
            "Seat Height": kwargs["Top Thickness"],
        },
    )

    seat_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generateseat,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    seat_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": seat_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [seat_instance, legs]}
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class OfficeChairParameters(AssetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    top_profile_width: Annotated[
        float, Field(alias="Top Profile Width", json_schema_extra={"editable": True})
    ]
    top_thickness: Annotated[
        float, Field(alias="Top Thickness", json_schema_extra={"editable": True})
    ]
    top_front_relative_width: Annotated[
        float,
        Field(alias="Top Front Relative Width", json_schema_extra={"editable": True}),
    ]
    top_front_bent: Annotated[
        float, Field(alias="Top Front Bent", json_schema_extra={"editable": True})
    ]
    top_seat_bent: Annotated[
        float, Field(alias="Top Seat Bent", json_schema_extra={"editable": True})
    ]
    top_mid_bent: Annotated[
        float, Field(alias="Top Mid Bent", json_schema_extra={"editable": True})
    ]
    top_mid_relative_width: Annotated[
        float,
        Field(alias="Top Mid Relative Width", json_schema_extra={"editable": True}),
    ]
    top_back_bent: Annotated[
        float, Field(alias="Top Back Bent", json_schema_extra={"editable": True})
    ]
    top_back_relative_width: Annotated[
        float,
        Field(alias="Top Back Relative Width", json_schema_extra={"editable": True}),
    ]
    top_mid_pos: Annotated[
        float, Field(alias="Top Mid Pos", json_schema_extra={"editable": True})
    ]
    height: Annotated[float, Field(alias="Height", json_schema_extra={"editable": True})]
    top_height: float = Field(alias="Top Height", json_schema_extra={"editable": False})
    leg_style: Literal["straight", "single_stand", "wheeled"] = Field(
        alias="Leg Style", json_schema_extra={"editable": False}
    )
    leg_ngon: int = Field(alias="Leg NGon", json_schema_extra={"editable": False})
    leg_placement_top_relative_scale: float = Field(
        alias="Leg Placement Top Relative Scale",
        json_schema_extra={"editable": False},
    )
    leg_placement_bottom_relative_scale: Annotated[
        float,
        Field(
            alias="Leg Placement Bottom Relative Scale",
            json_schema_extra={"editable": True},
        ),
    ]
    leg_height: float = Field(alias="Leg Height", json_schema_extra={"editable": False})
    leg_number: int | None = Field(
        default=None, alias="Leg Number", json_schema_extra={"editable": False}
    )
    leg_diameter: float | None = Field(
        default=None, alias="Leg Diameter", json_schema_extra={"editable": False}
    )
    leg_curve_control_points: list[tuple[float, float]] | None = Field(
        default=None,
        alias="Leg Curve Control Points",
        json_schema_extra={"editable": False},
    )
    strecher_relative_pos: float | None = Field(
        default=None,
        alias="Strecher Relative Pos",
        json_schema_extra={"editable": False},
    )
    strecher_increament: int | None = Field(
        default=None, alias="Strecher Increament", json_schema_extra={"editable": False}
    )
    leg_pole_number: int | None = Field(
        default=None, alias="Leg Pole Number", json_schema_extra={"editable": False}
    )
    leg_joint_height: float | None = Field(
        default=None, alias="Leg Joint Height", json_schema_extra={"editable": False}
    )
    leg_wheel_arc_sweep_angle: float | None = Field(
        default=None,
        alias="Leg Wheel Arc Sweep Angle",
        json_schema_extra={"editable": False},
    )
    leg_wheel_width: float | None = Field(
        default=None, alias="Leg Wheel Width", json_schema_extra={"editable": False}
    )
    leg_wheel_rot: float | None = Field(
        default=None, alias="Leg Wheel Rot", json_schema_extra={"editable": False}
    )
    leg_pole_length: float | None = Field(
        default=None, alias="Leg Pole Length", json_schema_extra={"editable": False}
    )
    top_material: Any = Field(alias="TopMaterial", json_schema_extra={"editable": False})
    leg_material: Any = Field(alias="LegMaterial", json_schema_extra={"editable": False})
    scratch: Any | None = Field(default=None, json_schema_extra={"editable": False})
    edge_wear: Any | None = Field(default=None, json_schema_extra={"editable": False})


@gin.configurable
class OfficeChairFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = OfficeChairParameters

    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super().__init__(factory_seed, coarse=coarse)
        self.dimensions = dimensions
        self.init_legacy_parameters()

    @staticmethod
    def _sample_materials() -> tuple[dict[str, Any], Any | None, Any | None]:
        params = {
            "TopMaterial": weighted_sample(material_assignments.large_seat_fabric)(),
            "LegMaterial": weighted_sample(material_assignments.furniture_leg)(),
        }
        wrapped_params = {k: v() for k, v in params.items()}
        scratch_prob, edge_wear_prob = material_assignments.wear_tear_prob
        scratch, edge_wear = material_assignments.wear_tear
        scratch = None if uniform() > scratch_prob else scratch()
        edge_wear = None if uniform() > edge_wear_prob else edge_wear()
        return wrapped_params, scratch, edge_wear

    def get_material_params(self, leg_style):
        return self._sample_materials()

    @staticmethod
    def sample_parameters(dimensions):
        geometry, _ = OfficeChairFactory._sample_geometry_parameters(dimensions)
        return geometry

    @staticmethod
    def _sample_geometry_parameters(dimensions):
        if dimensions is None:
            x = uniform(0.5, 0.6)
            z = uniform(1.0, 1.4)
            dimensions = (x, x, z)

        x, _, z = dimensions
        top_thickness = uniform(0.5, 0.7)
        leg_style = choice(["single_stand", "wheeled"])

        parameters = {
            "Top Profile Width": x,
            "Top Thickness": top_thickness,
            "Top Front Relative Width": uniform(0.5, 0.8),
            "Top Front Bent": uniform(-1.5, -0.4),
            "Top Seat Bent": uniform(-1.5, -0.4),
            "Top Mid Bent": uniform(-2.4, -0.5),
            "Top Mid Relative Width": uniform(0.5, 0.9),
            "Top Back Bent": uniform(-1, -0.1),
            "Top Back Relative Width": uniform(0.6, 0.9),
            "Top Mid Pos": uniform(0.4, 0.6),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": uniform(1.1, 1.3),
            "Leg Height": 1.0,
        }

        if leg_style == "single_stand":
            parameters.update(
                {
                    "Leg Number": 1,
                    "Leg Diameter": uniform(0.7 * x, 0.9 * x),
                    "Leg Curve Control Points": [
                        (0.0, uniform(0.1, 0.2)),
                        (0.5, uniform(0.1, 0.2)),
                        (0.9, uniform(0.2, 0.3)),
                        (1.0, 1.0),
                    ],
                }
            )
        elif leg_style == "straight":
            parameters.update(
                {
                    "Leg Number": 4,
                    "Leg Diameter": uniform(0.04, 0.06),
                    "Leg Curve Control Points": [
                        (0.0, 1.0),
                        (0.4, uniform(0.85, 0.95)),
                        (1.0, uniform(0.4, 0.6)),
                    ],
                    "Strecher Relative Pos": uniform(0.2, 0.6),
                    "Strecher Increament": choice([0, 1, 2]),
                }
            )
        elif leg_style == "wheeled":
            parameters.update(
                {
                    "Leg Number": 1,
                    "Leg Pole Number": choice([4, 5]),
                    "Leg Diameter": uniform(0.03, 0.05),
                    "Leg Joint Height": uniform(0.5, 0.8) * (z - top_thickness),
                    "Leg Wheel Arc Sweep Angle": uniform(120, 240),
                    "Leg Wheel Width": uniform(0.11, 0.15),
                    "Leg Wheel Rot": uniform(0, 360),
                    "Leg Pole Length": uniform(1.6, 2.0),
                }
            )
        else:
            raise NotImplementedError

        return parameters, leg_style

    def _sample_init_parameters(self, seed: int) -> OfficeChairParameters:
        geometry, _ = self._sample_geometry_parameters(self.dimensions)
        materials, scratch, edge_wear = self._sample_materials()
        return OfficeChairParameters.model_validate(
            {
                "seed": seed,
                "scratch": scratch,
                "edge_wear": edge_wear,
                **geometry,
                **materials,
            }
        )

    def apply_parameters(
        self, params: OfficeChairParameters, *, spawn_scope: bool = True
    ) -> None:
        self.params = params.model_dump(
            exclude={"seed", "scratch", "edge_wear"},
            by_alias=True,
            exclude_none=True,
        )
        self.scratch = params.scratch
        self.edge_wear = params.edge_wear
        self._use_fixed_spawn_draws = spawn_scope

    def create_asset(self, **params):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        surface.add_geomod(
            obj, geometry_assemble_chair, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)

        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
