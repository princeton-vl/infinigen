import logging

from shapely.geometry import Point, Polygon, MultiPolygon

from infinigen.core.util import blender as butil
import infinigen.core.constraints.constraint_language as cl
from infinigen.core.constraints.example_solver.state_def import State, ObjectState, RelationState
from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import constrain_contact, any_touching
from infinigen.core.constraints.example_solver.geometry.stability import stable_against, supported_by
from infinigen.core import tags as t

import gin

logger = logging.getLogger(__name__)

def all_relations_valid(state, name):

    rels = state.objs[name].relations
    for i, relation_state in enumerate(rels):
        match relation_state.relation:
            case cl.StableAgainst(child_tags, parent_tags, margin):
                res = stable_against(state, name, relation_state)
                if not res:
                    logger.debug(f'{name} failed relation {i=}/{len(rels)} {relation_state.relation} on {relation_state.target_name}')
                    return False
            case unmatched:
                raise TypeError(f"Unhandled {relation_state.relation}")
            
    return True

@gin.configurable
def check_post_move_validity(
    state: State, 
    name: str,
    disable_collision_checking=False,
    visualize=False
): 

    objstate = state.objs[name]

    collision_objs = [
        os.obj.name for k, os in state.objs.items() 
        if k != name and t.Semantics.NoCollision not in os.tags
    ]

    if len(collision_objs) == 0: 
    
    if not all_relations_valid(state, name):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f'validity_relations_fail_{name}'

        return False
    
    if disable_collision_checking:
        return True
    if t.Semantics.NoCollision in objstate.tags:
        return True

    touch = any_touching(scene, objstate.obj.name, collision_objs, bvh_cache=state.bvh_cache)
    if not constrain_contact(touch, should_touch=None, max_depth=0.0001):
        if visualize:
            vis_obj = butil.copy(objstate.obj)
            vis_obj.name = f'validity_contact_fail_{name}'

        contact_names = [[x for x in t.names if not x.startswith('_')] for t in touch.contacts]
        logger.debug(f'validity failed - {name} touched {contact_names[0]} {len(contact_names)=}')
