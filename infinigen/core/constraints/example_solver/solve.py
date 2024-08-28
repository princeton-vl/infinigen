# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import copy
import logging
from pathlib import Path

import bpy
import gin
import numpy as np
from tqdm import trange

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.evaluator import domain_contains
from infinigen.core.constraints.example_solver import (
    greedy,
    propose_continous,
    propose_discrete,
)
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.util import blender as butil

from .annealing import SimulatedAnnealingSolver
from .room.floor_plan import FloorPlanSolver

logger = logging.getLogger(__name__)


def map_range(x, xmin, xmax, ymin, ymax, exp=1):
    if x < xmin:
        return ymin
    if x > xmax:
        return ymax

    t = (x - xmin) / (xmax - xmin)
    return ymin + (ymax - ymin) * t**exp


@gin.register
class LinearDecaySchedule:
    def __init__(self, start, end, pct_duration):
        self.start = start
        self.end = end
        self.pct_duration = pct_duration

    def __call__(self, t):
        return map_range(t, 0, self.pct_duration, self.start, self.end)


@gin.configurable
class Solver:
    def __init__(
        self,
        output_folder: Path,
    ):
        """Initialize the solver

        Parameters
        ----------
        output_folder : Path
            The folder to save output plots to
        print_report_freq : int
            How often to print loss reports
        multistory : bool
            Whether to use the multistory room solver
        constraints_greedy_unsatisfied : str | None
            What do we do if relevant constraints are unsatisfied at the end of a greedy stage?
            Options are 'warn` or `abort` or None

        """

        self.output_folder = output_folder

        self.optim = SimulatedAnnealingSolver(
            output_folder=output_folder,
        )
        self.room_solver_fn = FloorPlanSolver
        self.state: State = None
        self.dimensions = None

    def _configure_move_weights(self, restrict_moves, addition_weight_scalar=1.0):
        schedules = {
            "addition": (
                propose_discrete.propose_addition,
                LinearDecaySchedule(
                    6 * addition_weight_scalar, 0.1 * addition_weight_scalar, 0.9
                ),
            ),
            "deletion": (
                propose_discrete.propose_deletion,
                LinearDecaySchedule(2, 0.0, 0.5),
            ),
            "plane_change": (
                propose_discrete.propose_relation_plane_change,
                LinearDecaySchedule(2, 0.1, 1),
            ),
            "resample_asset": (
                propose_discrete.propose_resample,
                LinearDecaySchedule(1, 0.1, 0.7),
            ),
            "reinit_pose": (
                propose_continous.propose_reinit_pose,
                LinearDecaySchedule(1, 0.5, 1),
            ),
            "translate": (propose_continous.propose_translate, 1),
            "rotate": (propose_continous.propose_rotate, 0.5),
        }

        if restrict_moves is not None:
            schedules = {k: v for k, v in schedules.items() if k in restrict_moves}
            logger.info(
                f"Restricting {self.__class__.__name__} moves to {list(schedules.keys())}"
            )

        return schedules

    @gin.configurable
    def choose_move_type(
        self,
        moves: dict[str, tuple],
        it: int,
        max_it: int,
    ):
        t = it / max_it
        names, confs = zip(*moves.items())
        funcs, scheds = zip(*confs)
        weights = np.array([s if isinstance(s, (float, int)) else s(t) for s in scheds])
        return np.random.choice(funcs, p=weights / weights.sum())

    def solve_rooms(self, scene_seed, consgraph: cl.Problem, filter: r.Domain):
        self.state, _, _ = self.room_solver_fn(scene_seed, consgraph).solve()
        return self.state

    @gin.configurable
    def solve_objects(
        self,
        consgraph: cl.Problem,
        filter_domain: r.Domain,
        var_assignments: dict[str, str],
        n_steps: int,
        desc: str,
        abort_unsatisfied: bool = False,
        print_bounds: bool = False,
        restrict_moves: list[str] = None,
        addition_weight_scalar: float = 1.0,
    ):
        moves = self._configure_move_weights(
            restrict_moves, addition_weight_scalar=addition_weight_scalar
        )

        filter_domain = copy.deepcopy(filter_domain)

        desc_full = (desc, *var_assignments.values())

        dom_assignments = {
            k: r.Domain(self.state.objs[objkey].tags)
            for k, objkey in var_assignments.items()
        }
        filter_domain = r.substitute_all(filter_domain, dom_assignments)

        if not r.domain_finalized(filter_domain):
            raise ValueError(
                f"Cannot solve {desc_full=} with non-finalized domain {filter_domain}"
            )

        orig_bounds = r.constraint_bounds(consgraph)
        bounds = propose_discrete.preproc_bounds(
            orig_bounds, self.state, filter_domain, print_bounds=print_bounds
        )

        if len(bounds) == 0:
            logger.info(f"No objects to be added for {desc_full=}, skipping")
            return self.state

        active_count = greedy.update_active_flags(self.state, var_assignments)

        n_start = len(self.state.objs)
        logger.info(
            f"Greedily solve {desc_full} - stage has {len(bounds)}/{len(orig_bounds)} bounds, "
            f"{active_count=}/{len(self.state.objs)} objs"
        )

        self.optim.reset(max_iters=n_steps)
        ra = trange(n_steps) if self.optim.print_report_freq == 0 else range(n_steps)
        for j in ra:
            move_gen = self.choose_move_type(moves, j, n_steps)
            self.optim.step(consgraph, self.state, move_gen, filter_domain)

        logger.info(
            f"Finished solving {desc_full}, added {len(self.state.objs) - n_start} "
            f"objects, loss={self.optim.curr_result.loss():.4f} viol={self.optim.curr_result.viol_count()}"
        )

        logger.info(self.optim.curr_result.to_df())

        violations = {
            k: v for k, v in self.optim.curr_result.violations.items() if v > 0
        }

        if len(violations):
            msg = f"Solver has failed to satisfy constraints for stage {desc_full}. {violations=}."
            if abort_unsatisfied:
                butil.save_blend(self.output_folder / f"abort_{desc}.blend")
                raise ValueError(msg)
            else:
                msg += " Continuing anyway, override `solve_objects.abort_unsatisfied=True` via gin to crash instead."
                logger.warning(msg)

        # re-enable everything so the blender scene populates / displays correctly etc
        for k, v in self.state.objs.items():
            greedy.set_active(self.state, k, True)

        return self.state

    def get_bpy_objects(self, domain: r.Domain) -> list[bpy.types.Object]:
        objkeys = domain_contains.objkeys_in_dom(domain, self.state)
        return [self.state.objs[k].obj for k in objkeys]
