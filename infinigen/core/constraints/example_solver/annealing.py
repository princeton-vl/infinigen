# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, Karhan Kayan

import logging
import os
import time
import typing
from pprint import pprint

import bpy
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.constraint_language import util as impl_util
from infinigen.core.constraints.evaluator import eval_memo, evaluate
from infinigen.core.util import blender as butil

from .moves import Move
from .state_def import State

logger = logging.getLogger(__name__)

BPY_GARBAGE_COLLECT_FREQUENCY = 20  # every X optim steps


@gin.configurable
class SimulatedAnnealingSolver:
    def __init__(
        self,
        max_invalid_candidates,
        initial_temp,
        final_temp,
        finetune_pct,
        checkpoint_best=False,
        output_folder=None,
        visualize=False,
        print_report_freq=1,
        print_breakdown_freq=0,
    ) -> None:
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_invalid_candidates = max_invalid_candidates
        self.finetune_pct = finetune_pct

        self.print_report_freq = print_report_freq
        self.print_breakdown_freq = print_breakdown_freq

        self.checkpoint_best = checkpoint_best
        if checkpoint_best:
            raise NotImplementedError(f"{checkpoint_best=}")

        self.output_folder = output_folder
        self.visualize = visualize

        self.cooling_rate = None
        self.last_eval_result = None

        self.eval_memo = {}
        self.stats = []

    def save_stats(self, path):
        if len(self.stats) == 0:
            return

        df = pd.DataFrame.from_records(self.stats)

        logger.info(f"Saving stats {path}")
        df.to_csv(path)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score", color="C0")
        ax1.plot(np.arange(len(df)), df["loss"], color="C0")

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Move Time', color='C1')
        # ax2.plot(df['curr_iteration'], df['move_dur'], color='C1')

        figpath = path.parent / (path.stem + ".png")
        logger.info(f"Saving plot {figpath}")
        plt.savefig(figpath)
        plt.close()

        logger.info(f"Total elapsed {path.stem} {self.stats[-1]['elapsed']:.2f}")

    def reset(self, max_iters):
        self.curr_iteration = 0
        self.curr_result = None
        self.best_loss = None
        self.eval_memo = {}

        self.optim_start_time = time.perf_counter()
        self.max_iterations = max_iters

        if max_iters == 0:
            self.cooling_rate = 0
        else:
            steps = max_iters * (1 - self.finetune_pct)
            ratio = self.final_temp / self.initial_temp
            self.cooling_rate = np.power(ratio, 1 / steps)

        logger.debug(
            f"Reset solver with {max_iters=} cooling_rate={self.cooling_rate:.4f}"
        )

    def checkpoint(self, state):
        filename = os.path.join(self.output_folder, "checkpoint_state.pkl")
        state.save(filename)

        if self.visualize:
            # save score plot
            plt.plot(self.score_history)
            plt.savefig(os.path.join(self.output_folder, "scores.png"))
            plt.close()

            # render image
            i = 1
            while os.path.exists(os.path.join(self.output_folder, f"{i:04}.png")):
                i += 1
            bpy.context.scene.render.filepath = os.path.join(
                self.output_folder, f"{i:04}.png"
            )
            bpy.ops.render.render(write_still=True)

    def validate_lazy_eval(
        self,
        state: State,
        consgraph: cl.Problem,
        prop_result: evaluate.EvalResult,
        filter_domain: r.Domain,
    ):
        test_memo = {}
        impl_util.DISABLE_BVH_CACHE = True
        real_result = evaluate.evaluate_problem(
            consgraph, state, filter_domain, memo=test_memo
        )
        impl_util.DISABLE_BVH_CACHE = False

        if real_result.loss() == prop_result.loss():
            return

        for n in consgraph.traverse(inorder=False):
            key = eval_memo.memo_key(n)
            if key not in self.eval_memo:
                continue
            lazy = self.eval_memo[key]
            if test_memo[key] == lazy:
                continue

            print("\n\n INVALID")
            pprint(n, depth=3)
            print(f"memo for node is out of sync, got {lazy=} yet {test_memo[key]=}")
        raise ValueError(f"{real_result.loss()=:.4f} {prop_result.loss()=:.4f}")

    @gin.configurable
    def _move(
        self,
        consgraph: cl.Node,
        state: State,
        move: Move,
        filter_domain: r.Domain,
        do_lazy_eval=True,
        validate_lazy_eval=False,
    ):
        if do_lazy_eval:
            eval_memo.evict_memo_for_move(consgraph, state, self.eval_memo, move)
            prop_result = evaluate.evaluate_problem(
                consgraph, state, filter_domain, self.eval_memo
            )
        else:
            prop_result = evaluate.evaluate_problem(
                consgraph, state, filter_domain, memo={}
            )

        if validate_lazy_eval:
            self.validate_lazy_eval(state, consgraph, prop_result, filter_domain)

        return prop_result

    @gin.configurable
    def retry_attempt_proposals(
        self,
        propose_func: typing.Callable,
        consgraph: cl.Node,
        state: State,
        temp: float,
        filter_domain: r.Domain,
    ) -> typing.Tuple[Move, evaluate.EvalResult, int]:
        move_gen = propose_func(consgraph, state, filter_domain, temp)

        move = None
        retry = None
        for retry, move in enumerate(move_gen):
            if retry == self.max_invalid_candidates:
                logger.debug(
                    f"{move_gen=} reached {self.max_invalid_candidates=} without succeeding an apply()"
                )
                break

            succeeded = move.apply(state)
            if succeeded:
                eval_memo.evict_memo_for_move(consgraph, state, self.eval_memo, move)
                result = self._move(consgraph, state, move, filter_domain)
                return move, result, retry

            logger.debug(f"{retry=} reverting {move=}")
            eval_memo.evict_memo_for_move(consgraph, state, self.eval_memo, move)
            move.revert(state)

        else:
            logger.debug(f"{move_gen=} produced {retry} attempts and none were valid")

        return move, None, retry

    def curr_temp(self) -> float:
        temp = self.initial_temp * self.cooling_rate**self.curr_iteration
        temp = np.clip(temp, self.final_temp, self.initial_temp)
        return temp

    def metrop_hastings_with_viol(self, prop_result: evaluate.EvalResult, temp: float):
        prop_viol = prop_result.viol_count()
        curr_viol = self.curr_result.viol_count()

        diff = prop_result.loss() - self.curr_result.loss()
        log_prob = -diff / temp

        viol_diff = prop_viol - curr_viol

        result = {"diff": diff, "log_prob": log_prob, "viol_diff": viol_diff}

        if viol_diff < 0:
            result["accept"] = True
            return result
        elif viol_diff > 0:
            result["accept"] = False
            return result

        # standard metropolis-hastings
        rv = np.log(np.random.uniform())
        result["accept"] = rv < log_prob
        return result

    def step(self, consgraph, state, move_gen_func, filter_domain):
        if self.curr_result is None:
            self.curr_result = evaluate.evaluate_problem(
                consgraph, state, filter_domain
            )

        move_start_time = time.perf_counter()

        is_log_step = (
            self.print_report_freq != 0
            and self.curr_iteration % self.print_report_freq == 0
        )
        is_report_step = (
            self.print_breakdown_freq != 0
            and self.curr_iteration % self.print_breakdown_freq == 0
        )

        temp = self.curr_temp()
        move, prop_result, retry = self.retry_attempt_proposals(
            move_gen_func, consgraph, state, temp, filter_domain
        )

        if prop_result is None:
            # set null values for logging purposes
            accept_result = {
                "accept": None,
                "diff": 0,
                "log_prob": 0,
                "viol_diff": None,
            }
        else:
            accept_result = self.metrop_hastings_with_viol(prop_result, temp)
            if accept_result["accept"]:
                self.curr_result = prop_result
                move.accept(state)
            else:
                eval_memo.evict_memo_for_move(consgraph, state, self.eval_memo, move)
                move.revert(state)

        dt = time.perf_counter() - move_start_time
        elapsed = time.perf_counter() - self.optim_start_time

        if (self.print_report_freq != 0 and accept_result["accept"]) or is_log_step:
            n = len(state.objs)
            move_log = move_gen_func.__name__ if move is None else move

            log_prob = accept_result["log_prob"]
            prob = (
                1 if log_prob > 7 else np.exp(accept_result["log_prob"])
            )  # avoid overflow warnings. clamp to exp = exp(7) ~= 1000

            loss = self.curr_result.loss()
            viol = self.curr_result.viol_count()
            diff = accept_result["diff"]
            accept = accept_result["accept"]
            viol_diff = accept_result["viol_diff"] or 0

            logger.info(
                f"it={self.curr_iteration}/{self.max_iterations} {dt=:.3f} {n=} "
                f"{loss=:.3e} {viol=:.1f} "
                f"{temp=:.2e} {diff=:.2f} {viol_diff=:.1f} {prob=:.2f} {accept=} "
                f"{move_log}"
            )

        if is_log_step:
            self.stats.append(
                dict(
                    curr_iteration=self.curr_iteration,
                    loss=self.curr_result.loss(),
                    viol=self.curr_result.viol_count(),
                    best_loss=self.best_loss,
                    temp=temp,
                    accept=accept,
                    move_gen=move_gen_func.__name__,
                    move_type=(move.__class__.__name__ if move is not None else None),
                    move_target=(
                        move.name
                        if move is not None and hasattr(move, "name")
                        else None
                    ),
                    move_dur=dt,
                    elapsed=elapsed,
                    retry=retry,
                )
            )

        if is_report_step and prop_result is not None:
            df = prop_result.to_df()

            if self.last_eval_result is not None:
                last_df = self.last_eval_result.to_df()
                diff_cols = [
                    c
                    for c in df.columns
                    if (
                        not last_df[c].equals(df[c])
                        or (
                            df[c]["viol_count"] is not None
                            and last_df[c]["viol_count"] > 0
                        )
                    )
                ]
                print(
                    self.last_eval_result.viol_count(),
                    self.curr_result.viol_count(),
                    prop_result.viol_count(),
                )
                last_df.index = ["prev_" + x for x in last_df.index]
                df = pd.concat([last_df[diff_cols], df[diff_cols]])

            print(df)

        if self.curr_iteration % BPY_GARBAGE_COLLECT_FREQUENCY == 0:
            butil.garbage_collect(butil.get_all_bpy_data_targets())

        if self.curr_iteration != 0 and self.curr_iteration % 50 == 0:
            print(f"CLUTTER REPORT {self.curr_iteration=}")
            print("  State Size", len(state.objs))
            print("  Trimesh", len(state.trimesh_scene.graph.nodes))
            print("  Objects", len(bpy.data.objects))
            print("  Meshes", len(bpy.data.meshes))
            print("  Materials", len(bpy.data.materials))
            print("  Textures", len(bpy.data.materials))

        self.curr_iteration += 1
        if prop_result is not None:
            self.last_eval_result = prop_result
