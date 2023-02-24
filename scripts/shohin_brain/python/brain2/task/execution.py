# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import brain2.utils.status as status
import brain2.utils.info as info

from brain2.task.action import Policy


class PlanExecutionPolicy(Policy):
    """
    Generic wrapper for ROS execution of a plan.
    """

    MODE_IDLE = 0
    MODE_GOAL = 1
    MODE_PLAN = 2
    MODE_POLICY = 3

    def __init__(self, domain, plan, goal, update_logical_state=True,
                 init_world_state=None):
        """
        Create plan execution policy. Will execute a specific sequence of
        operators until goal is met.
        """
        self.domain = domain
        self.plan = plan
        self.goal = goal
        self.reset()
        self.update_logical_state = update_logical_state
        if init_world_state is not None:
            self.enter(world_state)

    def reset(self):
        self.mode = self.MODE_IDLE
        self.op = None
        self.op_args = None
        self.current_plan = None
        self.current_goal = None
        self.implicit_conditions = None

    def enter(self, world_state=None, actor=None, goal=None):
        self.reset()

        if self.current_goal is not None or self.current_plan is not None:
            raise RuntimeError('did not properly exit plan')

        # compute goals based on templates
        # TODO: improve formatting so we can use more fields than just the
        # "goal" argument
        self.current_goal = [(g.format(goal), v) for g, v in self.goal]

        # Compute a plan
        if self.plan is None:
            raise RuntimeError('planning not yet supported')
        else:
            # TODO: improve formatting
            self.current_plan = [p.format(goal) for p in self.plan]

        self.implicit_conditions = self.domain.get_implicit_conditions(
            self.current_plan, self.current_goal)

        # Return true on successful creation of plans
        return True

    def exit(self, world_state=None, actor=None, goal=None):
        """
        This is called at the end to make sure we cleaned up the operators and
        policies properly.
        """
        if self.op is not None:
            self.op.exit(world_state, *self.op_args)
        self.reset()
        # return true on successful exit
        return True

    def is_subplan(self):
        """
        Tells execution this is a sublan so we should descend into it even if we
        are not actually moving the robot, and step this policy anyway.
        """
        return True

    def step(self, world_state, actor=None, goal=None,
             evaluate=False):
        """
        Apply policy or generate plan. Three modes:
        - plan and find a goal
        - follow a given plan
        - just execute a policy to test it

        This loop runs continuously at 10hz, 30hz, etc.
        It computes the current continuous world state:
            world_state = self.observer.observe()
        Then it updates the associated logical state:
            domain.update_logical(world_state)

        This is used in conjunction with a plan (a list of operators) to
        determine what policy to execute.
        """

        if self.current_plan is not None:
            if self.current_goal is None:
                raise RuntimeError('goal missing!')

            # Check goal status: run through all elements in plan, find the last
            # element in the plan whose conditions are met.
            # We're assuming that the plan has been properly parameterized by
            # the enter() function.

            res, self.op, self.op_args = self.domain.step(
                self.current_plan, self.current_goal, world_state,
                self.op, self.op_args, self.implicit_conditions,
                update_logical=self.update_logical_state,
                evaluate=evaluate)

            # Check return status
            if res == status.FINISHED:
                self.op = None
                self.op_args = None
                info.logwarn("Plan execution complete.")
                return status.FINISHED
            elif self.op is None or res < 0:
                info.logwarn("Could not continue execution: " + str(res))
                return status.FAILED
            else:
                return status.RUNNING
        else:
            # No plan or goals received
            info.logwarn("Executing without a goal/plan!")
            return status.IDLE

    def get_current_op(self):
        return self.op, self.op_args
