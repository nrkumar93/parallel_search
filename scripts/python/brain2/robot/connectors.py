# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np

from brain2.motion_planners.abstract import GoalConnector
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.motion_planners.linear import LinearPlan, JointSpaceLinearPlan
from brain2.motion_planners.rrt_connect import rrt_connect
from brain2.robot.trajectory import retime_trajectory
from brain2.task.action import ConnectionParams
from brain2.utils.extend import simple_extend


class LinearConnector(GoalConnector):
    """ Linear connnection between two states """

    def __init__(self, step_size=0.04):
        self.step_size = step_size

    def __call__(self, world_state, actor, goal, subgoal_params):
        ee_pose = world_state[actor].ee_pose
        #goal_pose = np.copy(ee_pose)
        # Just move up in the world Z axis
        # goal_pose[2,3] += self.plan_length
        goal_pose = subgoal_params.ee_pose
        actor_ref = world_state[actor].ref
        ik_solver = actor_ref.ik_solver
        # Create plan and return it
        plan = LinearPlan(world_state[actor],
                          ee_pose,
                          goal_pose,
                          ik_solver,
                          step_size=self.step_size,
                          suppressed_objs=set([goal]))
        return ConnectionParams(actor=actor, goal=goal, trajectory=plan, success=True)

class RRTConnector(GoalConnector):
    """ RRT in configuration space. Just a dumb connector that goes to the joint-space goal that
    we've already picked out. That's all. """
    def __init__(self, problem_config=None):
        if problem_config is not None:
            self.config = problem_config
        else:
            self.config = {
                'dof': None,
                'p_sample_goal': 0.2,
                'iterations': 100,
                # TODO: multiple goal options propagated through the planner
                'goal_iterations': 1, 
                'verbose': 0,
                'shortcut': True,
                'min_iterations': 10,
                'shortcut_iterations': 50,
                }
        self.problems = {}

    def _make_problem(self, world_state, actor):
        robot = world_state[actor].ref
        self.config['dof'] = robot.dof
        is_valid = lambda q: robot.validate(q, max_pairwise_distance=0.005)
        extend = lambda q1, q2: simple_extend(q1, q2, 0.2)
        is_done = lambda q: False
        problem = MotionPlanningProblem(sample_fn=robot.sample_uniform,
                                        goal_fn=lambda: subgoal_config.q,
                                        is_valid_fn=is_valid,
                                        extend_fn=extend,
                                        is_done_fn=is_done,
                                        config=self.config,
                                        distance_fn=None)
        return problem
 

    def __call__(self, world_state, actor, goal, subgoal_config):
        if actor not in self.problems:
            self.problems[actor] = self._make_problem(world_state, actor)
        problem = self.problems[actor]
        goal_fn = lambda: subgoal_config.q
        problem.reset_sample_goal_fn(goal_fn)
        
        # Actually do the planning part
        actor_state = world_state[actor]
        init = actor_state.q
        path, tree = rrt_connect(init, problem)
        #print(path)
        #input('!!!!!!!')
        if path is not None:
            res = retime_trajectory(actor_state, path)
            return ConnectionParams(actor=actor, goal=goal, trajectory=res, success=True, tree=tree)
        else:
            return ConnectionParams(actor=actor, goal=goal, success=False, tree=tree)
