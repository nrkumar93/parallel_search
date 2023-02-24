# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



from .interface import FlatWorld
from brain2.motion_planners.problem import MotionPlanningProblem
from brain2.utils.extend import simple_extend

def get_planning_problem(env):
    """Create environment and motion planning problem for flat world"""
    flat_world_config = {
            'dof': env.dof,
            'p_sample_goal': 0.2,
            'iterations': 100,
            'verbose': 0,
            'goal_iterations': 100,
            'neighborhood_radius' : 5,
            }
    extend = lambda q1, q2: simple_extend(q1, q2, step_size=env.step_size)
    return MotionPlanningProblem(sample_fn=env.sample,
                                 goal_fn=env.goal,
                                 extend_fn=extend,
                                 is_valid_fn=env.is_valid,
                                 is_done_fn=env.is_done,
                                 config=flat_world_config,
                                 distance_fn=None)
