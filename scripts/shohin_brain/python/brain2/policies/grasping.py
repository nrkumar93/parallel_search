# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np


def get_relative_goal_discrete_sampler(ik_solver, actor_ref, obj_pose,
                                       relative_poses,
                                       config=None,
                                       standoff=0.1,
                                       debug_entity=None,
                                       metric=None):
    """
    Sample a grasp
    """
    
    # Create function with these parameters
    # TODO: support metric
    obj_pose = np.copy(obj_pose)
    def _sample_goal():
        # internal function just for htis
        _idx = np.random.randint(len(relative_poses))
        _T = np.eye(4)
        _T[2,3] = -1*standoff
        goal_pose = obj_pose.dot(relative_poses[_idx]).dot(_T)
        if debug_entity:
            debug_entity.set_pose_matrix(goal_pose)
        _q = ik_solver(actor_ref,
                       goal_pose,
                       q0=config)
        return _q

    return _sample_goal

