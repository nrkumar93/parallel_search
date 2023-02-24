# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import brain2.utils.axis as axis
import brain2.utils.transformations as tra

from brain2.utils.info import logwarn, logerr


class ApproachRegionCondition(object):
    """ Defines a cylindrical or conical region between two poses """

    def __init__(self, approach_distance,
            approach_direction=axis.Z,
            verbose=False,
            slope=10.,
            pos_tol=1.5e-2,
            max_pos_tol=5.e-2,
            theta_tol=np.radians(10.)):

        self.verbose = verbose
        self.slope = slope
        self.pos_tol = pos_tol
        self.max_pos_tol = max_pos_tol
        self.theta_tol = theta_tol

        # -------
        # Set up pose
        self.approach_distance = approach_distance
        self.offset = np.eye(4)
        self.offset[approach_direction, axis.POS] = approach_distance


    def __call__(self, world_state, x, y):
        """ evaluates if x is approaching y """

        x_state = world_state[x]
        y_state = world_state[y]

        # See if we've chosen an approach vector for this object yet
        relative_pose = x_state.get_goal(y)
        if relative_pose is None or not y_state.observed:
            return False

        # Compute basic starting poses
        # cage_pose is the final grasp position
        if x_state.goal_is_relative:
            cage_pose = y_state.pose.dot(relative_pose)
        else:
            cage_pose = relative_pose

        # TODO remove this if we don't need it any more.
        # NOTE: this is for visualization of the approach region features
        # if x_state.ctrl is not None:
        #    x_state.ctrl.viz.visualize_pose(cage_pose)
        if self.verbose:
            logwarn("Entered: " + str(x) + ", " + str(y))


        # Approach pose is the standoff, defining the other end of the approach
        # "tube" that we are moving down
        approach_pose = cage_pose.dot(self.offset)
        # And this is the end effector positioin to check
        actor_pos = x_state.ee_pose[:3, axis.POS]

        # Compute poses from object - get positions to test
        # approach_pose = np.dot(y_state.pose, approach_pose)
        # cage_pose = np.dot(y_state.pose, cage_pose)
        frame_pos = approach_pose[:3, axis.POS]
        grasp_pos = cage_pose[:3, axis.POS]

        dr_a = x_state.inv_ee_pose.dot(approach_pose)
        dr_g = x_state.inv_ee_pose.dot(cage_pose)

        theta_a, _, _ = tra.rotation_from_matrix(dr_a)
        theta_grasp, _, _ = tra.rotation_from_matrix(dr_g)

        theta = min(abs(theta_a), abs(theta_grasp))

        # Approach vector is a line segment from "approach" to "cage"
        # Compute distance to line segment
        # Depending on where we are along this line we could increase
        # tolerance as well
        a = actor_pos - frame_pos
        b = grasp_pos - frame_pos
        l2 = np.linalg.norm(grasp_pos - frame_pos, ord=2)
        proj = np.dot(a, b) / l2
        dist_line = max(min(proj, 1), 0) / l2
        proj_pt = frame_pos + dist_line * (grasp_pos - frame_pos)
        dist = np.linalg.norm(proj_pt - actor_pos)
        pos_tol = min(self.pos_tol + (self.pos_tol * self.slope * l2), self.max_pos_tol)

        if self.verbose:
            # y_type = y_state.obj_type
            print('==============================')
            print("APPROACH FOR ", x, y) #, "type =", y_type)
            print("eeff pt =", actor_pos)
            print("proj pt =", proj_pt)
            print("appr pt =", frame_pos)
            print("cage pt =", grasp_pos)
            print("L2 =", l2)
            print("slope l2 =", l2 * self.slope)
            print("slope l2 tol =", l2 * self.slope * self.pos_tol)
            print("adj pos tol =", pos_tol)
            print("dist =", dist, "<", pos_tol)
            print("computed theta:", theta, "<", self.theta_tol, "...",
                  theta_a, theta_grasp)
            print("THETA:", theta < self.theta_tol)
            print("DIST:", dist < pos_tol)
            x_state.ref.ee_ref.set_pose_matrix(cage_pose)

        return dist < pos_tol and theta < self.theta_tol


def in_approach_region(
        D,
        ws,
        x,
        y,
        verbose=False,
        slope=0.,
        pos_tol=1.5e-2,
        max_pos_tol=5.e-2,
        theta_tol=np.radians(10.)):
    """
    Lookup object poses and check if we can grasp from negative x
    direction.
    """
    
    x_state = ws[x]
    y_state = ws[y]

    actor_pos = x_state.pose[:3, axis.POS]
    approach_pose = y_state.affordances["approach"]
    cage_pose = y_state.affordances["cage"]
    if approach_pose is not None and cage_pose is not None:

        # Compute poses from object
        approach_pose = np.dot(y_state.pose, approach_pose)
        cage_pose = np.dot(y_state.pose, cage_pose)
        frame_pos = approach_pose[:3, axis.POS]
        grasp_pos = cage_pose[:3, axis.POS]
        # frame_x = frame_pos[axis.X]
        # actor_x = actor_pos[axis.X]

        dr_a = x_state.inv_pose.dot(approach_pose)
        dr_g = x_state.inv_pose.dot(cage_pose)

        theta_a, _, _ = tra.rotation_from_matrix(dr_a)
        theta_grasp, _, _ = tra.rotation_from_matrix(dr_g)

        theta = min(abs(theta_a), abs(theta_grasp))

        # Approach vector is a line segment from "approach" to "cage"
        # Compute distance to line segment
        # Depending on where we are along this line we could increase
        # tolerance as well
        a = actor_pos - frame_pos
        b = grasp_pos - frame_pos
        l2 = np.linalg.norm(grasp_pos - frame_pos, ord=2)
        proj = np.dot(a, b) / l2
        dist_line = max(min(proj, 1), 0) / l2
        proj_pt = frame_pos + dist_line * (grasp_pos - frame_pos)
        dist = np.linalg.norm(proj_pt - actor_pos)
        pos_tol = min(pos_tol + (pos_tol * slope * l2), max_pos_tol)

        if verbose:
            y_type = y_state.obj_type
            print('==============================')
            print("APPROACH FOR ", x, y, "type =", y_type)
            print("proj pt =", proj_pt)
            print("L2 =", l2)
            print("slope l2 =", l2 * slope)
            print("slope l2 tol =", l2 * slope * pos_tol)
            print("dist =", dist, "<", pos_tol)
            print("computed theta:", theta, "<", theta_tol,
                "...",
                theta_a,
                theta_grasp)

        return dist < pos_tol and theta < theta_tol
    else:
        raise RuntimeError('you did not provide the right lookup tables: '
                           'could not find ' + str(y))
        # not supported from this direction
        return False





