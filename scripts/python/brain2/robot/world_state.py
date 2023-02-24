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
import brain2.utils.pose as pose_tools
import brain2.utils.transformations as tra

from brain2.utils.info import logwarn
from brain2.task.world_state import EntityState

class RobotEntityState(EntityState):

    def __init__(self, ref, obj_type=None, pose=None, config=None, control=None, *args, **kwargs):
        """
        :param ref: object reference allowing us to get the position
        :param pose: where the object is
        :param config: configuration of its joints
        :param ctrl: control interface for use moving the robot around
        """
        super(RobotEntityState, self).__init__()
        self.ref = ref
        self.pose = pose
        self.ee_pose = None
        self.inv_ee_pose = None
        self.q = config
        self.dq = None
        self.name = ref.name
        self.type = obj_type
        self.ctrl = control

        # Attachment belief
        self.attached_to = None
        self.attached_pose = None
        self.attached_inv_pose = None

        # Observation data
        self.observed = False
        self.stable = False
        self.stable = False
        self.obs_t = 0.

        # prev observation
        self.prev_obs_t = 0.
        self.prev_pose = None
        self._v = float('Inf')
        self._dt = 0.

        self.clear()
    
        # Gripper position
        self.gripper_state = 0.
        self.gripper_fully_open = False
        self.gripper_fully_closed = False
        self.gripper_moving = False

        # Goals and other information
        self.goal_pose = None
        self.goal_obj = None
        self.goal_is_relative = False

        # Planning data
        # These are for storing information on motion planning from this world state
        self.arm_tree = None
        self.base_tree = None

    def is_visible(self, *args, **kwargs):
        """ Can we be expected to see this object given current world state? """
        return self.ref.is_visible(self.pose, *args, **kwargs)

    def set_ctrl(self, ctrl):
        """ Update the control used to move to robot """
        if self.ctrl is not None:
            raise RuntimeError('control interface should not change! What are '
                               'you trying to do here?')
        self.ctrl = ctrl

    def get_control_interface(self):
        """ Return control interface for sending commands to a real or imagined robot. """
        return self.ctrl

    def reset(self):
        """ Clear plans for control. """
        # Gripper position
        self.gripper_state = 0.
        self.gripper_fully_open = False
        self.gripper_fully_closed = False
        self.gripper_moving = False

        # Goals and other information
        self.goal_pose = None
        self.goal_obj = None
        self.goal_is_relative = False

        # Trajectory planning distributions
        self.arm_tree = None
        self.base_tree = None

    def set_goal(self, obj, pose, relative=True):
        """ Not sure if this is really how we want to do it. But this will
        tell us how we were planning to grasp an object. """
        self.goal_pose = pose
        self.goal_obj = obj
        self.goal_is_relative = relative

    def get_goal(self, obj):
        """
        When planning a motion we can set a pose for the next policy to follow.
        """
        if obj == self.goal_obj:
            return self.goal_pose
        else:
            return None

    def clear_goal(self):
        self.goal_pose = None
        self.goal_obj = None
        self.goal_is_relative = False

    def copy(self):
        """
        Create a duplicate of the object with metadata reference and state
        information.

        We DO NOT copy:
        - control information
        - planning data
        """
        pose = self.pose.copy() if self.pose is not None else None
        q = self.q.copy() if self.q is not None else None
        new_state = type(self)(self.ref, self.type, pose, q, self.ctrl)
        new_state.observed = self.observed
        new_state.obs_t = self.obs_t
        new_state.prev_obs_t = self.prev_obs_t
        new_state._dt = self._dt
        new_state._v = self._v
        new_state.attached_to = self.attached_to
        new_state.attached_pose = self.attached_pose
        new_state.attached_inv_pose = self.attached_inv_pose
        new_state.ee_pose = self.ee_pose.copy() if self.ee_pose is not None else None
        return new_state

    def attach(self, to_obj, to_pose=None, to_obj_state=None):
        self.attached_to = to_obj
        self.attached_pose = to_pose
        if to_pose is not None:
            self.attached_inv_pose = tra.inverse_matrix(to_pose)

    def detach(self):
        """ Detach state and clear goals """
        self.attached_to = None
        self.attached_pose = None
        self.attached_inv_pose = None
        # self.goal_pose = None
        # self.goal_obj = None

    def _update(self, base_pose=None, ee_pose=None, q=None, obs_t=0., t=0.,
                max_obs_age=0.1):
        """
        Update internals. This should be called by everything that tries to
        update the state of the entity.
        """
        self.observed = True
        self.prev_pose = self.pose
        self.pose = base_pose

        # New sensor measurement has arrived
        # Update measurement information
        if (obs_t - self.obs_t) > 1e-6:
            self.prev_obs_t = self.obs_t
        self.obs_t = obs_t
        self.update_t = t
        self._dt = abs(self.obs_t - self.prev_obs_t)

        # Pose informaiton if we didn't update anything else
        if self.pose is not None:
            self.inv_pose = tra.inverse_matrix(self.pose)
        else:
            self.observed = False
            self.stable = False
            return False

        if ee_pose:
            self.ee_pose = ee_pose
            self.inv_ee_pose = tra.inverse_matrix(self.ee_pose)
        self.q = q

        # print(self._dt, self.obs_t, self.prev_obs_t)
        if self.pose is not None and self.prev_pose is not None:
            self._v = (np.linalg.norm(self.pose[:3, 3] - self.prev_pose[:3, 3])
                                      / self._dt)
            self.stable = True if self._v < 5e-3 else False
            # print("---", self.name, self._dt, self._v, self.stable)
        else:
            self.stable = False
            # print("---", self.name, "unstable not observed enough")

        # print("obs t", obs_t, "t", t, t - obs_t)
        # Check to see if this is within reasonable measurement time
        if self.pose is not None and abs(t - obs_t) > max_obs_age:
            self.observed = False
        else:
            self.observed = True

        # print(self.name, "observed =", self.observed)
        return self.observed

    def set_base_pose_quat(self, trans, rot):
        self.pose = pose_tools.make_pose(trans, rot)

    def set_base_pose(self, pose):
        self.pose = pose

    def set_ee_pose(self, pose):
        self.ee_pose = pose
        self.inv_ee_pose = tra.inverse_matrix(self.ee_pose)

    def set_config(self, q, dq=None):
        self.q = q
        self.dq = dq

class HumanHandState(RobotEntityState):

    def __init__(self, ref, obj_type=None, pose=None, config=None, *args, **kwargs):
        super(HumanHandState, self).__init__(ref, obj_type, pose, config, *args, **kwargs)
        self.obj_pose = None
        self._obs_count =  0
        self._obs_obj_count =  0
        self.hand_shape = "NA"
        self.has_obj = False

    def _update(self, base_pose=None, obj_pose=None,
                has_obj=False,
                obj_observed=False,
                obs_t=0., t=0.,
                max_obs_age=0.1,
                hand_shape='NA'):
        """
        Update internals. This should be called by everything that tries to
        update the state of the entity.
        """
        self.hand_shape = hand_shape
        
        # TODO support object pose
        if has_obj and obj_observed:
            base_pose[:3, 3] = obj_pose[:3, 3]

        # New sensor measurement has arrived
        # print("obs t", obs_t, "t", t, t - obs_t)
        # Check to see if this is within reasonable measurement time
        if base_pose is None or abs(t - obs_t) > max_obs_age:
            # If observation is too old, we've lost track of the hand.
            self.observed = False
            self._obs_count =  0
            self._obs_obj_count =  0
        elif (obs_t - self.obs_t) > 1e-6:
            # If observation is too young, assume it's the same as the last one
            # and ignore it. Otherwise, we'll count this as an observation and
            # continue on from here.
            self.prev_obs_t = self.obs_t
            self.observed = True
            self._obs_count += 1
            self._obs_obj_count += 1
            self.prev_pose = self.pose
            self.pose = base_pose
            self.inv_pose = tra.inverse_matrix(self.pose)
        else:
            # Observed, but no new observation has arrived
            # Do not update if the observation was too new / had the same time
            # as an old message -- this is probably just a network blip, or
            # we're updating faster than the neural networks are.
            self.observed = True


        # Update object pose
        self.obj_pose = obj_pose
        self.has_obj = has_obj

        self.obs_t = obs_t
        self.update_t = t
        self._dt = abs(self.obs_t - self.prev_obs_t)
        if self.prev_pose is not None and self._obs_count > 5:
            self._v = (np.linalg.norm(self.pose[:3, 3] - self.prev_pose[:3, 3])
                                      / self._dt)
            self.stable = True if self._v < 0.5 else False
            #if self.name == "right":
            #    print("OBSERVING:", self.name, "timestep =", self._dt, "velocity =",
            #            self._v, "stable =", self.stable, "has obj =",
            #            self.obj_pose is not None)
        else:
            #if self.name == "right":
            #    print("OBSERVING", self.name, "unstable not observed enough")
            #    self.stable = False
            pass

        return self.observed

