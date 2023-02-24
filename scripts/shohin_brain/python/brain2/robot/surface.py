# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
from pyquaternion import Quaternion

import brain2.robot.entity as entity
import brain2.utils.transformations as tra
from brain2.task.world_state import EntityState


class Surface(entity.RobotEntity):
    """
    Describes a region in which objects can be placed, on top of another object or link.
    """

    def __init__(self, name, offset, extent,
                 rgba=[0., 0., 1., 0.5],
                 parent_frame=None,
                 opts=None, joint=False):
        entity_type = entity.JOINT if joint else entity.SURFACE
        super(Surface, self).__init__(name, None, None, physical=False, mobile=False, entity_type=entity_type)

        self.name = name
        self.offset = offset
        self.extent = extent
        self.rgba = rgba
        self.pose = np.eye(4)
        self.parent_inv_pose = np.eye(4)
        self.collisions = []
        self.ref = None # Reference to implementation specific data

        # Client ID used by the physics backend
        # Probably necessary
        self.id = None
        # Contains the reference to the actual parent physics object
        self.obj_ref = None

        # This is used for tracking possible alignments of a "top" surface
        self.opts = opts

        # Parent frame
        self.parent_frame = parent_frame

    def sample_pose(self, pose=None, height=0., var_theta=0):
        """
        Sample a position on the surface, according to the reference. Assume
        only variation in rotation is around theta axis.
        """
        if self.pose is None:
            raise RuntimeError('surface ' + str(self.name) + ' has not been initialized')


        x = (np.random.random() * self.extent[0]) - (0.5 * self.extent[0])
        y = (np.random.random() * self.extent[1]) - (0.5 * self.extent[1])
        z = height
        #z = self.extent[2] + height

        # Orientation: sample random thetas
        if var_theta > 0:
            theta = np.random.rand() * 2.0 * np.pi * var_theta
            #sampled_pose = sampled_pose.dot(tra.euler_matrix(theta, 0, 0))
            sampled_pose = tra.euler_matrix(0, 0, theta)
        else:
            sampled_pose = np.eye(4)

        sampled_pose[:3, 3] = np.array([x, y, z])

        T = self.pose.dot(sampled_pose)

        if pose is not None:
            pose = np.copy(pose)
            pose[:3, 3] = np.array([0, 0, 0,])
            # apply canonical pose so objects are oriented properly after the fact
            return T.dot(pose)
        else:
            return T

    def sample_pos(self, height=0.):
        x = (np.random.random() * self.extent[0]) - (0.5 * self.extent[0])
        y = (np.random.random() * self.extent[1]) - (0.5 * self.extent[1])
        z = height
        return np.array([x, y, z])

    def sample_xyz(self):
        """ Sample a fully random point in the volume """
        x = (np.random.random() * self.extent[0]) - (0.5 * self.extent[0])
        y = (np.random.random() * self.extent[1]) - (0.5 * self.extent[1])
        z = (np.random.random() * self.extent[2]) - (0.5 * self.extent[2])
        return np.array([x, y, z])

    def sample_rot(self, var_theta=0.1):
        if var_theta > 0:
            theta = np.random.rand() * 2.0 * np.pi * var_theta
        else:
            theta = 0.
        return Quaternion(axis=np.array([0., 0., 1.,]), radians=theta)

    def update(self):
        raise NotImplementedError('must provide an update function')

    def overlap_check(self, ref):
        raise NotImplementedError('must provide an overlap check function')


class SurfaceState(EntityState):
    """ Placeholder object to store surface entities """
    def __init__(self, surface):
        super(SurfaceState, self).__init__()
        self.surface = surface
        self.pose = np.eye(4)
        self.ref = surface.ref
        self.observed = False
        self.updated = False
        self.stable = True
        self.name = surface.name

        self.overlap_check = self.surface.overlap_check

        self.objs = []

    def copy(self):
        return SurfaceState(self.surface)

    def apply(self):
        """ Make the changes necessary to the world representation -- in this case that's nothing
        since the surfaces are managed via their parent objects and should be constant. """
        pass

    def update(self, t):
        """ No update function here, since these are created and managed via the planner """
        self.pose = self.surface.pose
        self.updated = True
        self.observed = True


class DrawerState(EntityState):
    def __init__(self, ref, surface, q=0, pose=None):
        super(DrawerState, self).__init__()
        self.surface = surface
        self.ref = ref
        self.q = q
        self.name = surface.name

        # Initialize with this pose
        if pose is None:
            self.pose = np.copy(surface.pose)
        else:
            self.pose = pose

        self.observed = False
        self.updated = False
        self.stable = True

        self.obs_t = 0
        self.t = 0

        self.overlap_check = self.surface.overlap_check

        self.objs = []

    def _update(self, pose=None, q=None, obs_t=0., t=0.,
                max_obs_age=0.1):
        self.obs_t = obs_t
        self.t = t
        #self.pose = pose
        self.ref.set_joint_position(self.q)
        self.q = q
        self.pose = self.ref.get_pose()

        #print("==========")
        #print(self.surface.name)
        #print(self.pose)
        #print("==========")
        #raw_input('ENTER')

        self.stable = True
        self.updated = True
        if abs(t - obs_t) > max_obs_age:
            self.observed = False
        else:
            self.observed = True

    def copy(self):
        return DrawerState(self.ref, self.surface, self.q, np.copy(self.pose))

    def apply(self):
        """ Make the changes necessary to the world representation """
        self.ref.set_joint_position(self.q)
