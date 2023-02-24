# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

from brain2.utils.info import logwarn, logerr
from collections import namedtuple
from brain2.robot.entity import RobotEntity
from brain2.bullet.load import create_obj
from brain2.utils.camera import get_camera_preset

import brain2.utils.image as image
import brain2.utils.axis as axis
import brain2.utils.transformations as tra
import brain2.robot.entity as entity
import pybullet as pb
import pybullet_data

import math
import numpy as np
import time

# ------------------------------------------------
# TODO: this is copied from caelan
JOINT_TYPES = {
    pb.JOINT_REVOLUTE: 'revolute', # 0
    pb.JOINT_PRISMATIC: 'prismatic', # 1
    pb.JOINT_SPHERICAL: 'spherical', # 2
    pb.JOINT_PLANAR: 'planar', # 3
    pb.JOINT_FIXED: 'fixed', # 4
    pb.JOINT_POINT2POINT: 'point2point', # 5
    pb.JOINT_GEAR: 'gear', # 6
}
JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction',
                                     'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName',
                                     'jointAxis',
                                     'parentFramePos', 'parentFrameOrn',
                                     'parentIndex'])

CameraInfo = namedtuple('CameraInfo', ['width', 'height', 'viewMatrix', 'projectionMatrix', 'cameraUp', 'cameraForward',
                                       'horizontal', 'vertical', 'yaw', 'pitch', 'dist', 'target'])

RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal'])

def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])

def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])

def quat_from_euler(euler):
    return pb.getQuaternionFromEuler(euler)

def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return (point, quat_from_euler(euler))

def invert(pose):
    (point, quat) = pose
    return p.invertTransform(point, quat)

def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = pb.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose

def tform_point(affine, point):
    return multiply(affine, Pose(point=point))[0]

def get_camera(client=0):
    return CameraInfo(*pb.getDebugVisualizerCamera(physicsClientId=client))

def set_camera(yaw, pitch, distance, target_position=np.zeros(3), client=0):
    pb.resetDebugVisualizerCamera(distance, yaw, pitch, target_position,
            physicsClientId=client)

def get_pitch(point):
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))

def get_yaw(point):
    dx, dy, dz = point
    return np.math.atan2(dy, dx)

def set_camera_pose(camera_point, target_point=np.zeros(3), client=0):
    delta_point = np.array(target_point) - np.array(camera_point)
    distance = np.linalg.norm(delta_point)
    yaw = get_yaw(delta_point) - np.pi/2 # TODO: hack
    pitch = get_pitch(delta_point)
    pb.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                 target_point, physicsClientId=client)

def set_camera_pose2(world_from_camera, distance=2, client=9):
    target_camera = np.array([0, 0, distance])
    target_world = tform_point(world_from_camera, target_camera)
    pos, rot = world_from_camera
    camera_world = pos
    set_camera_pose(camera_world, target_world, client=client)
# ---------- end Caelan -------------------


class CameraReference(object):
    """ Class storing camera information and providing easy image capture """

    def __init__(self, pose, pose_is_matrix=True, distance=2, parent=None,
            link_idx=None,
            proj_near=0.01, proj_far=5., proj_fov=60., img_width=640,
            img_height=480, client=0):

        self.client = client
        self.proj_near = proj_near
        self.proj_far = proj_far
        self.proj_fov = proj_fov
        self.img_width = img_width
        self.img_height = img_height
        self.x_offset = self.img_width / 2.
        self.y_offset = self.img_height / 2.

        # Projection Matrix 
        self.proj_matrix = pb.computeProjectionMatrixFOV(fov=self.proj_fov,
                                                         aspect=float(self.img_width)/self.img_height,
                                                         nearVal=self.proj_near,
                                                         farVal=self.proj_far,)

        self.set_pose(pose, pose_is_matrix, distance)

    def set_pose(self, pose, matrix=True, distance=1):
        """ Set the position of a camera """
        if matrix:
            pos, rot = pose[:3, 3], tra.quaternion_from_matrix(pose)
            roll, pitch, yaw = tra.euler_from_matrix(pose)
        else:
            pos, rot = pose
            roll, pitch, yaw = tra.euler_from_quaternion(pose)
        
        self.view_matrix = pb.computeViewMatrixFromYawPitchRoll(pos, distance,
                                                                yaw=yaw,
                                                                pitch=pitch,
                                                                roll=roll,
                                                                upAxisIndex=2,
                                                                physicsClientId=self.client)
        # For debugging 
        # set_camera_pose2((pos, rot), distance=2, client=self.client)

    def capture(self):
        """ returns data from camera """
        w, h, rgb, depth, mask = pb.getCameraImage(self.img_width,
                                                   self.img_height,
                                                   viewMatrix=self.view_matrix,
                                                   projectionMatrix=self.proj_matrix,
                                                   flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX |
                                                         pb.ER_BULLET_HARDWARE_OPENGL)
        return rgb, depth, mask


class JointReference(RobotEntity):
    """Create reference to a single joint"""
    def __init__(self, parent, link_name, joint_idx):
        super(JointReference, self).__init__(link_name, parent.id, interface=None,
                physical=True, verbose=0, mobile=False, entity_type=entity.JOINT)
        self.parent = parent
        self.joint_idx = joint_idx
        self.surface = {}
        self.min = 0
        self.max = 0
        self.id = parent.id
        self.client = self.parent.client

    def set_joint_position(self, q):
        """ Teleport the joint around """
        pb.resetJointState(self.parent.id,
                           self.joint_idx,
                           q,
                           targetVelocity=0,
                           physicsClientId=self.client)

    def get_joint_position(self):
        return pb.getJointState(self.parent.id, self.joint_idx, self.client)[0]

    def get_pose(self, matrix=True):
        """ Read the pose of the link from the simulation backend """
        res = pb.getLinkState(self.id, self.joint_idx)
        pos, rot = res[:2]

        if matrix:
            pose = tra.quaternion_matrix(rot)
            pose[:axis.POS, axis.POS] = pos
            return pose
        else:
            return pos, rot

    def get_link_idx(self):
        return self.idx


class ArticulatedObject(RobotEntity):
    """Create reference to a world object"""
    def __init__(self, name, object_id, interface, physical=True, verbose=0,
                 mobile=False, padding=0.):
        super(ArticulatedObject, self).__init__(name, object_id, interface,
                physical, verbose, mobile, entity_type=entity.OBJECT)

        self.iface = interface
        self.client = interface.client
        self._compute_default_pose()

        self._get_joint_info()

        self.dof = len(self.joint_info)
        self.min = np.zeros(self.dof,)
        self.max = np.zeros(self.dof,)
        self.range = np.zeros(self.dof,)
        self.num_joints = self.dof

        # ===========
        # collisions
        self.allowed_collisions = {}
        self.allowed_self_collisions = np.zeros((self.dof, self.dof), dtype=np.bool)
        # ===========

        self._active_joints = range(self.dof)

        self.padding = padding
        for i in range(self.dof):
            info = self.joint_info[i]
            lv = info.jointLowerLimit
            uv = info.jointUpperLimit
            if uv < lv:
                tmp = uv
                uv = lv
                lv = tmp
            self.range[i] = uv - lv
            self.min[i] = lv + (self.padding * self.range[i])
            self.max[i] = uv - (self.padding * self.range[i])

        # --- Active joints setup
        self.active_dof = self.dof
        self.active_min = np.copy(self.min)
        self.active_max = np.copy(self.max)
        self.active_range = np.copy(self.range)

        self.ee_idx = None

        # Gripper stuff
        self._gripper_open_pos = None
        self._gripper_closed_pos = None
        self._gripper_joints = None
        self._gripper_range = None
        self._gripper_dof = 0

        self.verbose = verbose

        self.constraint = None
        #self.constraints_to = set()

    def get_link_idx(self):
        if self.ee_idx is not None:
            return self.ee_idx
        else:
            return -1

    def attach_to(self, other, pose):
        """ Create a constraint between these two objects """
        pos1 = pose[:3, 3]
        rot1 = tra.quaternion_from_matrix(pose)
        pose = tra.inverse_matrix(pose)
        pos2 = pose[:3, 3]
        rot2 = tra.quaternion_from_matrix(pose)
        if other.constraint is None:
            self.constraint = pb.createConstraint(self.id, -1, other.id, other.get_link_idx(),
                                                  pb.JOINT_FIXED, [0, 0, 0], pos1, pos2,
                                                  physicsClientId=self.client)
            return True
        else:
            return False

    def detach(self, other):
        if self.constraint is not None:
            pb.removeeConstraint(self.constraint, physicsClientId=self.client)

    def get_joint_reference(self, joint_name=None, link_name=None):
        if joint_name is not None:
            idx = self.get_joint_index(joint_name)
        elif link_name is not None:
            idx = self.get_link_index(joint_name)
        else:
            raise RuntimeError('joint rference could not be created')

        link_name = self.joint_info[idx].linkName.decode('ascii')
        ref = JointReference(self, link_name, idx)
        for key, val in self.surfaces.items():
            if val.parent_frame == link_name:
                # Add surface reference here
                ref.add_surface_reference(key, val)
                    
                # Add affordances from surface
                ref.affordances.update(val.affordances)
                
        ref.min =  self.joint_info[idx].jointLowerLimit
        ref.max =  self.joint_info[idx].jointUpperLimit

        return ref

    def set_active_joints(self, names):
        """Get the dof and a list of active joint limits; set robot up
        properly for control and planning."""
        self._active_joints = []
        for name in names:
            info = self.get_joint_info_by_name(name)
            self._active_joints.append(info.jointIndex)
        self.dof = len(self._active_joints)
        self._update_limits()

    def set_gripper_joints(self, joint_names, open_pos, closed_pos):
        self._gripper_joints = []
        for name in joint_names:
            info = self.get_joint_info_by_name(name)
            self._gripper_joints.append(info.jointIndex)
        self._gripper_open_pos = open_pos
        self._gripper_closed_pos = closed_pos
        self._gripper_dof = len(self._gripper_joints)
        self._gripper_range = open_pos - closed_pos
 
    def open_gripper(self):
        self.set_joint_positions([self._gripper_open_pos] * self._gripper_dof, self._gripper_joints)

    def close_gripper(self):
        self.set_joint_positions([self._gripper_closed_pos] * self._gripper_dof,
                                 self._gripper_joints)

    def close_gripper_around(self, ref, step=0.02, max_distance=0):
        """
        Close the gipper, why not.
        To be a bit more helpful, this makes it so that we can guess how open
        or closed the gripper will be when we grab this object. it does so by
        stepping closer until it detects a contact.
        """
        prog = 0
        moving = np.ones(self._gripper_dof)
        pos = np.ones(self._gripper_dof) * self._gripper_open_pos
        while np.any(moving):
            # check for contacts on each finger
            pos -= (step * self._gripper_range * moving)
            self.set_joint_positions(pos, self._gripper_joints)
            for i in range(self._gripper_dof):
                res = pb.getClosestPoints(bodyA=self.id,
                                          bodyB=ref.id,
                                          distance=max_distance,
                                          linkIndexA=self._gripper_joints[i],
                                          linkIndexB=-1,
                                          physicsClientId=self.client)
                if len(res) > 0 or pos[i] <= self._gripper_closed_pos:
                    moving[i] = 0

    def get_joint_names(self):
        """ access list of joint names """
        return [self.joint_info[i].jointName for i in self._active_joints]

    def set_ee(self, name):
        """Set end effector index for planning"""
        info = self.get_joint_info_by_name(name)
        self.ee_idx = info.jointIndex
        print("Setting ee idx for obj =", self.id, "to", self.ee_idx)
        # raw_input('has been set')

    def get_ee_pose(self, matrix=True):
        """ Get the pose of the end effector after setting the joint position """
        #raw_input('has been set')
        res = pb.getLinkState(self.id, self.ee_idx, physicsClientId=self.client)
        pos, rot = res[:2]

        if matrix:
            pose = tra.quaternion_matrix(rot)
            pose[:axis.POS, axis.POS] = pos
            return pose
        else:
            return pos, rot
    
    def get_all_active_joints_poses(self):
        output = {}
        for joint_name, joint_index in zip(self.get_joint_names(), self._active_joints):
            res = pb.getLinkState(self.id, joint_index, physicsClientId=self.client)
            pos, rot = res[:2]
            output[joint_name] = tra.quaternion_matrix(rot)
            output[joint_name][:axis.POS, axis.POS] = pos
        return output


        
    def get_max_velocities(self):
        """
        Get max velocities for each joint
        """
        return [self.joint_info[i].jointMaxVelocity for i in self._active_joints]

    def _update_limits(self):
        self.active_dof = len(self._active_joints)
        self.active_min = np.zeros(self.active_dof,)
        self.active_max = np.zeros(self.active_dof,)
        self.active_range = np.zeros(self.active_dof,)
        for i, idx in enumerate(self._active_joints):
            info = self.joint_info[idx]
            lv = info.jointLowerLimit
            uv = info.jointUpperLimit
            self.active_min[i] = lv
            self.active_max[i] = uv
            self.active_range[i] = uv - lv

    def _get_joint_info(self):
        """Get and update joint information"""
        # TODO: this code is pybullet specific
        infos = []
        for i in range(pb.getNumJoints(self.id, physicsClientId=self.client)):
            info = JointInfo(*pb.getJointInfo(self.id, i, physicsClientId=self.client))
            infos.append(info)
        self.joint_info = infos

    def add_surface(self, surface):
        """Attach semantics for placing on top of other objects."""

        oid = self.interface.add_surface(self.name, surface)
        surface.id = oid
        surface.obj_ref = self
        surface.overlap_check = (lambda ref: 
            len(pb.getClosestPoints(bodyA=surface.id,
                                bodyB=ref.id,
                                distance=0.,
                                physicsClientId=self.client)) > 0)
        if surface.parent_frame is None:
            def _update_surface_pos(matrix=False):
                res = pb.getBasePositionAndOrientation(self.id)
                pos = [x1 + x2 for x1, x2 in zip(res[0], surface.offset)]
                if matrix:
                    pose = tra.quaternion_matrix(res[1])
                    pose[:3, 3] = pos
                    return pose
                else:
                    return pos, res[1]
        else:
            link_idx = -1
            for i, info in enumerate(self.joint_info):
                # print(self.joint_info[i].linkName)
                if info.linkName.decode('ascii') == surface.parent_frame:
                    link_idx = i
                    break
            else:
                raise RuntimeError('link not found: ' + str(surface.parent_frame))
            def _update_surface_pos(matrix=False):
                res = pb.getLinkState(self.id, link_idx)
                pos = [x1 + x2 for x1, x2 in zip(res[0], surface.offset)]
                if matrix:
                    pose = tra.quaternion_matrix(res[1])
                    pose[:3, 3] = pos
                    return pose
                else:
                    return pos, res[1]

        # Fix pose for the surface
        surface.update = _update_surface_pos
        surface.pose = surface.update(matrix=True)

        # Add the surface data and ID
        self.add_surface_reference(surface.name, surface)

    def get_link_index(self, name):
        for i, info in enumerate(self.joint_info):
            if info.linkName.decode('ascii') == name:
                return i
        return None

    def get_joint_index(self, name):
        for i, info in enumerate(self.joint_info):
            if info.linkName.decode('ascii') == name:
                return i
        return None

    def is_visible(self, camera_pose=None):
        """ Do a single ray trace from position to camera pose """
        if camera_pose is None:
            camera_pose = self.iface.camera_pose[0]

        # Do a ray check
        result = pb.rayTest(camera_pose, self.pose[0], physicsClientId=self.client)
        visible = result[0] >= 0
        if not visible:
            print(RayResult(*result))
            raw_input()
        return visible

    def print_joint_info(self):
        """Just a helper to debug things"""
        for idx, i in enumerate(self.joint_info):
            print("---------", idx, "----------")
            print(i)

    def sample_surface_pose(self, surface_name, var_theta=1.):
        """ Sample a pose on top of this object """
        surface = self.surfaces[surface_name]
        pos = surface.sample_pos() + self.pose[0] + surface.offset
        rot = self.pose[1] * surface.sample_rot()
        #return surface.sample_pose(np.eye(4))
        return pos, rot

    def set_pose(self, pos, quat, wxyz=False):
        """Set pose in the scene, by name."""
        self.pose = (pos, quat)
        if wxyz:
            quat = [quat[1], quat[2], quat[3], quat[0]]
        pb.resetBasePositionAndOrientation(self.id, pos, quat, physicsClientId=self.client)
        for surface in self.surfaces.values():
            spos, squat = surface.update(matrix=False)
            surface.pose[:3, 3] = spos
            pb.resetBasePositionAndOrientation(surface.id, spos, quat, physicsClientId=self.client)

    def reset_pose(self):
        self.set_pose(self.pose[0], self.pose[1])

    def get_pose(self, matrix=True):
        """ Get base position """
        pos, rot = pb.getBasePositionAndOrientation(self.id)

        if matrix:
            pose = tra.quaternion_matrix(rot)
            pose[:axis.POS, axis.POS] = pos
            return pose
        else:
            return pos, rot

    def _compute_default_pose(self):
        pos, quat = pb.getBasePositionAndOrientation(self.id)
        pose = tra.quaternion_matrix(quat)
        pose[:3, 3] = pos
        self.default_pose = pose

    def set_pose_matrix(self, pose):
        pos = pose[:axis.POS, axis.POS]
        quat = tra.quaternion_from_matrix(pose)
        self.pose = pos, quat
        pb.resetBasePositionAndOrientation(self.id, pos, quat, physicsClientId=self.client)

        # TODO: code re-use
        for sid, surface in zip(self.surface_ids, self.surfaces):
            spos = [x1 + x2 for x1, x2 in zip(pos, surface.offset)]
            pb.resetBasePositionAndOrientation(sid, spos, quat, physicsClientId=self.client)

    def get_joint_info_by_name(self, name):
        for info in self.joint_info:
            if info.jointName.decode('ascii') == name:
                return info
        else:
            raise ValueError('invalid joint name {}'.format(name))

    def set_allowed_self_collisions(self, i, j, val):
        """Make sure we don't check collisions between things unnecessarily"""
        info1 = self.get_joint_info_by_name(i)
        info2 = self.get_joint_info_by_name(j)
        i = info1.jointIndex
        j = info2.jointIndex
        self.allowed_self_collisions[i, j] = val
        self.allowed_self_collisions[j, i] = val

    def set_allowed_collisions(self, other, idx=-1, joint_name=None, val=1.):
        """What are other parts of the robot allowed to touch?"""
        if other.id not in self.allowed_collisions:
            self.allowed_collisions[other.id] = np.zeros(self.num_joints+1, dtype=np.bool)
        if joint_name is not None:
            info1 = self.get_joint_info_by_name(joint_name)
            if info1 is None:
                raise RuntimeError('not understood: joint named ' + str(i)
                                   + ' does not seem to exist.')
            i = info1.jointIndex
            self.allowed_collisions[other.id][i + 1] = val
        else:
            self.allowed_collisions[other.id][idx + 1] = val

    def sample_uniform(self):
        return (np.random.random() * self.active_range) + self.active_min

    def set_joint_positions(self, positions, joint_idx=None):
        if joint_idx is None:
            joint_idx = self._active_joints
        for i, q in zip(joint_idx, positions):
            pb.resetJointState(self.id, i,
                               q,
                               targetVelocity=0,
                               physicsClientId=self.client)

    def get_joint_positions(self):
        res = pb.getJointStates(self.id, self._active_joints, self.client)
        return np.array([r[0] for r in res])

    def config_within_limits(self, q):
        """ is this actually valid? """
        #if self.iface.verbose:
        #    #logerr("Joint position not within limits:")
        #    #logerr("pos = " + str(q))
        #    #logerr("min = " + str(self.active_min))
        #    #logerr("max = " + str(self.active_max))
        return (q >= self.active_min).all() and (q <= self.active_max).all()

    def check_self_collisions(self, max_distance=0):
        """
        Check all self collisions. Could be made more efficient.
        """
        for i in range(self.dof):
            for j in range(self.dof):
                if abs(i - j) < 2 or self.allowed_self_collisions[i, j]:
                    continue
                res = pb.getClosestPoints(bodyA=self.id,
                                          bodyB=self.id,
                                          distance=max_distance,
                                          linkIndexA=i,
                                          linkIndexB=j,
                                          physicsClientId=self.client)
                if len(res) > 0:
                    if self.verbose:
                        print("links", i, self.joint_info[i].jointName, "and",
                              j, self.joint_info[j].jointName,
                              "are colliding")
                    return True
        return False

    def check_pairwise_collisions(self, other, max_distance=0, verbose=False,
                                  suppressed_links=None):
        """
        Check all pairwise collisions.
        :param other - the other articulated body which we want to check against
        """
        res = pb.getClosestPoints(bodyA=self.id,
                                  bodyB=other.id,
                                  distance=max_distance,
                                  physicsClientId=self.client)
        #print("=========")
        #print(self.name, other.name)
        #print(self.allowed_collisions)
        #print(res)
        #print(suppressed_links)
        _check_links = suppressed_links is not None

        # If we have an allowed collision matrix...
        if other.id in self.allowed_collisions:
            # Check allowed collision matrix
            acm = self.allowed_collisions[other.id]
            for pt in res:
                _, body1, body2, link1, link2 = pt[:5]
                if not acm[link1+1]:
                    if verbose:
                        if link1 >= 0:
                            link1 = self.joint_info[link1].jointName
                        if link2 >= 0:
                            link2 = other.joint_info[link2].jointName
                        print("links", self.name, body1, link1, "and",
                              other.name, body2, link2, "are colliding")
                    return True
        elif _check_links:
            for pt in res:
                # Get contact information here
                _, body1, body2, link1, link2 = pt[:5]
                if suppressed_links and (body2, link2) in suppressed_links:
                    continue
            raise NotImplementedError('Tried to check links but did not check anything')
        elif len(res) > 0:
            if verbose:
                print("pairwise collision:", self.id, self.name, other.id,
                        other.name)
            return True
        return False

    def validate(self, q=None, max_pairwise_distance=0, illegal_affordances=None,
                 suppressed_objs=None, suppressed_refs=None, verbose=False):
        """ Is [q] a valid configuration for this entity?
        State:
            - Given by q, assumed to be a configuration only
        Returns:
            - True if the state is valid (given by q)
            - False if the state is invalid """
        return not self.iface.check_collisions(self, q, max_pairwise_distance,
                illegal_affordances, suppressed_objs, suppressed_refs,
                verbose=verbose)

class BulletInterface(object):
    """
    Track objects and do collision queries. Should be replaced by gym or
    omniverse when they support ROS integration.
    """

    def __init__(self, gui=False, add_ground_plane=False, verbose=0):
        self.name_to_id = {}
        self.id_to_pose = {}
        self.allowed_collisions = {}
        self.gui = gui
        self.verbose = verbose

        self.all_objects = set()
        self.object_interfaces = {}

        # contains ids only for physical objects
        self.oids = []

        if self.gui:
            iface = pb.GUI
        else:
            iface = pb.DIRECT

        # Set up the client
        self.client = pb.connect(iface)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client)
        if add_ground_plane:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            self.ground_plane = pb.loadURDF("plane.urdf", physicsClientId=self.client)

        self.camera_pose = None
        self.cameras = []

    def check_collisions(self, obj, q=None, max_pairwise_distance=0,
                         illegal_affordances=None, suppressed_objs=None,
                         suppressed_refs=None, verbose=False):
        """
        Simple collision check for an object, assuming object is immobile.
        This could be made more efficient, but it's ok for now.

        Returns TRUE if in collision, FALSE if not in collision.
        """
        if q is not None and obj.dof > 0 and not obj.config_within_limits(q):
            logerr("Joint position not within limits: " + str(q))
            return True # invalid pose

        _check_affordances = illegal_affordances is not None
        _check_suppressed = suppressed_objs is not None
        _check_suppressed_refs = suppressed_refs is not None
        _ignore_ids = False
        ignored_ids = set()

        # Ignore specific parts of an object
        if _check_suppressed_refs:
            suppressed_links = {}
            for ref in suppressed_refs:
                if ref.entity_type == entity.JOINT:
                    s_idx = ref.joint_idx + 1
                    if ref.id not in suppressed_links:
                        suppressed_links[ref.id] = set()
                    suppressed_links[ref.id].add(s_idx)
                else:
                    _ignore_ids = True
                    ignored_ids.add(ref.id)
        else:
            suppressed_links = None

        if q is not None and obj.dof > 0:
            # Reset joints to right position
            obj.set_joint_positions(q)

        # Loop over all the objects and perform collision check
        for name, obj2 in self.object_interfaces.items():
            res = None
            if _ignore_ids and obj2.id in ignored_ids:
                # skip this object
                continue
            if name == obj.name:
                res = obj.check_self_collisions()
                if self.verbose and res:
                    print("self collision found in", obj.name)
            elif (obj2.is_physical
                    and (not _check_suppressed or name not in suppressed_objs)
                    or (_check_affordances and name not in illegal_affordances)):

                # Only do this if its present
                if _check_suppressed and _check_suppressed_refs:
                    _obj_suppressed_links = suppressed_links[obj2.id]
                else:
                    _obj_suppressed_links = None

                # Not just something we added to visualize
                res = obj.check_pairwise_collisions(obj2, max_distance=max_pairwise_distance,
                                                    suppressed_links=_obj_suppressed_links,
                                                    verbose=verbose)
                if self.verbose and res:
                    print("pairwise collision found", name, obj.name)
            if res:
                return True
        else:
            # If we finish successfully, just return False
            return False


    def get_object(self, name):
        return self.object_interfaces[name]

    def get_all_objects(self):
        return self.object_interfaces.items()

    def load_obj(self, name, model, pkg=None, physical=True, scale=1., mass=0.1,
                 padding=0.):
        """Add model to the scene from obj file"""

        if pkg is not None:
            pb.setAdditionalSearchPath(pkg)
        if mass <= 0:
            raise RuntimeError('mass does not make sense: ' + str(mass))
        if scale <= 0:
            raise RuntimeError('scale does not make sense: ' + str(scale))
    
        #oid = create_obj(model, mass=mass, client=self.client, scale=scale)
        scale = scale * np.ones(3)
        cid = pb.createCollisionShape(pb.GEOM_MESH, fileName=model, meshScale=scale,
                                         physicsClientId=self.client)
        vid = pb.createVisualShape(pb.GEOM_MESH, fileName=model, meshScale=scale,
                                      physicsClientId=self.client)
        oid = pb.createMultiBody(baseMass=mass,
                                baseCollisionShapeIndex=cid,
                                baseVisualShapeIndex=vid,
                                physicsClientId=self.client)
        self.name_to_id[name] = oid
        self.oids.append((name, oid))

        self.allowed_collisions[name] = set()
        self.all_objects.add(name)

        obj = ArticulatedObject(name, oid, self,
                                physical=physical,
                                verbose=self.verbose,
                                padding=padding)
        self.object_interfaces[name] = obj
        return obj

    def load_primitive(self, name, extent, shape="box", physical=True, mass=0.1,
            rgba=[1,1,1,1], padding=0.):
        """ Create primitive objects """

        if shape == "box":
            assert len(extent) == 3
            oid = self.make_box(half_extents=[0.5 * e for e in extent],
                                rgba=rgba, mass=mass)
        elif shape == "cylinder":
            assert len(extent) == 2
            height = extent[0]
            radius = extent[1]
            oid = self.make_cylinder(height, radius, rgba=rgba, mass=mass)
        self.name_to_id[name] = oid
        self.oids.append((name, oid))

        self.allowed_collisions[name] = set()
        self.all_objects.add(name)

        obj = ArticulatedObject(name, oid, self,
                                physical=physical,
                                verbose=self.verbose,
                                padding=padding)
        self.object_interfaces[name] = obj
        return obj

    def load_urdf(self, name, model, pkg=None, physical=True, fixed_base=False,
            padding=0., base_position=[0,0,0], base_orientation=[0,0,0,1]):
        """ Add model to the scene """

        if pkg is not None:
            pb.setAdditionalSearchPath(pkg, physicsClientId=self.client)
    
        oid = pb.loadURDF(model, useFixedBase=fixed_base, physicsClientId=self.client, basePosition=base_position, baseOrientation=base_orientation)
        self.oids.append((name, oid))
        self.name_to_id[name] = oid

        self.allowed_collisions[name] = set()
        self.all_objects.add(name)

        obj = ArticulatedObject(name, oid, self,
                                physical=physical,
                                verbose=self.verbose,
                                padding=padding)
        self.object_interfaces[name] = obj
        return obj

    def update(self, world_state):
        """
        Move objects around according to the file
        """

        for name, obs in world_state.entities.items():
            if name in self.name_to_id:
                entity = self.get_object(name)
                if not obs.observed:
                    entity.set_pose([0, 0, 1000], [0, 0, 0, 1], wxyz=False)
                    continue
                elif obs.pose is None:
                    logwarn("Pose for " + str(name) + " was missing!")
                    continue
                else:
                    if obs.pose is not None:
                        xyz = obs.pose[:3, axis.POS]
                        xyzw = tra.quaternion_from_matrix(obs.pose)
                        entity.set_pose(xyz, xyzw, wxyz=False)
                    if obs.q is not None:
                        entity.set_joint_positions(obs.q)

            else: # obs.ref is not None and obs.ref.entity_type == entity.JOINT:
                # if is a surface...
                obs.apply()

    def add_surface(self, obj_name, surface):
        oid = self.make_box(half_extents=[0.5 * e for e in surface.extent],
                            rgba=surface.rgba)
        name = obj_name + "_surface_" + surface.name
        if name in self.name_to_id:
            raise RuntimeError('name clash: ' + str(name))
        self.name_to_id[name] = oid

        self.allowed_collisions[name] = set()
        self.allowed_collisions[name].add(obj_name)

        return oid

    def make_box(self, half_extents, rgba, mass=0):
        cid = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
        vid = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=self.client)
        return pb.createMultiBody(mass, cid, vid, physicsClientId=self.client)

    def make_cylinder(self, height, radius, rgba, mass=0):
        cid = pb.createCollisionShape(pb.GEOM_CYLINDER, height=height, radius=radius,
                physicsClientId=self.client)
        vid = pb.createVisualShape(
            pb.GEOM_CYLINDER, length=height, radius=radius, rgbaColor=rgba,
            physicsClientId=self.client)
        return pb.createMultiBody(mass, cid, vid, physicsClientId=self.client)

    def colliding(self, obj1, obj2, test_distance=0):
        oid1 = self.name_to_id[obj1]
        oid2 = self.name_to_id[obj2]
        pts = pb.getClosestPoints(bodyA=oid1, bodyB=oid2,
                                  distance=test_distance,
                                  physicsClientId=self.client)
        return len(pts) > 0

    def update_collisions(self, test_distance=0):
        collisions = {}
        for n1, oid1 in self.name_to_id.items():
            collisions[n1] = set()
            for n2 in self.all_objects:
                oid2 = self.name_to_id[n2]
                if n1 == n2:
                    continue
                if n2 in self.allowed_collisions[n1]:
                    continue

                pts = pb.getClosestPoints(bodyA=oid1, bodyB=oid2,
                                          distance=test_distance,
                                          physicsClientId=self.client)
                if len(pts) > 0:
                    collisions[n1].add(n2)

        return collisions

    def set_joints(self, name, q):
        object_id = self.name_to_id[name]
        for i in len(q):
            pb.resetJointState(object_id, i, q[i], physicsClientId=self.client)

    def set_pose_by_id(self, object_id, pos, quat):
        """Set pose in the scene, by integer ID instead of name. Not for general use."""
        pb.resetBasePositionAndOrientation(object_id, pos, quat, physicsClientId=self.client)


    def replay_trajectory(self, trajectory, actor="robot", timestep=0.05):
        print("--------- DISPLAYING TRAJECTORY ----------")
        robot = self.get_object(actor)
        for i, q in enumerate(trajectory):
            print(i, "=", q)
            robot.set_joint_positions(q)
            time.sleep(timestep)
        print("------------------------------------------")

    def animate_plan(self, world_state, plan, iterations=100, dt=0.1):
        """ For debugging -- call this to show a task and motion plan we've created and
        instantiated. This will show all the predicted motions and how the arm will move around and
        what it will do... """
        
        for _ in iterations:
            self.update(world_state)

            # Animate a whole plan
            for action, params in plan:
                print("=======>", action)
                actor = world_state[params.actor].ref
                goal = world_state[params.actor].ref
                if params.trajectory is not None:
                    for t, pt in params.trajectory:
                        print(t)
                        time.sleep(dt)
                time.sleep(dt)


    def test_config(self, obj_name, obj_pose, ignored=[], restore=True, test_distance=0.):
        """
        Pass in a set of names, update them in the world.

        obj_name: unique string name of object
        obj_pose: unique 6dof pose of object
        ignored: list of objects it's ok to hit (usually table surfaces)
        restore: should objects be reset after collisions
        test_distance: only checkpoints closer than this distance
        """
        oid = self.name_to_id[obj_name]
        orig = self.id_to_pose[oid]
        xyz = obj_pose[:3, axis.POS]
        xyzw = tra.quaternion_from_matrix(obj_pose)
        pb.resetBasePositionAndOrientation(oid, xyz, xyzw, physicsClientId=self.client)

        ignored_ids = set()
        ignored_ids.add(oid)
        for name in ignored:
            ignored_ids.add(self.name_to_id[name])

        violations = []

        # Test for collisions
        for name2, oid2 in self.oids:
            if oid2 in ignored_ids:
                continue
            pts = pb.getClosestPoints(bodyA=oid, bodyB=oid2,
                                      distance=test_distance,
                                      physicsClientId=self.client)
            if len(pts) > 0:
                violations.append((name2, pts))
        
        if len(violations) > 0:
            logwarn("Violations found: " + str(violations))
        
        # TODO remove this
        # print("PRESS ENTER"); raw_input()
        if restore:
            pb.resetBasePositionAndOrientation(oid, orig[0], orig[1], physicsClientId=self.client)

        return violations

    def disable_surfaces(self, send_away=False, verbose=False):
        for name, obj in self.object_interfaces.items():
            group = 0 # other objects dont collide with me
            mask = 0 # don't collide with other objects
            if verbose:
                print("Checking name =", name)
            if not obj.is_physical:
                if verbose:
                    print("\t ... was not a physicial object.")
                pb.setCollisionFilterGroupMask(obj.id, -1, group, mask)
                if send_away:
                    pb.resetBasePositionAndOrientation(obj.id, [0, 0, 1000], [0, 0, 0, 1],
                                                       physicsClientId=self.client)
            for sname, surface in obj.surfaces.items():
                if verbose:
                    print("\t ... disabling surface =", sname)
                pb.setCollisionFilterGroupMask(surface.id, -1, group, mask)
                if send_away:
                    pb.resetBasePositionAndOrientation(surface.id, [0, 0, 1000], [0, 0, 0, 1],
                                                       physicsClientId=self.client)

    def enable_surfaces(self):
        for name, obj in self.object_interfaces.items():
            group = 1 # other objects collide with me
            mask = 1 # collide with other objects
            if not obj.is_physical:
                pb.setCollisionFilterGroupMask(obj.id, -1, group, mask)
            for sname, surface in obj.surfaces.items():
                pb.setCollisionFilterGroupMask(surface.id, -1, group, mask)

    def spin(self, **kwargs):
        """ Run physics forward and see what happens """
        self.simulate(steps=None, **kwargs)

    def simulate(self, steps, verbose=True, freeze_robots=False):
        """ run some number of simulations to check physics or something like
        that in order to check object placements and so on """
        self.disable_surfaces()
        i = 0
        # TODO 
        if freeze_robots:
            # Send motion commands to robots to keep them from doing anything
            # dumb and moving/flopping around
            for name, obj in self.object_interfaces.items():
                if not obj.mobile:
                    continue
                if verbose:
                    print(name, obj.dof)
        if steps is None:
            done = lambda x: False
        else:
            done = lambda x: x >= steps
        while not done(i):
            pb.stepSimulation()
            time.sleep(1./240.)
            i += 1
        # TODO make this work
        if freeze_robots:
            # Unfreeze the robots 
            for name, obj in self.object_interfaces.items():
                if not obj.mobile:
                    continue
                obj.reset_pose()

    def set_camera(self, pose, matrix=False):
        if matrix:
            pose = pose[:3, 3], tra.quaternion_from_matrix(pose)
        set_camera_pose2(pose, distance=2, client=self.client)
        self.camera_pose = pose

    def add_camera(self, pose, matrix=True, preset=None, **params):
        if preset is not None:
            h, w, fov = get_camera_preset(preset)
            self.cameras.append(CameraReference(pose, pose_is_matrix=matrix,
                                client=self.client,
                                img_height=h,
                                img_width=w,
                                proj_fov=fov,
                                **params))
        else:
            self.cameras.append(CameraReference(pose, pose_is_matrix=matrix,
                                client=self.client,
                                **params))

    def capture(self):
        self.disable_surfaces(send_away=True)
        return [camera.capture() for camera in self.cameras]

    def get_projection_matrices(self):
        return [camera.proj_matrix for camera in self.cameras]

    def get_view_matrices(self):
        return [camera.view_matrix for camera in self.cameras]

    def get_camera_params(self):
        return [(camera.proj_near, camera.proj_far, camera.proj_fov) for camera in self.cameras]

    def disconnect(self):
        pb.disconnect(self.client)

    def __del__(self):
        pb.disconnect(self.client)

    def simulate_connection(self, domain, init_world_state, action, subgoal, connection, writer=None, verbose=False):
        """ Visualize a connection. """
        if verbose:
            print()
            print("==================")
            print(action.opname)
            print()
        self.update(init_world_state)
        ws = init_world_state.fork()  # Create a copy of the world state

        # predict the results after executing the action via the world state model
        domain.apply_action(init_world_state, ws, subgoal)
        domain.update_logical(ws)
        self.update(ws)
        return ws
