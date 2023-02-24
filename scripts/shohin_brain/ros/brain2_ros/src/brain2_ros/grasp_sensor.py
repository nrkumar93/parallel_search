# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from brain2.utils.pose import make_pose
from brain2.utils.info import logwarn, logerr
import brain2.utils.status as status
from utils import make_pose_from_unstamped_pose_msg

from brain2.utils.pose import make_pose
import brain2.utils.transformations as tra
import numpy as np
import tf2_ros
import tf
import threading


class GraspSensor(object):
    """
    Listens to MarkerArray status messages
    These should be lists of grasp poses
    """

    root_frame = "world"
    #hand_to_grasp = make_pose((0, 0, 0.1), (0, 0, 1., 0))
    # TODO: Wei can tune this if he wants to
    _base_hand_to_grasp = make_pose((0, 0, 0.15), (0, 0, 1., 0))
    _apply_rotation = make_pose((0, 0, 0), (0, 0, -0.707, 0.707))

    def _cb(self, msg):
        """
        Store information
        """
        self._msg = msg
        with self._lock:
            self.poses = []
            self.scores = []
            for marker in self._msg.markers:
                self.obs_t = marker.header.stamp.to_sec()
                frame = marker.header.frame_id
                T_cam = self.observer.get_relative_tf(self.root_frame, frame)
                T_grasp = make_pose_from_unstamped_pose_msg(marker.pose)
                pose = T_cam.dot(T_grasp).dot(self.hand_to_grasp)
                pose2 = pose.dot(tra.euler_matrix(0, 0, np.pi))
                # score = marker.color.g - marker.color.r
                score = marker.color.a
                self.scores.append(score)
                self.poses.append(pose)
                self.scores.append(score)
                self.poses.append(pose2)

    def _obj_cb(self, marker):
        with self._obj_lock:
            self.obj_obs_t = marker.header.stamp.to_sec()
            frame = marker.header.frame_id
            T_cam = self.observer.get_relative_tf(self.root_frame, frame)
            T_grasp = make_pose_from_unstamped_pose_msg(marker.pose)
            pose = T_cam.dot(T_grasp)
            self.obj_pose = pose

    def print_info(self):
        print("====", self._obj, "====")
        print("Observed:", self.obs_t)
        print("# Poses:", len(self.poses))
        print("=====" + ("=" * len(self._obj)) + "=====")

    def __init__(self, observer, obj="obj", topic="grasps_sampling_left",
                 obj_topic="object_crop_center",
                 grasps=None, max_obs_age=0.5):
        """
        observer: the observer object managing world state
        rostopic: the topic messages will be arriving on
        """
        self.observer = observer
        self.obs_t = rospy.Time(0).to_sec()
        self.obj_obs_t = rospy.Time(0).to_sec()
        self.poses = []
        self.scores = []
        self.max_obs_age = max_obs_age
        self.hand_to_grasp = self._base_hand_to_grasp.dot(self._apply_rotation)
        self.obj_pose = None

        # For debugging purposes
        self._tf_broadcaster = observer.tf_broadcaster
        self._tf_listener = observer.tf_listener

        # Track supported hands
        self._obj = obj
        self._grasps_topic = topic
        self._obj_topic = obj_topic

        if grasps is None:
            raise NotImplementedError('must provide grasps sampler obj')
        self.grasps = grasps
        self._lock = threading.Lock()
        self._obj_lock = threading.Lock()

        # Initialize parameters
        self._hand_msg_sub = rospy.Subscriber(self._grasps_topic,
                                              MarkerArray,
                                              self._cb,
                                              queue_size=1)
        self._obj_msg_sub = rospy.Subscriber(self._obj_topic,
                                              Marker,
                                              self._obj_cb,
                                              queue_size=1)

    def has(self, obj):
        """ Each sensor only tracks a single hand """
        return obj == self._obj

    def _get_poses(self):
        """ Convert the poses over to quaternions for now """
        pq = np.zeros((len(self.poses), 7))
        for i, pose in enumerate(self.poses):
            pq[i][:3] = pose[:3, 3]
            pq[i][3:] = tra.quaternion_from_matrix(pose)
        return pq

    def update(self, entity, t):
        """ update the object with important information about what it is """
        if t - self.obs_t < -1.0:
            raise RuntimeError('You have a timing issue! cannot observe into'
                    ' the past: ' + str(t) + ", " + str(self.obs_t)
                    + "diff =" + str(t - self.obs_t))
        with self._lock:
            obs_t = self.obs_t
            if obs_t < t - self.max_obs_age:
                self.grasps.update_grasps_for_obj(self._obj, [], t, scores=[])
                obs_t = t
            else:
                self.grasps.update_grasps_for_obj(self._obj, self._get_poses(), t,
                                                  scores=self.scores)
        has_obj = False
        with self._obj_lock:
            if self.obj_pose is not None:
                has_obj = True
                obj_pose = np.copy(self.obj_pose)
            obj_t = self.obj_obs_t
        if has_obj:
            res = entity._update(base_pose=obj_pose,
                                 obs_t=min(obs_t, obj_t),
                                 t=t,
                                 max_obs_age=self.max_obs_age)
        else:
            res = False
        return status.SUCCESS if res else status.FAILED
