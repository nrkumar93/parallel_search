# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import rospy
from brain2_ros.lula_franka_control import LulaFrankaControl
from brain2_ros.franka_gripper_control import FrankaGripperControl

rospy.init_node('relay_commands_to_franka_node')

# Create interfaces with real robot
gripper = FrankaGripperControl()
arm = LulaFrankaControl(base_link="base_link",
                           ee_link="right_gripper",
                           dof=7,
                           sim=False)
gripper.open()
gripper.close()
gripper.open()

gripper_open = True

import threading
lock = threading.Lock()

def _js_cb(msg):
    global gripper_open
    gripper_open = msg.position[7] > 0.03

def _cb(msg):
    """ Command gripper to  """
    with lock:
        global gripper_open
        global lock
        print(msg.position)
        if msg.position is not None and len(msg.position) >= 7:
            q = msg.position[:7]
            gripper_cmd = msg.position[7] > 0.01
        else:
            q = None
            gripper_cmd = gripper_open

        if q is not None:
            arm.go_local(q=q)
        if gripper_cmd and not gripper_open:
            gripper.open(wait=True)
            #gripper_open = True
        elif not gripper_cmd and gripper_open:
            gripper.close(wait=True)
            #gripper_open = False
        rospy.sleep(0.1)

from sensor_msgs.msg import JointState
cmd_sub = rospy.Subscriber('brain/joint_cmd', JointState, _cb, queue_size=100)
js_sub = rospy.Subscriber('joint_states', JointState, _js_cb, queue_size=1000)

r = rospy.Rate(10)
while not rospy.is_shutdown():
    r.sleep()
