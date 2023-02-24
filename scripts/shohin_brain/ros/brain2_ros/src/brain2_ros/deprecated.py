# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.msg import DisplayRobotState
from moveit_msgs.msg import RobotTrajectory


def publish_display_trajectory(robot_state, joint_trajectory, frame="base_link"):
    display_trajectory_pub = rospy.Publisher('/display_planned_path', DisplayTrajectory, queue_size=1)

    display_trajectory = DisplayTrajectory()
    #display_trajectory.model_id = 'pr2'
    # display_trajectory.trajectory_start = moveit.robot_state.get_current_state()
    display_trajectory.trajectory_start = msg_tools.makeRobotStateMsg(robot_state)
    robot_state_trajectory = RobotTrajectory(joint_trajectory=joint_trajectory)
    # TODO: get this in the base_link frame
    robot_state_trajectory.joint_trajectory.header.frame_id = frame
    display_trajectory.trajectory.append(robot_state_trajectory)
    display_trajectory_pub.publish(display_trajectory)
    # moveit.display_trajectory_publisher.publish(display_trajectory)

    robot_state_state_pub = rospy.Publisher('/display_robot_state_state', DisplayRobotState, queue_size=1)
    display_state = DisplayRobotState()
    display_state.state = display_trajectory.trajectory_start
    last_conf = joint_trajectory.points[-1].positions
    joint_state = display_state.state.joint_state
    joint_state.position = list(joint_state.position)
    for joint_name, position in zip(joint_trajectory.joint_names, last_conf):
        joint_index = joint_state.name.index(joint_name)
        joint_state.position[joint_index] = position
    robot_state_state_pub.publish(display_state)

    return display_trajectory


