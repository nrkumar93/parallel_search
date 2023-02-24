# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
cmake_minimum_required(VERSION 2.8.3)
project(brain2_msgs)
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  trajectory_msgs
)
add_message_files(
  FILES
  Predicate.msg
  Goal.msg
  Grasp.msg
  PredicateList.msg
  Attach.msg
  TrackedBody.msg
  HandStatus.msg
)
add_service_files(
  FILES
  GetPath.srv
)
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  trajectory_msgs
)
catkin_package()
include_directories()
