# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
This file holds poses related to the kitchen domain
"""

import brain2.utils.pose as pose
import numpy as np

# Drawer -- indigo
# Angled approach
# drawer_to_approach_handle = ([0.420, 0.054, 0.111], [0.741, -0.610, -0.210, 0.188])
# drawer_to_cage_handle = ([0.332, 0.049, 0.038], [0.741, -0.610, -0.210, 0.188])
# Straight on to handle
to_handle_q1 = [0.533, -0.479, -0.501, 0.485]
to_handle_q2 = [0.485, 0.540, -0.480, -0.493]
drawer_to_approach_handle = {
    "indigo_drawer": [
        pose.make_pose([0.40, 0.0, 0.0], to_handle_q1),
    ]
}
drawer_to_cage_handle = {
    "indigo_drawer": [
        pose.make_pose([0.28, 0.0, 0.0], to_handle_q1),
    ]
}
drawer_to_release_handle = {
    "indigo_drawer": [
        pose.make_pose([0.38, 0.0, 0.08], to_handle_q1),
        # pose.make_pose([0.38, 0.0, 0.08], to_handle_q2),
    ]
}
drawer_to_push_handle = {
    "indigo_drawer": [
        [pose.make_pose([0.247, 0.0, 0.0], to_handle_q1)],
    ]
}

# - Translation: [0.578, -0.033, -0.210]
# - Rotation: in Quaternion [0.519, -0.494, -0.445, 0.536]
# TOP
# - Translation: [0.603, -0.037, -0.079]
# - Rotation: in Quaternion [0.529, -0.484, -0.502, 0.484]
# BTM
# - Translation: [0.597, -0.034, -0.210]
# - Rotation: in Quaternion [0.521, -0.492, -0.444, 0.539]
cabinet_to_open = {
    "indigo_drawer_top": pose.make_pose([0.578, -0.033, -0.079],
                                        [0.519, -0.494, -0.445, 0.536]),
    "indigo_drawer_bottom": pose.make_pose([0.578, -0.033, -0.210],
                                           [0.519, -0.494, -0.445, 0.536]),
}

# Mustard
mustard_to_approach1 = ([-0.093, 0.017, 0.012], [0.368, 0.565, 0.646, 0.357])
mustard_to_cage1 = ([-0.023, 0.046, -0.002], [0.531, 0.455, 0.540, 0.468])

# ----------------------------
# List of frames for use when computing if we need to backoff from something
interaction_frames = {
    "indigo_drawer": drawer_to_cage_handle["indigo_drawer"][0],
    # "indigo_drawer_top": drawer_to_cage_handle["indigo_drawer"],
    # "indigo_drawer_bottom": drawer_to_cage_handle["indigo_drawer"],
}

backoff_step = pose.make_pose((0, 0, -0.1), (0, 0, 0, 1))

# Potted Meat
spam_to_approach_negx = ([0.03, 0.0, 0.12], [0.033, 0.997, -0.016, -0.065])
spam_to_cage_negx = ([0.0, 0.0, 0.015], [0.033, 0.997, -0.015, -0.065])
spam_approach_pose_negx = pose.make_pose(*spam_to_approach_negx)
spam_cage_pose_negx = pose.make_pose(*spam_to_cage_negx)

top_to_parent = 0.084
open_to_parent = -0.30

# Potted meat can
spam_q1 = [0.979, 0.192, 0.048, -0.050]
spam_q2 = [0.994, 0.048, -0.095, -0.014]
spam_q3 = [0.035, 0.999, 0.022, -0.022]
spam_approach_front_pos = [0.03, 0.0, 0.12]
spam_approach_back_pos = [-0.03, 0.0, 0.12]
spam_grasp_pos = [0.0, 0.0, 0.026]
# WORKED WITH LULA
# spam_over_pos = [0.455, 0.074, 0.263]
# spam_place_pos = [0.13, -0.12, 0.05]
spam_over_pos = [0.455, -0.12, 0.28]
spam_place_pos = [0.15, -0.12, 0.05]

# Sugar Box
sugar_q1 = [0.695, -0.632, -0.263, -0.218]
sugar_q2 = [0.669, 0.662, 0.244, -0.236]
sugar_approach_front_pos = [0.0, -0.09, 0.13]
sugar_approach_back_pos = [0.0, 0.09, 0.13]
sugar_over_pos = [0.50, -0.2, 0.33]
sugar_place_pos = [0.18, 0.09, 0.140]
sugar_place_rot1 = [-0.175, 0.834, 0.242, -0.465]
sugar_place_rot2 = [0.837, 0.173, -0.464, -0.235]

sugar_approach_top_q = [-0.671, 0.741, -0.005, -0.017]
sugar_approach_top_pos = [-0.001, -0.001, 0.179]

sugar_cage_top_q = [-0.671, 0.741, -0.005, -0.017]
sugar_cage_top_pos = [-0.001, -0.001, 0.064]

# [0.993, 0.065, -0.097, -0.017]
sugar_over_top_q = [0.979, 0.192, 0.048, -0.051]
sugar_over_top_pos = [0.6, 0.2, 0.4]  # [-0.339, -0.130, 0.380]

sugar_place_top_q = [-0.954, 0.020, -0.092, 0.286]
sugar_place_top_pos = [0.11, 0.2, 0.1]  # [-0.395, 0.154, 0.190]


# ------------------
# Tomato soup
# - Translation: [0.003, -0.008, 0.014]
# - Rotation: in Quaternion [0.658, -0.632, -0.292, -0.285]
soup_q1 = [0.658, -0.632, -0.292, -0.285]
soup_approach_pos = [0.003, -0.02, 0.04]
soup_cage_pos = [0.003, -0.008, 0.014]
soup_over_pos = [0.16 - open_to_parent, 0.18, 0.125 - top_to_parent]
soup_place_pos = [0.16, 0.26, 0.07]
soup_place_rot = [0.739, 0.654, -0.107, -0.116]

ycb_approach = {
    "spam": [
        pose.make_pose(spam_approach_front_pos, spam_q3),
        pose.make_pose(spam_approach_front_pos, spam_q2),
        pose.make_pose(spam_approach_back_pos, spam_q3),
        pose.make_pose(spam_approach_back_pos, spam_q2),
    ],
    "sugar": [
        # pose.make_pose(sugar_approach_front_pos, sugar_q1),
        pose.make_pose(sugar_approach_top_pos, sugar_approach_top_q),
        # TODO enable this
        # pose.make_pose(sugar_approach_front_pos, sugar_q2),
    ],
    "tomato_soup": pose.make_pose(soup_approach_pos, soup_q1),
}
ycb_cage = {
    "spam": [
        pose.make_pose(spam_grasp_pos, [0.035, 0.999, 0.022, -0.022]),
        pose.make_pose([0.0, 0.0, 0.015], [0.994, 0.048, -0.095, -0.014]),
    ],
    "sugar": [
        pose.make_pose(sugar_cage_top_pos, sugar_cage_top_q),
        pose.make_pose([0.0, -0.03, 0.064], sugar_q1),
        pose.make_pose([0.0, -0.03, 0.064], sugar_q2),
        pose.make_pose([0.0, 0.03, 0.064], sugar_q1),
        pose.make_pose([0.0, 0.03, 0.064], sugar_q2),
    ],
    "tomato_soup": [pose.make_pose(soup_cage_pos, soup_q1)],
}

supported_ycb_objects = ["spam", "sugar", "tomato_soup",
                         "mustard_bottle", "cracker"]

ycb_over_drawer = {
    "spam": [
        pose.make_pose(spam_over_pos, [0.994, 0.048, -0.095, -0.014]),
        pose.make_pose(spam_over_pos, [0.979, 0.192, 0.048, -0.051]),
    ],
    "sugar": [
        # pose.make_pose(sugar_over_pos, sugar_place_rot1),
        # pose.make_pose(sugar_over_pos, sugar_place_rot2),
        pose.make_pose(sugar_over_top_pos, sugar_over_top_q),
    ],
    "tomato_soup": [
        pose.make_pose(soup_over_pos, soup_place_rot)
    ],
}

ycb_place = {
    "spam": [
        pose.make_pose(spam_place_pos, spam_q1),
        pose.make_pose(spam_place_pos, spam_q2),
    ],
    "sugar": [
        # pose.make_pose(sugar_place_pos, sugar_place_rot1),
        # pose.make_pose(sugar_place_pos, sugar_place_rot2),
        pose.make_pose(sugar_place_top_pos, sugar_place_top_q),
    ],
    "tomato_soup": [
        pose.make_pose(soup_place_pos, soup_place_rot)
    ],
}

ycb_place_in_drawer_q = np.array([-0.28879132866859436, -0.8507246375083923,
                                  -0.2146039754152298, -2.6736812591552734,
                                  0.08498863875865936, 3.3035223484039307,
                                  0.7772023677825928])
open_drawer_q = np.array([-1.6299439694612394, 1.6920185401816117,
                          0.9416284438649645, -1.7287199226126881, 0.9634530876179042,
                          2.537315251694788, -0.541907630907302, ])
