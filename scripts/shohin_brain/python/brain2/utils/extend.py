# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np


def simple_extend(q1, q2, step_size=2.5):
    """Connect q1 and q2; assume both are numpy arrays."""
    dq = q2 - q1
    dist = np.linalg.norm(q2 - q1)
    direction = dq / dist
    step = direction * step_size
    while True:
        dist = np.linalg.norm(q2 - q1)
        if dist < step_size:
            yield q2
            break
        else:
            q1 = np.copy(q1)
            q1 += step
            yield q1
    return 

def simple_extend_once(q1, q2, step_size=2.5):
    """Connect q1 and q2; assume both are numpy arrays."""
    dq = q2 - q1
    dist = np.linalg.norm(q2 - q1)
    direction = dq / dist
    step = direction * step_size

    dist = np.linalg.norm(q2 - q1)
    if dist < step_size:
        yield q2
    else:
        q1 = np.copy(q1)
        q1 += step
        yield q1

    return 

def scaled_extend(q1, q2, robot_ref, step_sizes):
    # Scale extension according to robot joint radii
    dq = q2 - q1
    dist = np.linalg.norm(q2 - q1)
    direction = dq / dist
    step = np.abs((ref.active_max - ref.active_min) * step_size) * direction
    while True:
        dist = np.linalg.norm(q2 - q1)
        if dist < step_size:
            yield q2
            break
        else:
            q1 = np.copy(q1)
            q1 += step
            yield q1
    return 


