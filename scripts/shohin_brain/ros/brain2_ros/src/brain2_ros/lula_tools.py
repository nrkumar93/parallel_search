# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

from lula_pyutil import util
from lula_pyutil.math_util import (
    pack_transform, numpy_vec, numpy_quat)
import numpy as np
import pickle

import brain.axis as axis


def lula_go_local(ee, T, high_precision=True, wait_for_target=False):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, axis.POS]
    axis_x = T[:3, axis.X]
    axis_z = T[:3, axis.Z]
    ee.go_local(
        orig=orig,
        axis_x=axis_x,
        axis_z=axis_z,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=wait_for_target)


def lula_go_local_y_axis(ee, T, high_precision=True):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, axis.POS]
    axis_y = T[:3, axis.Y]
    # axis_z = T[:3,axis.Z]
    ee.go_local(
        orig=orig,
        axis_y=axis_y,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=False)


def lula_go_local_no_orientation(ee, T, high_precision=True):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, 3]
    ee.go_local(
        orig=orig,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=False)


def lula_go_local_async(ee, T, high_precision=True):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, axis.POS]
    axis_x = T[:3, axis.X]
    axis_z = T[:3, axis.Z]
    ee.async_go_local(
        orig=orig,
        axis_x=axis_x,
        axis_z=axis_z,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=False)


def lula_go_local_no_orientation_async(ee, T, high_precision=True):
    """
    Convert between transform and the lula format. Specify goal.
    """
    orig = T[:3, 3]
    ee.go_local(
        orig=orig,
        use_target_weight_override=high_precision,
        use_default_config=False,
        wait_for_target=False)


def load_recorded_frame(rospath, offset=[0., 0., 0.]):
    offset = np.array(offset)
    path = util.parse_pkg_name(rospath)
    relative_transform_stamped, config = pickle.load(open(path, 'rb'))
    relative_T = pack_transform(
        numpy_vec(relative_transform_stamped.transform.translation),
        numpy_quat(relative_transform_stamped.transform.rotation))
    relative_T[0:3, 3] += offset
    rel_to = relative_transform_stamped.header.frame_id

    return relative_T, config, rel_to
