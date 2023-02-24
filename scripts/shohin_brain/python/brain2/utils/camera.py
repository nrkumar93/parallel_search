# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import open3d

def get_camera_presets():
    return [
            "azure_nfov",
            "realsense",
            ]

def get_camera_preset(name):

    if name == "azure_depth_nfov":
        # Setting for depth camera is pretty different from RGB
        height, width, fov = 576, 640, 75
    if name == "azure_720p":
        # This is actually the 720p RGB setting
        # Used for our color camera most of the time
        height, width, fov = 720, 1280, 90
    elif name == "realsense":
        height, width, fov = 480, 640, 60
    else:
        raise RuntimeError(('camera "%s" not supported, choose from: ' +
            str(get_camera_presets())) % str(name))
    return height, width, fov


def get_matrix_of_indices(height, width):
    """ Get indices """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)


# --------------------------------------------------------
# NOTE: this code taken from Arsalan
def compute_xyz(depth_img, camera_params, visualize_xyz=False):
    """ Compute ordered point cloud from depth image and camera parameters

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used.
                              For real data, it needs these keys:
                                    - img_width
                                    - img_height
                                    - fx
                                    - fy
                                    - x_offset (optional, will default to W/2)
                                    - y_offset (optional, will default to H/2)

    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = get_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]

    if visualize_xyz:
        unordered_pc = xyz_img.reshape(-1, 3)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(unordered_pc) 
        pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]) # Transform it so it's not upside down
        open3d.visualization.draw_geometries([pcd])
        
    return xyz_img

