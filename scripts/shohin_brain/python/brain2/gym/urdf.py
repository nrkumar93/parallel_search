# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import trimesh
import trimesh.transformations as tra
import os
import glob
import numpy as np
import json
import random
import scene_visualizer
from lxml import etree
import math
from threading import Lock
import copy
from visualization_utils import draw_scene

class ObjectInfo:
    __slots__ = ('transform', 'name', 'mesh', 'color', 'urdf_path', 'grasps')
    def __init__(
        self,
        name,
        mesh,
        transform=np.eye(4),
        color=[0.,0.,0.],
        urdf_path=None,
        grasps=None,
    ):
        self.name = name
        self.transform = transform
        self.mesh = mesh
        self.color = np.asarray(color)
        self.urdf_path = urdf_path
        self.grasps = grasps


class LinkInfo:
    __slots__ = ('obj_path', 'transform', 'mesh')
    def __init__(
        self,
        obj_path,
        transform,
    ):
        self.obj_path = obj_path
        self.transform = transform


def parse_urdf(robot_urdf_path, robot='franka'):
    """ Load dictionary of URDF links from arsalan"""

    print('loading {}'.format(robot_urdf_path))
    parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(robot_urdf_path, parser=parser) 
    def dfs(x, depth):
        # indent = '    ' * depth
        # print(indent, x.tag)
        # if x.tag == 'link':
            # print(indent, x.values()[0])
            
        num_childs = 0
        value_dict = {}
        for c in x: 
            if depth == 0 and c.tag != 'link': 
                continue
            if depth == 1 and c.tag != 'collision' :
                continue
            num_childs += 1
            subtree = dfs(c, depth + 1) 
            for k, v in subtree.items():
                value_dict[k] = v
        
        if num_childs == 0:
            for k, v in x.items():
                value_dict[k] = v
            # print(indent, x.items())
            if x.tag == 'geometry':
                return x.values()[0]
        
        # print('{} ~{}: {}'.format(indent, x.tag, num_childs))
        
        if x.tag == 'link':
            key = x.values()[0]
        else:
            key = x.tag
        return {key: value_dict}
            
    
    traverse_dict = dfs(tree.getroot(), 0)['robot']
    output = {}
    if robot == 'franka':
        replace_string = robot_urdf_path[:robot_urdf_path.find('franka_description')]
    else:
        raise NotImplementedError('{} is not supported'.format(robot))
    for link_name, link_dict in traverse_dict.items():
        obj_path = link_dict['collision']['geometry']['mesh']['filename']
        obj_path = obj_path.replace(
            'package://', 
            replace_string,
        )

        rpy = (0, 0, 0)
        xyz = (0, 0, 0)

        if 'origin' in link_dict['collision']:
            rpy = tuple([float(t) for t in link_dict['collision']['origin']['rpy'].split(' ')])
            xyz = tuple([float(t) for t in link_dict['collision']['origin']['xyz'].split(' ')])
        transform = tra.euler_matrix(*rpy)
        transform[:3, 3] = list(xyz)
        # print(link_name, os.path.isfile(obj_path), rpy, xyz)
        output[link_name] = LinkInfo(obj_path, transform)

    return output

    
