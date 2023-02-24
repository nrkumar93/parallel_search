# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import sys
import yaml
from brain2.flat_world.interface import FlatWorld
from brain2.motion_planners.rrt import Tree, Node

import numpy as np

def simplify_tree(env, tree, path=None):
    data = {}
    if path is not None:
        data["path"] = {}
        data["path"]["x"] = [float(pt[0]) for pt in path]
        data["path"]["y"] = [float(pt[1]) for pt in path]

    data["nodes"] = []
    #data["nodes"]["x"] = []
    #data["nodes"]["y"] = []
    #data["nodes"]["parent_idx"] = []
    for node in tree.nodes:
        if node.parent is None:
            parent_idx = -1
        else:
            parent_idx = tree.nodes.index(node.parent)
            if parent_idx is None:
                parent_idx = -1
        x = node.q[0]
        y = node.q[1]
        print(x, y, parent_idx)
        data["nodes"].append([float(x), float(y), int(parent_idx)])

    data["goal"] = {}
    data["goal"]["x"] = float(env.q_goal[0])
    data["goal"]["y"] = float(env.q_goal[1])

    data["obstacles"] = []
    for obs in env.obstacles:
        data["obstacles"].append([float(v) for v in obs])   

    return data


def write_tree_to_file(filename, env, tree, path=None):
    data = simplify_tree(env, tree, path)
    with open(filename, 'w') as f:
        res = yaml.dump(data, f)
    

def load_from_file(filename):
    """Load the same format back out from files."""
    env = FlatWorld()
    pt0 = [0, 0]
    path = None
    rrt = None
    print(filename)
    with open(filename, 'r') as f:
        data = yaml.load(f)
        if "obstacles" in data:
            for obs in data["obstacles"]:
                env.add_obstacle(*list(obs))
        if "path" in data:
            x = data["path"]["x"]
            y = data["path"]["y"]
            path = np.zeros((len(x), 2))
            for i, (xx, yy) in enumerate(zip(x, y)):
                path[i, 0] = xx
                path[i, 1] = yy
        # We always have a tree.
        env.q_goal = [float(data["goal"]["x"]), float(data["goal"]["y"])]
        for i, node in enumerate(data["nodes"]):
            pt = np.array(node[:2])
            idx = int(node[2])
            if i == 0:
                rrt = Tree(None, pt0)
            elif idx < 0:
                rrt.nodes.append(Node(pt, None))
            else:
                rrt.nodes.append(Node(pt, rrt.nodes[idx]))

    return env, rrt, path

if __name__ == '__main__':
    filename = sys.argv[1]
    env, tree, path = load_from_file(filename)
    env.show_path(path, tree)
