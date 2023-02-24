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
import mayavi.mlab as mlab
import json
import random
from lxml import etree
import brain2.bullet.problems as problems
import brain2.bullet.ik 
import math
from threading import Lock
import copy
from visualization_utils import draw_scene
import mayavi.mlab as mlab
# import ik_solver

# internal tools
from brain2.gym.urdf import ObjectInfo, LinkInfo
from brain2.gym.urdf import parse_urdf


def plot_mesh(mesh, colormap='Blues'):
    assert type(mesh) == trimesh.base.Trimesh
    mlab.triangular_mesh (
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        mesh.faces,
        colormap=colormap
    )

class ObjectPlacementNotFound(Exception):
    pass


class SceneManager(object):
    """ Create manager to handle interface to Isaac Gym, create objects, move
    them around, etc. Based on code from Arsalan Mousavian.
    """

    def __init__(
        self, 
        robot_urdf_path,
        table_dims, 
        gravity_axis=1,
        obj_scale_range=[1.0, 1.0],
        obj_mass_range=[0.03, 0.20],
        drop_distance_from_table=0.005,
        split='train',
        seed=None,
        headless=False,
        load_grasps=False,
        ):
        assert (len(dataset_folder) != 0)
        self._object_cats = ['Bottle', 'Bowl', 'box', 'cylinder', 'mug']
        #self._dataset_root = dataset_folder 
        #self._cache_folder = cache_folder
        #self._meshes_folder = os.path.join(cache_folder, 'meshes')
        self._object_scales = np.load(os.path.join(dataset_folder, 'object_scales.npy'), allow_pickle=True).item()
        self._obj_files = {c: [] for c in self._object_cats}
        for c in self._object_cats:
            json_files = json.load(open(os.path.join(dataset_folder, 'splits', c + '.json')))[split]
            self._obj_files[c] = [(self.json_to_obj_path(f), f) for f in json_files]
        
        self.json_root = os.path.join(dataset_folder, 'grasps')
        self._table_dims = table_dims
        self._gravity_axis = gravity_axis
        self._obj_scale_range = obj_scale_range
        self._drop_distance_from_table = drop_distance_from_table
        self._obj_mass_range = obj_mass_range
        self._manager = trimesh.collision.CollisionManager()
        self.obj_info = {}
        self._robot_info = {}
        self._robot_link_dict = parse_urdf(robot_urdf_path)
        self.current_plan = None
        self.current_plan_index = 0
        #self.vhacd_decomposition = vhacd_decomposition
        self.load_grasps = load_grasps

        self._visualizer = None
        if not headless:
            self._visualizer = scene_visualizer.SceneVisualizerProcess()
            self._visualizer.start()
        
        self._load_robot_link_meshes(self._robot_link_dict)
        self._should_update_robot_state = True
        self.iface = None
        self.gym_robot_q = None
        self.robot_lock = Lock()

    @property
    def robot_link_dict(self):
        return self._robot_link_dict
    
    @property
    def visualizer(self):
        return self._visualizer

    def _load_robot_link_meshes(self, robot_link_dict):
        """ Take dict of robot links and load each one individually """
        for k in robot_link_dict:
            mesh = trimesh.load(robot_link_dict[k].obj_path)
            # Rotate the links and then remove the transform
            transform = robot_link_dict[k].transform
            mesh.vertices = mesh.vertices.dot(transform[:3, :3].T)
            mesh.vertices += np.expand_dims(transform[:3, 3], 0)
            mesh.visual.face_colors = np.ones(mesh.faces.shape, dtype=np.float32) * 0.8

            robot_link_dict[k].mesh = mesh
            robot_link_dict[k].transform = np.eye(4, dtype=np.float32)

            if self._visualizer is not None:
                self._visualizer.add_object(k, mesh)

    
if __name__ == '__main__':
    import time 

    init_rt_in_base = np.asarray([[-7.07106781e-01,  2.09060286e-16,  7.07106781e-01,
         0.2938899],
       [ 8.65956056e-17,  1.00000000e+00, -2.09060286e-16,
         0.18403221],
       [-7.07106781e-01, -8.65956056e-17, -7.07106781e-01,
         0.37154609],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
        

    # TODO 
    default_robot_path = '~/src/brain_gym/assets/urdf/franka_description/robots/franka_panda.urdf'
    default_robot_path = os.path.expanduser(default_path)
    manager = SceneManager(default_robot_path, [2, 2, 2])

    robot_link_dict = manager.robot_link_dict
    for k, v in robot_link_dict.items():
        print(k, v)
    
    TARGET_LINK = 'panda_hand'
    target_mesh = robot_link_dict[TARGET_LINK].mesh.copy()
    
    target_transform = init_rt_in_base
    manager.add_object('target_panda_hand', target_mesh, target_transform, color=(0,1.,0))


    solver = IKSolver(default_path, TARGET_LINK, robot_link_dict)
    for _ in range(100):
        # target_transform = tra.random_rotation_matrix()
        # target_transform[:3, 3] = np.random.uniform(-0.6, 0.6, size=3)
    
        ik_time = time.time()
        found, solution, err = solver.solve(target_transform, 20, 3e-3, random_init=True)
        print(time.time() - ik_time)
        manager.update_state({'robot': solution, 'target_panda_hand': target_transform})
        print('found {} residual_xyz {} residual_angle_cos {} self_collision {}'.format(found, err[0], err[1], err[2]))
        if input('continue?') == 'n':
            del manager
            exit()

    # print(found, solution['panda_hand'][:3, 3], init_rt_in_base[:3, 3])
    # print(solution['panda_hand'])
    # print(init_rt_in_base)
    # print('diff', np.abs(solution['panda_hand'] - init_rt_in_base))
    # print(np.all(np.abs(solution['panda_hand'] - init_rt_in_base) < 1e-3))
    input('press any key to end')
    del manager
