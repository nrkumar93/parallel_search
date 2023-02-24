# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

# NVIDIA grasps dataset and associated objects
import numpy as np
import os
from tqdm import tqdm
import pickle
import random
import gzip
import os
import json
import pandas as pd
import open3d

from brain2.datasets.shapenet import ShapenetModelDataset
from brain2.utils.pose import make_pose
import brain2.utils.transformations as tra

class Grasps(object):
    """ Class containing grasps and semantic information for a particular
    object. """
    def __init__(self, category, json_file, md=None, cd=None, root="."):
        self.category = category
        self.filename = json_file
        self.lang = []

        hand_to_grasp = make_pose((0, 0, 0.1), (0, 0, 1., 0))
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.grasps = np.array(data['poses'])
                self.obj = data['object'].split('/')[-1].split('.')[0]
                self.key = self.obj
                self.mesh_filename = data['object']
                self.scale = data['object_scale']
                self.mass = data['object_mass']
                self.mesh_filename = os.path.join(root, data["object"])
                
                # Put all our grasps in the panda grasp frame
                for i in range(self.grasps.shape[0]):
                    grasp = tra.quaternion_matrix(self.grasps[i][3:])
                    grasp[:3, 3] = self.grasps[i][:3]
                    grasp = grasp.dot(hand_to_grasp)
                    self.grasps[i][:3] = grasp[:3, 3]
                    self.grasps[i][3:] = tra.quaternion_from_matrix(grasp)
                self.loaded = True
        except Exception as e:
            print(">>>> COULD NOT PARSE JSON:", json_file)
            print(">>>> REASON:", str(e))
            self.loaded = False

        if md is None or cd is None:
            return

        # get category-level data
        if cd is not None:
            row = cd.loc[cd['category'] == category]
            try:
                words = row['synset words'].iloc[0].split(',')
                self.lang += words
            except Exception as e:
                print("Failed to parse synset words:", row['synset words'])
            try:
                gloss = row['synset gloss'].iloc[0]
                self.lang.append(gloss)
            except Exception as e:
                print("Failed to parse synset gloss:", row['synset gloss'])
        if md is not None:
            key = "wss." + self.obj
            row = md.loc[md['fullId'] == key]
            try:
                words = row['wnlemmas'].iloc[0].split(',')
                self.lang += words
            except Exception as e:
                print("Failed to parse wnlemmas:", row['wnlemmas'])
            try:
                name = row['name'].iloc[0]
                self.lang.append(name)
            except Exception as e:
                print("Failed to parse name:", row['name'])


class ScaledMesh(object):
    def __init__(self, mesh, scale, mass, scaled_pts=None, cat=None):
        self.mesh = mesh
        self.scale = scale
        self.mass = mass
        self.scaled_pts = scaled_pts
        self.cat = cat

    def save(self, filename):
        data = {self.cat: self.scaled_pts}
        np.savez(filename, **data)


class GraspDataset(object):

    def load_semantics(self, semantic_dir, metadata, categories):
        """ Get shapenet semantic information """
        md = pd.read_csv(os.path.join(semantic_dir, metadata))
        cd = pd.read_csv(os.path.join(semantic_dir, categories))
        return md, cd

    def __init__(self, grasp_dir='/data/NVGraspDataset_clemens_complete', 
                 shapenet_dir='/data/ShapeNetCore.v2',
                 semantic_dir='/data/ShapeNetSemantic',
                 visuals=False,
                 meshes=True,
                 preload=False,):
        """ Create by pointing at shapenet and the other data """
        self.grasp_dir = grasp_dir
        self.preload = preload
        self._load_meshes = meshes
        self.md = None
        self.cd = None
        self.loaded = False

        # Get a list of the objects we might want to use
        grasp_dir = os.path.join(self.grasp_dir, 'grasps')
        self.all_objs = [f for f in os.listdir(grasp_dir) if f[0] != '.']
        self.obj_to_idx = {}

        # Hold the object IDs
        for i, obj in enumerate(self.all_objs):
            self.obj_to_idx[obj] = i

        # Actually store/load the grasps
        self.objs = {}
        if self.preload:
            self._load(self.grasp_dir)

        if visuals:
            raise RuntimeError('visuals - we do not support this')
            self._shapenet = ShapenetModelDataset(shapenet_dir=shapenet_dir)

    def _load(self, md=None, cd=None):
        """ Helper function actually loads and creates a whole set of objects
        that we can then read and use for other stuff. """
        grasp_dir = os.path.join(self.grasp_dir, 'grasps')
        print("[ACRONYM] Preloading object dataset information...")
        for i, obj in enumerate(tqdm(self.all_objs, ncols=50)):
            filename = os.path.join(grasp_dir, obj)
            cat = obj.split('_')[0]
            self.objs[i] = Grasps(cat, filename, md, cd, self.grasp_dir)
        self.loaded = True

    def load_meshes(self, visualize=False, verbose=False):
        """ load all the meshes to memory so we can use them for stuff I guess """
        meshes = {}
        for i, obj in self.objs.items():
            if verbose:
                print(i, obj)
            meshes[obj.key] = self.load_mesh(i, visualize)
        return meshes

    def get_grasps_from_file(self, filename, md=None, cd=None):
        grasp_dir = os.path.join(self.grasp_dir, 'grasps')
        if filename not in self.obj_to_idx:
            raise RuntimeError('file not recognized: ' + str(filename))
        idx = self.obj_to_idx[filename]
        cat = filename.split('_')[0]
        filename = os.path.join(grasp_dir, filename)
        self.objs[idx] = Grasps(cat, filename, md, cd, self.grasp_dir)
        if not self.objs[idx].loaded:
            raise RuntimeError('failed to load grasps!')
        return self.objs[idx]
        
    def get_mesh_info(self, idx=None, filename=None):
        """ Return only one mesh """
        if filename is not None:
            idx = self.obj_to_idx[filename]
        elif idx is None:
            idx = np.random.randint(len(self.all_objs))
        if idx not in self.objs:
            grasp_dir = os.path.join(self.grasp_dir, 'grasps')
            cat = self.all_objs[idx]
            filename = os.path.join(grasp_dir, cat)
            cat = cat.split('_')[0]
            self.objs[idx] = Grasps(cat, filename, self.md, self.cd, self.grasp_dir)

        return (os.path.join(self.grasp_dir, self.objs[idx].mesh_filename),
                self.objs[idx].scale,
                self.objs[idx].mass)

    def load_mesh(self, idx, visualize=False):
        """
        Load mesh information
        """
        filename, scale, mass = self.get_mesh_info(idx)
        mesh = open3d.io.read_triangle_mesh(filename)
        pcd = open3d.geometry.PointCloud()
        pts = np.asarray(mesh.vertices) * scale
        pcd.points = open3d.utility.Vector3dVector(pts)
        if visualize:
            open3d.visualization.draw_geometries([pcd])
        return ScaledMesh(mesh, scale, mass, scaled_pts=pts, cat=self.objs[idx].category)

    def add_mesh_to_env(self, name, env, idx=None):
        """ Create mesh given physics backend `env` and index `idx`. If index is none, then choose a
        random mesh and add it to the environment. 
        Returns: newly added object reference, config dictionary. """
        filename, scale, mass = self.get_mesh_info(idx)
        obj = env.load_obj(name, filename, scale=scale, mass=mass)
        conf = {
                "obj_type": "shapenet",
                }
        return obj, conf

if __name__ == "__main__":
    data = GraspDataset()
    meshes = data.load_meshes()
    out_dir = "/data/grasping-meshes"
    for key, mesh in tqdm(meshes.items()):
        print(key)
        mesh.save(os.path.join(out_dir, "pos_" + key + "_data"))

