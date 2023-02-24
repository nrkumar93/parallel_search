# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import os
import json

class ShapenetModelDataset(object):
    
    def __init__(self, shapenet_dir='/data/ShapeNetCore.V2', config_dir='./config/'):
        self.dir = shapenet_dir
        self.taxonomy_file = os.path.join(self.dir, "taxonomy.json")
        with open(self.taxonomy_file, 'r') as f:
            self.taxonomy = json.load(f)
        self.synsets = [f for f in os.listdir(self.dir) if f[0] != '.']
        self.objs_by_synset = []

        # Loading config files
        self.objects_file = os.path.join(self.dir, 'training_shapenet_objects.json')
        self.tables_file = os.path.join(self.dir, 'training_shapenet_tables.json')

        # Process configuration files
        
    def add_mesh(self, name, env, idx=None):
        """ Return information for adding one mesh to a scene. Creates the mesh and adds it to the
        environment, so that we can create random scenes and test them out in the world.
        Returns: newly added object reference, config dictionary. """
        pass
        
    def sample_mesh(self):
        """ Loading mesh """
        synset_to_sample = np.random.choice(self.synsets.keys())
        model_dir = np.random.choice(self.synsets[synset_to_sample])
        model_dir = self.shapenet_base_dir + self.params['taxonomy_dict'][synset_to_sample] + '/' + model_dir + '/'
        obj_mesh_filename = model_dir + 'models/model_normalized.obj'
        obj_mesh_filenames.append(model_dir + 'models/model_normalized.obj')
