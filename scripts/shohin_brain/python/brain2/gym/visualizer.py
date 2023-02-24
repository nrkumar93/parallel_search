# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import pyrender
import multiprocessing as mp
import queue as Queue
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import os

class SceneVisualizer():
    """ Visualize meshes and stuff like that """

    def __init__(self):
        self._scene = None
        self._node_dict = None
        self._should_stop = False
        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._mesh_dict = {}
        self._viewer = pyrender.Viewer(self._scene, use_raymond_lighting=True, run_in_thread=True)
        self._pcs = {}


    def add_object(self, name, mesh, transform=None):
        self.lock_viewer()
        self._add_object(name, mesh, transform)
        self.unlock_viewer()
    
    def _add_object(self, name, mesh, transform):
        node = pyrender.Node(
            name=name, 
            mesh=pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
        )
        self._node_dict[name] = node
        self._mesh_dict[name] = mesh.copy()
        self._scene.add_node(node)
        # print('======> visualizer: adding {}'.format(name))
        if transform is None:
            transform = np.eye(4, dtype=np.float32)
        if transform is not None:
            self.update_pose(name, transform)
    
    def _add_pointcloud(self, name, npoints, cube_size, color):
        final_color = np.asarray(color)
        for i in range(npoints):
            mesh = trimesh.creation.box([cube_size for _ in range(3)])
            mesh.visual.face_colors = np.tile(
                np.reshape(final_color, [1, 3]), 
                [mesh.faces.shape[0], 1]
            )
            self._add_object(
                '{}/{}'.format(name, i), 
                mesh, 
                np.eye(4, dtype=np.float32)
            )

    def update_pointcloud_pose(self, name, xyzs, cube_size, color):
        self._pcs[name] = xyzs.copy()
        # self.lock_viewer()
        
        # if '{}/{}'.format(name, 0) not in self._node_dict:
        #     self._add_pointcloud(
        #         name, xyzs.shape[0], cube_size, color
        #     )
        
        # transform = np.eye(4)
        # for i, xyz in enumerate(xyzs):
        #     transform[:3, 3] = xyz
        #     self.update_pose(
        #         '{}/{}'.format(name, i), 
        #         transform
        #     )

        # self.unlock_viewer()
    
    def lock_viewer(self):
        self._viewer.render_lock.acquire()
    
    def unlock_viewer(self):
        self._viewer.render_lock.release()

    def update_pose(self, name, transform):
        self._scene.set_pose(self._node_dict[name], transform)
    
    def save_all_meshes(self, output_folder):
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        
        for name, mesh in self._mesh_dict.items():
            mesh.export(os.path.join(output_folder, name + '.obj'))
    
    def __del__(self):
        print('-->killing visualizer')
        self._viewer.close_external()

class SceneVisualizerProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self._visualizer = None
        self._queue = mp.Queue()
        self._should_stop = False
    
    def run(self):
        self._visualizer = SceneVisualizer()
        while True:
            try:
                request = self._queue.get(timeout=1)
            except Queue.Empty:
                continue

            if request[0] == 'update_pose':
                self._visualizer.update_pose(request[1], request[2])
            elif request[0] == 'add_object':
                self._visualizer.add_object(request[1], request[2].copy(), request[3])
            elif request[0] == 'terminate':
                del self._visualizer
                break
            elif request[0] == 'lock':
                self._visualizer.lock_viewer()
            elif request[0] == 'unlock':
                self._visualizer.unlock_viewer()
            elif request[0] == 'update_pc':
                self._visualizer.update_pointcloud_pose(*request[1:])
            elif request[0] == 'save':
                self._visualizer.save_all_meshes(request[1])
            else:
                raise NotImplementedError('invalid request {}'.format(request[0]))
    
    def terminate(self):
        self._queue.put(('terminate',))
    
    
    def add_object(self, name, mesh, transform=None):
        self._queue.put(('add_object', name, mesh, transform))
    
    
    def update_pose(self, name, transform):
        self._queue.put(('update_pose', name, transform))
    
    def lock_viewer(self):
        self._queue.put(('lock',))
    
    def unlock_viewer(self):
        self._queue.put(('unlock',))
    
    def update_pointcloud_pose(self, name, xyzs, cube_size, color):
        self._queue.put(('update_pc', name, xyzs, cube_size, color))
    
    def save_all_meshes(self, output_folder):
        self._queue.put(('save', output_folder))


class AsyncImageVisualizer(mp.Process):
    def __init__(
        self, 
        num_images_per_row, 
        output_resolution,
        num_classes = 50,
        ):
        super().__init__()
        self._grid_len = num_images_per_row
        self._output_resolution = output_resolution
        self._num_classes = num_classes
        self._queue = mp.Queue()
        cm = plt.get_cmap('gist_rainbow')
        self._colors = [cm(1. * i/num_classes) for i in range(num_classes)]
        self._colors = [(int(x[0]*255), int(x[1]*255), int(x[2]*255)) for x in self._colors]
    
    
    def make_grid(self, images, encodings):
        grid_cells = []
        for image, encoding in zip(images, encodings):
            if encoding == 'rgb':
                grid_cells.append(image[:,:,::-1])
            elif encoding == 'segmentation':
                output = [
                    np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for _ in range(3)
                ]
                for i in range(self._num_classes):
                    if i == 0:
                        continue
                    mask = image == i
                    for j in range(3):
                        output[j][mask] = self._colors[i][j]
                for j in range(3):
                    output[j] = np.expand_dims(output[j], -1)
                output[j] = np.concatenate(output, -1)
                grid_cells.append(output[j])
            elif encoding == 'depth':
                image = (image * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
                grid_cells.append(image)
            else:
                raise NotImplementedError('unknown encoding ({})'.format(encoding))
                
        
        extra = len(grid_cells) % self._grid_len
        if extra > 0:
            extra = 3 - extra
        # print(
        #     'extra {} len grid_cell {} grid_len {}'.format(
        #         extra, len(grid_cells), self._grid_len))
        for _ in range(extra):
            grid_cells.append(
                np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            )
        
        grid_cells = np.concatenate(grid_cells, 0)
        # print('grid_cells.shape = {}'.format(grid_cells.shape))
        step = self._grid_len * image.shape[0]
        
        grid_cols = []
        for i in range(0, grid_cells.shape[0], step):
            grid_cols.append(grid_cells[i:i+step, :, :])
        
        grid_cells = np.concatenate(grid_cols, 1)
        # grid_cells = cv2.resize(grid_cells, (self._output_resolution, self._output_resolution))


        # for i in range(0, len(grid_cells), self._grid_len):
            # grid_cols.append(grid_cells[i * image.shape[0]:(i+self._grid_len) * image.shape[0], :, :])
            # print(grid_cols[-1].shape)
        
        # return np.concatenate(grid_cols, 1)
        return grid_cells

    
    def run(self):
        while True:
            request = self._queue.get()
            if request[0] == 'stop':
                print('stopping async image visualizer')
                break
            elif request[0] == 'show':
                # print('show')
                images = request[1]
                encodings = request[2]
                grid_images = self.make_grid(images, encodings)
                cv2.imshow('async visualizer', grid_images)
                if cv2.waitKey(1) == 27:
                    print('stopping async image visualizer')
                    break
            else:
                raise ValueError('invalid request {}').format(request)
    
    def show_grid(self, images, encodings):
        assert len(images) == len(encodings)
        for e in encodings:
            if e not in ['rgb', 'segmentation', 'depth'] :
                raise ValueError('invalid encoding {}'.format(e))
        
        self._queue.put(('show', images, encodings))
    
    def terminate(self):
        self._queue.put(('stop',))

    








