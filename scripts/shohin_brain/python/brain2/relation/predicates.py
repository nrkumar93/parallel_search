# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import numpy as np
import itertools


class PredicateEvaluator(object):

    def __init__(self, backend):
        self.backend = backend


    def evaluate(self, world_state):
        self.backend.update(world_state)
        for name1, e1 in world_state.entities.items():
            for name2, e2 in world_state.entities.items():
                if name1 == name2:
                    continue
                preds = self.compute_pairwise(world_state, camera, e1, e2)
    
    def compute_pairwise(self, ws, cam, e1, e2):
        """
        Compute the set of entities
        """
        predicates = []
        predicates += self.right_left(e1, e2)
        predicates += self.front_back(e1, e2)
        predicates += self.up_down(e1, d2)
        predicates += self.stable(ws, e1, e2)
        predicates += self.occlusion(ws, cam, e1, e2)
        predicates += self.contact(e1, e2)
        pass

    def right_left(self, e1, e2):
        """
        Computes if e1 is left or right of e2. Compute left-right predicates.
        Use camera frame coordinates.

        Relation rules:
            1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
            2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
            3) do same as 1) for xy
            4) do same as 2) for xy
            5) o1 center MUST be to left of all o2 corners
            6) All o1 corners MUST be to the left of o2 center
        """
        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = sim_util.rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = sim_util.rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = sim_util.rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = sim_util.rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = e1.ref.id
        obj2_id = e2.ref.id

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
           other_args['predicates'].append((obj1_id, obj2_id, 'left'))
           other_args['predicates'].append((obj2_id, obj1_id, 'right'))    


    def front_back(self, e1, e2):
        """
        Computes if e1 is in front of or behind e2 in world frame
        """
        pass

    def up_down(self, e1, e2):
        """
        Computes if e1 is above/below e2 in world frame
        """
        pass

    def stable(self, ws, e1, e2):
        """
        Computes if e1 is stable
        """
        pass

    def occlusion(self, ws, cam, e1, e2):
        """
        Computes is_occluding, occludes given camera "cam"
        """
        pass

    def contact(self, e1, e2):
        """
        Computes is_colliding, is_touching
        """
        pass
