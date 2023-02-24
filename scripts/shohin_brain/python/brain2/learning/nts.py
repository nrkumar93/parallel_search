# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function


class NeuralTreeNode(object):
    """
    Represents only a single tree search node (for now)
    """
    pass

class NeuralTreeSearch(object):
    """
    Contains a tree and builds it based on the models trained from language data.
    """
    def __init__(self, domain_model):
        self.domain = domain_model


    def search(self, rgb, goal_lang):
        outputs, h0, _ = self.domain.forward(goal_lang, rgb)
        root = self.NeuralTreeNode(h0)

