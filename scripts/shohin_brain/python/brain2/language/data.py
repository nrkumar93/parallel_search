# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function
import json
import os

class LanguageWriter(object):
    """ Class to wrap data file creation """

    def __init__(self, filename=None):
        self.filename = filename

    def add(self, code, lang, goal, goal_lang, plan, plan_lang):

        data = [{
                'code': code,
                'lang': lang,
                'goal': goal,
                'goal_lang': goal_lang,
                'plan': plan,
                'plan_lang': plan_lang}]

        if os.path.exists(self.filename):
            # Read and append to existing data
            with open(self.filename) as f:
                orig_data = json.load(f)
            with open(self.filename, 'w') as f:
                orig_data.append(data)
                json.dump(orig_data, f, indent=4)
        else:
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=4)
