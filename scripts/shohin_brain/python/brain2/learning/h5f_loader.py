# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

import torch
import h5py
import numpy as np
import os

from brain2.utils.info import logwarn
import brain2.utils.image as img
from torchvision import transforms
import PIL

blocks_seg_ids = {'red_block': 6, 'green_block': 7, 'blue_block': 8, 'yellow_block': 9}
blocks_seg_classes = {6: 0, 7: 1, 8: 2,9: 3}

def get_rgb_transforms(mode):
    if mode == 'train':
        transform =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.05, contrast=0.25, saturation=0.25, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # TODO add background augmentation
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    return transform

def get_seg_transforms():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        ])
    return transform


class H5fDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab,
                 max_length=32, domain=None, exp_type=None, filenames=[],
                 mode='train', verbose=False, max_plan_length=32, model_type='seq2seq',
                 **kwargs):
        """ Initialize a dataset with h5f files for the different objects. """
        self.data_root = data_root
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        self.dec_verb_vocab = dec_verb_vocab
        self.dec_object_vocab = dec_object_vocab
        self.max_length = max_length
        self.max_plan_length = max_plan_length
        self.domain = domain
        self.exp_type = exp_type
        self.mode = mode
        self.model_type = model_type
        self.verbose = verbose
        self.num_seg_classes = len(blocks_seg_ids.keys())

        # Index needs to store file indices for all the different possible objects
        self.index = {}

        assert self.mode in {'train', 'valid', 'test'}

        # Load from data root
        self.language_keys = ["lang_goal", "lang_plan", "lang_description", "sym_plan", "sym_goal"]

        if len(filenames) > 0:
            self.files = [os.path.join(data_root, filename) for filename in filenames]
        else:
            self.files = [f for f in os.listdir(self.data_root)
                          if f[0] != '.' and f.endswith('.h5')]
        self._process_files()

        self.transform_rgb = get_rgb_transforms(self.mode)
        self.transform_seg = get_seg_transforms()

    def _process_files(self):
        """ Handling for adding new files to this dataset """

        # Store information
        num = 0
        self.file_to_count = []
        self.data_files = []
        # create splits based onm 
        self.train_files = []
        self.test_files = []

        # Loop over the files
        for i, f in enumerate(self.files):
            if f[0] == '.' or not f.endswith('.h5'): continue
            filename = f
            data = h5py.File(filename, 'r')
            for key in self.language_keys:
                lang = data[key][()]
                # pre-process symbol plan and symbol goal for logical tokenization
                if 'sym_goal' in key:
                    lang = self.dec_vocab.add_goal_states(lang, data['sym_values'][()])
                if 'sym_plan' in key:
                    lang, verbs, objects = self.dec_vocab.add_plan_states(lang)
                # print for more logs
                if self.verbose:
                    print(key, "=", lang)
                # append new tokens to vocab
                if 'lang' in key:
                    self.enc_vocab.add_sentence(lang)
                elif 'sym' in key:
                    self.dec_vocab.add_sentence(lang)
                    if (self.model_type == 'maskpred' or self.model_type == 'maskselect') and 'sym_plan' in key:
                        self.dec_verb_vocab.add_sentence(verbs)
                        self.dec_object_vocab.add_sentence(objects)
                else:
                    logwarn("Unknown key: " + str(key))


            task_code = data['task_code'][()]

            # Add actions from avail/unavail
            ua = data['unavailable_actions'][()].split('\n')
            aa = data['available_actions'][()].split('\n')
            [self.dec_vocab.add_sentence(a) for a in ua + aa if len(a) > 0]

            if self.exp_type == 'lang2sym':
                # consider one h5 file as one sample only
                # this would translate idx in __getitem__ as 0 always to pick one item
                num_indices = 1
            else:
                num_indices = data['q'].value.shape[0]
            num += num_indices
            self.file_to_count.append(num)

            self.data_files.append(filename)

            data.close()

        self.file_to_count = np.array(self.file_to_count)
        self.num_indices = num

    def __len__(self):
        return self.num_indices

    def __getitem__(self, idx):
        # Create the datum to return
        file_idx = np.argmax(idx < self.file_to_count)
        data = h5py.File(self.data_files[file_idx], 'r')
        if file_idx > 0:
            # for lang2sym, idx is always 0
            idx = idx - self.file_to_count[file_idx - 1]

        seq_length = data['q'][()].shape[0]

        q = data['q'][()][idx] # initial pose
        logical = data['logical'][()][idx]
        dmin = data['depth_min'][idx]
        dmax = data['depth_max'][idx]

        # get verb/object pairs for the plan sequence
        sym_plan_clean, sym_plan_clean_verbs, sym_plan_clean_objects = self.dec_vocab.add_plan_states(data['sym_plan'][()])

        if self.model_type == 'maskpred' or self.model_type == 'maskselect':
            # stack all rgb images and seg masks, pad with zeros tensors
            all_rgb = []
            all_seg = []
            all_seg_classes = []
            for one_step in range(self.max_plan_length):
                if one_step < seq_length-1 and seq_length > 1:
                    rgb = img.PNGToNumpy(data['rgb'][one_step])[:,:,:3] # remove alpha channel

                    if self.model_type == 'maskselect':
                        # use full segmentation mask
                        seg = (img.PNGToNumpy(data['seg'][one_step])).astype(np.uint8)
                    else:
                        # convert segmentation mask to use only current object
                        seg = (img.PNGToNumpy(data['seg'][one_step]) == blocks_seg_ids[sym_plan_clean_objects.split(' ')[one_step]]).astype(np.uint8)
                    # get class number for current object id
                    seg_class = torch.Tensor([[blocks_seg_classes[blocks_seg_ids[sym_plan_clean_objects.split(' ')[one_step]]]]]
                        ).type(torch.int64)
                else:
                    rgb = np.uint8(np.zeros((224, 224, 3)))
                    seg = np.uint8(np.zeros((224, 224)))
                    seg_class = torch.Tensor([[-1]]).type(torch.int64)
                all_rgb.append(self.transform_rgb(rgb))
                all_seg.append(self.transform_seg(seg))
                all_seg_classes.append(seg_class)
            all_rgb = torch.stack(all_rgb)
            all_seg = torch.stack(all_seg)
            all_seg_classes = torch.stack(all_seg_classes)
        else:
            all_rgb = torch.stack([self.transform_rgb(img.PNGToNumpy(data['rgb'][idx])[:,:,:3])])
            all_seg = torch.stack([self.transform_seg(img.PNGToNumpy(data['seg'][idx]))])
            all_seg_classes = torch.zeros(self.max_plan_length, dtype=torch.int64)

        # for depth, just taking first step for now
        depth = img.PNGToNumpy(data['depth'][idx]) / 255. * (dmax - dmin) + dmin

        # embed language data
        lang_desc = self.enc_vocab.embed_sentence(data['lang_description'][()], length=self.max_length)
        lang_goal = self.enc_vocab.embed_sentence(data['lang_goal'][()], length=self.max_length)
        lang_plan = self.enc_vocab.embed_sentence(data['lang_plan'][()], length=self.max_length)

        sym_plan = self.dec_vocab.embed_sentence(sym_plan_clean, length=self.max_length)
        sym_plan_verbs = self.dec_verb_vocab.embed_sentence(sym_plan_clean_verbs, length=self.max_plan_length)
        sym_plan_objects = self.dec_object_vocab.embed_sentence(sym_plan_clean_objects, length=self.max_plan_length)

        sym_goal_with_states = self.dec_vocab.add_goal_states(data['sym_goal'][()], data['sym_values'][()])
        sym_goal = self.dec_vocab.embed_sentence(sym_goal_with_states, length=self.max_length)

        oa = data['sym_plan'][()].split(',')
        ua = data['unavailable_actions'][()].split('\n')
        aa = data['available_actions'][()].split('\n')
        skip = False
        if len(aa) < len(oa):
            print("avail actions =", aa, "\nlen =", len(aa))
            print("filename:", self.data_files[file_idx])
            skip = True
            raise RuntimeError('not enough listed actions')
        #assert len(aa) >= len(oa)

        # assert(len(oa) < self.max_plan_length)
        obs_verbs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        good_verbs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        bad_verbs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        obs_to_objs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        good_to_objs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        bad_to_objs = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        #obs_done = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        obs_done = torch.zeros(self.max_plan_length).type(torch.FloatTensor)
        done_mask = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        #bad_done = torch.zeros(self.max_plan_length).type(torch.LongTensor)
        #good_done = torch.zeros(self.max_plan_length).type(torch.LongTensor)

        # Mark the final step as "done"
        final_pos = min(len(oa), self.max_plan_length-1) # this is the state where we are DONE
        done_mask[final_pos] = 1

        # Loop over the whole plan to compute the actions we can execute going forward in time
        # this is necessary for training the "executability" model
        for i, action in enumerate(oa):
            if i >= self.max_plan_length:
                continue
            if len(action) > 0:
                obs_verb, obs_to_obj = self.dec_vocab.embed_action(action)
                obs_verbs[i] = obs_verb
                obs_to_objs[i] = obs_to_obj

            obs_done[i] = final_pos - i

            # TODO: remove this bad-data hack until we understand why it happens
            if not skip:
                # Good (feasible/executable) actions
                good_acts = [a for a in aa[i].split(',') if len(a) > 0]
                if len(good_acts) > 0:
                    # Try to avoid choosing the same action we chose above
                    res = np.random.choice(good_acts)
                    if len(good_acts) > 1:
                        while res == action:
                            res = np.random.choice(good_acts)
                    good_verb, good_to_obj  = self.dec_vocab.embed_action(res)
                    good_verbs[i] = good_verb
                    good_to_objs[i] = good_to_obj

                # Bad (infeasible/non-executable) actions
                bad_acts = [a for a in ua[i].split(',') if len(a) > 0]
                if len(bad_acts) > 0:
                    # print(i, bad_acts)
                    bad_verb, bad_to_obj  = self.dec_vocab.embed_action(np.random.choice(bad_acts))
                    bad_verbs[i] = bad_verb
                    bad_to_objs[i] = bad_to_obj

        # DEBUG
        debug = False
        if debug:
            print("obs", obs_verbs)
            print('LANGUAGE: '+data['lang_goal'][()])
            print('SYMBOL: '+data['sym_goal'][()])
            import matplotlib.pyplot as plt;
            plt.figure()
            rgb = np.rollaxis(all_rgb[0].cpu().numpy(), 0, 3)
            plt.imshow(rgb)
            #plt.subplot(1,3,1); plt.imshow(rgb)
            #plt.subplot(1,3,2); plt.imshow(seg)
            #plt.subplot(1,3,3); plt.imshow(depth)
            plt.show()

        datum = {
                    'filename': self.data_files[file_idx],
                    'q': q,
                    'rgb': all_rgb,
                    'depth': depth,
                    'segmentation': all_seg,
                    'segmentation_classes': all_seg_classes,
                    # TODO: this is causing issues
                    #'logical': logical,
                    'lang_description': lang_desc,
                    'lang_goal': lang_goal,
                    'lang_plan': lang_plan,
                    'sym_plan': sym_plan,
                    'sym_plan_verbs': sym_plan_verbs,
                    'sym_plan_objects': sym_plan_objects,
                    'sym_goal': sym_goal,
                    # Action information for executability model
                    'sym_verb_good': good_verbs,
                    'sym_verb_obs': obs_verbs,
                    'sym_verb_bad': bad_verbs,
                    'sym_to_obj_good': good_to_objs,
                    'sym_to_obj_obs': obs_to_objs,
                    'sym_to_obj_bad': bad_to_objs,
                    # Plan completion for executabiltiy model
                    'done': obs_done,
                    'done_mask': done_mask,
                    #'good_done': good_done,
                    #'bad_done': bad_done,
                }

        # TODO: remove this. for data size issues
        dbg = False
        if dbg:
            for k, v in datum.items():
                try:
                    print(k, v.shape)
                except Exception as e:
                    print("!!!", k, e)

        # Search through the domain
        if self.domain is not None:
            for entity, state in self.domain.root.items():
                pass

        return datum
