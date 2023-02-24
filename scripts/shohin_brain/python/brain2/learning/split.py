# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import print_function

from brain2.utils.info import logerr, logwarn
from brain2.language.vocab import Vocabulary
from brain2.learning.h5f_loader import H5fDataset

import h5py
import numpy as np
import os
import torch

from tqdm import tqdm


def split_h5_by_code(data_dir, train_val=0.9, verbose=True):
    """ Split directory by code. Create a bunch of different files containing
    lists of data files used for train/val/test.

    Held-out "test" examples are determined by task code. """
    if train_val <= 0 or train_val >= 1:
        raise RuntimeError('train val ratio must be > 0 and < 1')
    train_count = int(10 * train_val)
    if train_count <= 0 or train_val >= 1:
        raise RuntimeError('could not make ratio of ' + str(train_val)
                            + 'work; try some fraction of 10.')

    counter = 0
    files = os.listdir(data_dir)
    files = [f for f in files if f[0] != '.']
    train_file = open(os.path.join(data_dir, 'train.csv'), 'w')
    valid_file = open(os.path.join(data_dir, 'valid.csv'), 'w')
    test_file = open(os.path.join(data_dir, 'test.csv'), 'w')
    ntrain, nvalid, ntest = 0, 0, 0
    skipped = []
    for filename in tqdm(files):
        if not filename.endswith('.h5'):
            continue
        try:
            trial = h5py.File(os.path.join(data_dir, filename), 'r')
            code = trial['task_code'][()] 
            trial.close()
        except Exception as e:
            logerr('Problem handling file:' + str(filename))
            logerr('Failed with exception: ' + str(e))
            skipped.append(filename)
            continue
        if code  % 10 > 7:
            # we have a TEST example!
            test_file.writelines(filename + "\n")
            sort = 'test'
            ntest += 1
        else:
            if counter < train_count:
                train_file.writelines(filename + "\n")
                sort = 'train'
                ntrain += 1
            else:
                valid_file.writelines(filename + "\n")
                sort = 'valid'
                nvalid += 1
            counter += 1
            counter = counter % 10
        if verbose:
            print(filename, "is", sort, "\t#train/valid/test =", (ntrain, nvalid, ntest))

    if len(skipped) > 0:
        logwarn("Split finished with errors. Had to skip the following files: " + str(skipped))
    train_file.close()
    valid_file.close()
    test_file.close()


def _get_files(data_filename, *args, **kwargs):
    files = []
    with open(data_filename, 'r') as f:
        while True:
            fname = f.readline()
            if fname is not None and len(fname) > 0:
                files.append(fname.strip())
            else:
                break
    return files
    
def _get_files_from_dir(data_dir, split, *args, **kwargs):
    return _get_files(os.path.join(data_dir, split + '.csv'), *args, **kwargs)


def _get_all_splits_from_dir(data_dir, *args, **kwargs):
    return (_get_files_from_dir(data_dir, 'train', *args, **kwargs),
            _get_files_from_dir(data_dir, 'valid', *args, **kwargs),
            _get_files_from_dir(data_dir, 'test', *args, **kwargs))


def create_data_loaders(data_dir, enc_vocab_path, dec_vocab_path, exp_type,
                        max_length, batch_size, num_workers, model_type,
                        dec_verb_vocab_path, dec_object_vocab_path, verbose=True,
                        **kwargs):
    """
    Create the three separate data loaders for train/test/valid; create the vocabulary as well.
    Return all of these things so that the learning code has access.
    Additional parameters are passed down to the H5f dataset.
    """

    # This is spread across two lines for clarity + readability.
    # Get all the files we care about and put them in separate lists.
    train, valid, test = _get_all_splits_from_dir(data_dir, verbose)
    modes = ["train", "valid", "test"]
    splits = [train, valid, test]

    enc_vocab = Vocabulary(enc_vocab_path)
    dec_vocab = Vocabulary(dec_vocab_path)
    dec_verb_vocab = Vocabulary(dec_verb_vocab_path)
    dec_object_vocab = Vocabulary(dec_object_vocab_path)

    datasets = []
    loaders = []
    for mode, split in zip(modes, splits):
        print("\tmode =", mode)
        dataset = H5fDataset(data_dir,
                             enc_vocab=enc_vocab,
                             dec_vocab=dec_vocab,
                             dec_verb_vocab=dec_verb_vocab,
                             dec_object_vocab=dec_object_vocab,
                             max_length=max_length,
                             exp_type=exp_type,
                             filenames=split,
                             mode=mode,
                             model_type=model_type,
                             **kwargs)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=(mode == "train"),
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        datasets.append(dataset)
        loaders.append(dataloader)

    # For readability - separate this out and make it clear what we're returning here.
    # Again this is not really necessary
    train_loader, valid_loader, test_loader = loaders
    train_dataset, valid_dataset, test_dataset = datasets
    return enc_vocab, dec_vocab, dec_verb_vocab, dec_object_vocab, train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset
