#!/usr/bin/env python3
from os.path import isdir, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import glob
from sklearn.model_selection import train_test_split
import os
# import torch
import pdb
"""
Interval & length of the sequence is different, but doesn't seem to matter from the original dataset.
Do we need a window size?
Don't need a target label?
There is no mask with all zeros?
"""

pkg_path = '/home/yeji/mifwlstm'
data_path = './Data/'
T1_traj_path = glob.glob(data_path +'*L_*.bag')
T2_traj_path = glob.glob(data_path +'*R_*.bag')


def make_sequence(data, window_len=0, offset=0):
    """
    return 
    true : ground truth
    base : straight line connecting goal and last observation
    mask : zero mask for observation
    """
    # for i in range(len(data)):
    ## Currently using just the fixing observation length
    data_len = len(data)
    obs_len = 50

    mask = np.ones(data_len)
    mask[:obs_len] = 0.
    last_obs = data[obs_len-1]
    goal_pos = data[-1]
    # ! Is base length len(data)-obs_length???
    interval_num = data_len-obs_len
    base_x = np.linspace(last_obs[0], goal_pos[0], interval_num)
    base_y = np.linspace(last_obs[1], goal_pos[1], interval_num)
    # print(base_x.shape, base_y.shape)
    base = np.stack((base_x, base_y)).T # (T, 2)

    return base, np.array(data), mask # base, true, mask




def DataProcessing(T_path, target):
    bases = []
    trues = []
    masks = []
    for bag_path in T_path:
        bag = rosbag.Bag(bag_path)
        prev_t = None
        traj = []
        for topic, msg, t in bag.read_messages(topics=['/q3/hand_pixel_position']):
            pos = msg.data
            time = t.to_sec()
            if prev_t == None:
                rel_t = 0.
                start_time = t.to_sec()
            else:
                rel_t = time - start_time
            traj.append([msg.data[0], msg.data[1]]) # [target, rel_t, msg.data[0], msg.data[1]]
            prev_t = time
        base, true, mask = make_sequence(traj)
        bases.append(base)
        trues.append(true)
        masks.append(mask)
        # all_traj.append(np.array(traj))
        bag.close()
    bases = np.array(bases) # (num_traj, )
    trues = np.array(trues) # (num_traj, )
    masks = np.array(masks) # (num_traj, )
    
    # # train, test split
    # n_traj = np.arange(len(bases))
    # train_idx, test_idx = train_test_split(n_traj, test_size=0.2, random_state=42)
    # train_bases, test_bases = bases[train_idx], bases[test_idx]
    # train_trues, test_trues = trues[train_idx], trues[test_idx]
    # train_masks, test_masks = masks[train_idx], masks[test_idx]
    return  bases, trues, masks# (24,3), (6,3)

bases, trues, masks = [], [], []
for i, T_path in enumerate([T1_traj_path, T2_traj_path]):
    base, true, mask = DataProcessing(T_path, i)    
    bases.append(base)
    trues.append(true)
    masks.append(mask)

bases = np.concatenate(bases, axis=0) # torch.Tensor(np.concatenate(train_sets, axis=0)) # (48, 3)
trues = np.concatenate(trues, axis=0) #torch.Tensor(np.concatenate(test_sets, axis=0)) # (12, 3)
masks = np.concatenate(masks, axis=0)

n_traj = np.arange(len(bases))
train_idx, test_idx = train_test_split(n_traj, test_size=0.2, random_state=42)
train_bases, test_bases = bases[train_idx], bases[test_idx]
train_trues, test_trues = trues[train_idx], trues[test_idx]
train_masks, test_masks = masks[train_idx], masks[test_idx]



x_dict_list_tensor={'train':{'base':train_bases, 'true':train_trues, 'loss_mask':train_masks},
'test':{'base':test_bases, 'true':test_trues, 'loss_mask':test_masks}}


dataset_folder = join(pkg_path, 'Dataset')
if not os.path.exists(dataset_folder):
    os.mkdir(dataset_folder) 
dataset_filename = '2Targets.p'
dataset_filepath = join(pkg_path, 'Dataset', dataset_filename)

with open(dataset_filepath,'wb') as f:
    pickle.dump(x_dict_list_tensor, f)
