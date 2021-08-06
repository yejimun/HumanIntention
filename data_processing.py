#!/usr/bin/env python3
from os.path import isdir, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import glob
from sklearn.model_selection import train_test_split
from itertools import groupby
from operator import itemgetter
import os
import torch
import logging
import sys
import pdb

"""
source myenv/bin/activate
source /opt/ros/melodic/setup.bash
python data_processing.py

Interval & length of the sequence is different, but doesn't seem to matter from the original dataset.
Do we need a window size?
Don't need a target label?
There is no mask with all zeros?
"""

pkg_path = '/home/yeji/HumanIntention'
data_path = './Data/'
T1_traj_path = glob.glob(data_path +'*L_*.bag')
T2_traj_path = glob.glob(data_path +'*R_*.bag')
obs_ratio = 0.4
    


def make_sequence(data, window_len=0, offset=0, obs_ratio):
    """
    return 
    true : ground truth
    base : straight line connecting goal and last observation
    mask : zero mask for observation
    """
    # for i in range(len(data)):
    ## Currently using just the fixing observation length
    data_len = len(data)
    obs_len = int(data_len*obs_ratio)

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
    base = np.concatenate((np.array(data)[~mask.astype(bool)], base))

    return base, np.array(data), mask #torch.FloatTensor(base), torch.FloatTensor(np.array(data)), torch.FloatTensor(mask) # base, true, mask

def cut_start_end(data, timewindow=5, limit_std=2.):
    """
    input
    data : list of positions from one trajectory
    
    return : list of positions from cut trajectory
    """
    # monitor data sequence where std is less than a limit_std
    cut_ind = []
    if len(data)/timewindow > 0:
        for i in range(int(len(data)/timewindow)):
            dt = data[timewindow*i:timewindow*(i+1)]
            std = np.std(dt,axis=0)
            # print(dt, ' / std:', std)
            if np.all(std < limit_std):
                cut_ind.append(i)        
    else: 
        raise NotImplementedError

    # pick only the first/last few, continuing sequence within the limit
    cut_group = []
    for k, g in groupby(enumerate(cut_ind), lambda ix : ix[0] - ix[1]):
        indx = list(map(itemgetter(1), g))
        cut_group.append(indx)
    # print(cut_group)
    
    if len(cut_group)>1:
        start_cut = cut_group[0][-1]
        end_cut = cut_group[-1][0]
        # delete those sequence from the data / deal with the leftover
        cut_data = data[timewindow*(start_cut+1):timewindow*end_cut]
    elif len(cut_group)==1:
        if 0 in cut_group[0]:
            start_cut = cut_group[0][-1]
            end_cut = []
            cut_data = data[timewindow*(start_cut+1):]
        else:
            start_cut = []
            end_cut = cut_group[-1][0]
            cut_data = data[:timewindow*end_cut]
    else:
        pass
    # print(start_cut, end_cut)
    # print(len(data), len(cut_data))
    # pdb.set_trace()

    return cut_data

def stack_wtensors(data, train_idx, test_idx):
    train_bases = [torch.FloatTensor(data[idx]) for idx in train_idx]
    test_bases = [torch.FloatTensor(data[idx]) for idx in test_idx]
    return train_bases, test_bases


def DataProcessing(T_path, target):
    bases = []
    trues = []
    masks = []
    data_length = []
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
        traj = cut_start_end(traj)
        data_length.append(len(traj))

        base, true, mask = make_sequence(traj)
        bases.append(base)
        trues.append(true)
        masks.append(mask)
        # all_traj.append(np.array(traj))
        bag.close()
    # bases = np.array(bases) # (num_traj, )
    # trues = np.array(trues) # (num_traj, )
    # masks = np.array(masks) # (num_traj, )
    
    # # train, test split
    # n_traj = np.arange(len(bases))
    # train_idx, test_idx = train_test_split(n_traj, test_size=0.2, random_state=42)
    # train_bases, test_bases = bases[train_idx], bases[test_idx]
    # train_trues, test_trues = trues[train_idx], trues[test_idx]
    # train_masks, test_masks = masks[train_idx], masks[test_idx]
    return  bases, trues, masks, data_length# (24,3), (6,3)

bases, trues, masks, data_lens = [], [], [], []
for i, T_path in enumerate([T1_traj_path, T2_traj_path]): # T1_traj_path
    base, true, mask, data_length = DataProcessing(T_path, i)    
    bases.append(base)
    trues.append(true)
    masks.append(mask)
    data_lens.append(data_length)

bases = np.concatenate(bases, axis=0) # torch.Tensor(np.concatenate(train_sets, axis=0)) # (48, 3)
trues = np.concatenate(trues, axis=0) #torch.Tensor(np.concatenate(test_sets, axis=0)) # (12, 3)
masks = np.concatenate(masks, axis=0)

n_traj = np.arange(len(bases))
train_idx, test_idx = train_test_split(n_traj, test_size=0.2, random_state=42)


train_bases, test_bases = stack_wtensors(bases, train_idx, test_idx)
train_trues, test_trues = stack_wtensors(trues, train_idx, test_idx)
train_masks, test_masks = stack_wtensors(masks, train_idx, test_idx)


x_dict_list_tensor={'train':{'base':train_bases, 'true':train_trues, 'loss_mask':train_masks},
'test':{'base':test_bases, 'true':test_trues, 'loss_mask':test_masks}}


dataset_folder = join(pkg_path, 'Dataset')
if not os.path.exists(dataset_folder):
    os.mkdir(dataset_folder) 
dataset_filename = '2Targets.p'
dataset_filepath = join(pkg_path, 'Dataset', dataset_filename)

with open(dataset_filepath,'wb') as f:
    pickle.dump(x_dict_list_tensor, f)
log_file = os.path.join(dataset_folder, 'data.log')
mode = 'a'
file_handler = logging.FileHandler(log_file, mode=mode)
stdout_handler = logging.StreamHandler(sys.stdout)
level = logging.INFO
logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                    format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %%M:%S")
logging.info('data length (mean/std) T1, T2: ({:.2f}/{:.2f}), ({:.2f}/{:.2f}), observation ratio : {:.2f}'. format(np.mean(data_lens[0]), np.std(data_lens[0]), np.mean(data_lens[1]), np.std(data_lens[1]),obs_ratio))   

