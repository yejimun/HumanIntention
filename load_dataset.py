#!/usr/bin/env python3
from os.path import isdir, join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import glob
from sklearn.model_selection import train_test_split
import os
import pdb
"""
Interval & length of the sequence is different, but doesn't seem to matter from the original dataset.
Do we need a window size?
Don't need a target label?
There is no mask with all zeros?
"""

pkg_path = '/home/yeji/HumanIntention'
dataset_filename = '2Targets.p'
dataset_filepath = join(pkg_path, 'Dataset', dataset_filename)
with open(dataset_filepath,'rb') as f:
    x_dict_list_tensor = pickle.load(f)
print()
print('LOAD DATASET')
print(dataset_filename+' is loaded.')
print()
traj_base_train, traj_true_train, traj_loss_mask_train, \
traj_base_test, traj_true_test, traj_loss_mask_test = \
x_dict_list_tensor['train']['base'], x_dict_list_tensor['train']['true'], x_dict_list_tensor['train']['loss_mask'], \
x_dict_list_tensor['test']['base'], x_dict_list_tensor['test']['true'], x_dict_list_tensor['test']['loss_mask']

for i in range(20):
    save_path = pkg_path+'/Dataset/ex/'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    print('length of eg. trajectory :',traj_loss_mask_train[i].shape)
    plt.figure()
    plt.plot(traj_base_train[i][:,0], traj_base_train[i][:,1])
    plt.plot(traj_true_train[i][:,0], traj_true_train[i][:,1])
    plt.savefig(save_path+'true_base_ex'+str(i)+'.png')

    plt.figure()
    plt.plot(traj_base_train[i][:,0], traj_base_train[i][:,1])
    plt.plot(traj_true_train[i][:,0], traj_true_train[i][:,1])
    traj_true_pred_train = traj_loss_mask_train[i].unsqueeze(1)*traj_true_train[i]
    # traj_true_pred_train = traj_loss_mask_train[i][...,np.newaxis]*traj_true_train[i] # traj_loss_mask_train[0][...,np.newaxis]
    plt.plot(traj_true_pred_train[:,0], traj_true_pred_train[:,1])
    plt.savefig(save_path+'true_base_mask_ex'+str(i)+'.png')

### Plotting all collected trajectories into a single image 
plt.figure()
for i in range(len(traj_true_train)):
    plt.plot(traj_true_train[i][:,0], traj_true_train[i][:,1])
plt.savefig(save_path+'collected_traj.png')
plt.show()