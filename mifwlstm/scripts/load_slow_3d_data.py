# from os.path import isdir, join
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# # import rosbag
# import glob
# from sklearn.model_selection import train_test_split, KFold
# import os
# import pdb
# import math
# import torch
# from datetime import datetime
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from datetime import datetime
# from os.path import join
# from os import makedirs
# import time
# import torch
import pickle
from pathhack import pkg_path
# device = "cuda:0" if torch.cuda.is_available() else "cpu"


dataset_filepath = 'datasets/multi_intention_dataset_slow.p'#'short_prediction_data.pt'

with open(dataset_filepath,'rb') as f:
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    raw_data = u.load()

data = {}
for intention, traj in zip(raw_data['intentions'], raw_data['trajs']):
    if intention not in data.keys():
        data[intention] = []
    # traj = torch.from_numpy(traj).float()*100.
    data[intention].append(traj*100.)
data[1] = data[2]
import pdb; pdb.set_trace();
# dataset = torch.load(join(pkg_path, 'datasets', dataset_filepath))

# xobs_train, xpred_train, yintention_train, xobs_test, xpred_test, yintention_test = \
# dataset["xobs_train"], dataset["xpred_train"], dataset["yintention_train"], dataset["xobs_test"], \
# dataset["xpred_test"], dataset["yintention_test"]
# obs_seq_len, pred_seq_len = dataset["obs_seq_len"], dataset["pred_seq_len"]
# print(xobs_train)
# short_prediction_multi_intention_data_slow