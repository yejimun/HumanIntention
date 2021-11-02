import pickle
from os.path import join
import numpy as np
import torch
import pdb

with open('Dataset/multi_intention_dataset.p', 'rb') as f:
    # dataset = pickle.load(f)
    # for python3
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    dataset = u.load()
intentions, trajs = dataset['intentions'], dataset['trajs']
print(intentions) # np array (40,) # with intentions 1, 2, and 3
print((intentions==1).sum()) # 12
print((intentions==2).sum()) # 16
print((intentions==3).sum()) # 12

# with open('Dataset/2TargetsLabeled.p', 'rb') as f:
#     # dataset = pickle.load(f)
#     # for python3
#     u = pickle._Unpickler( f )
#     u.encoding = 'latin1'
#     dataset = u.load()

n_intentions = len(np.unique(intentions))
sort_indx = np.argsort(intentions)


# ! intention starts from 1
data = {}
for i in range(n_intentions):
    data[i] = []
    indx = np.where(intentions==i+1)[0]
    for idx in indx:
        data[i].append(torch.Tensor(dataset['trajs'][idx]))
    

dataset_filename = 'multi_intention_labeled.p'
dataset_filepath = join('Dataset', dataset_filename)
with open(dataset_filepath,'wb') as f:
    pickle.dump(data, f)
    # 0: list of (variable_t, 2) tensors with intention 0
    # 1: list of (variable_t, 2) tensors with intention 1
    print(dataset_filepath+" is dumped.")




pdb.set_trace()



"""
trajs: list of trajectories with length of 40. trajs correspond to the intentions with same index.
    - trajectory: np array with shape # (T, 4) # (x,y,z,dt)
        # where dt is the current timestep - previous time step.

"""
