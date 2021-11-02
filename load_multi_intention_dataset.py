from logging import raiseExceptions
import pickle
from os.path import join
import numpy as np
import torch
import matplotlib.pyplot as plt
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


def cut_start_end(traj, nth_traj, obs_len=4, threshold=0.02):
    stds = []
    # if len(traj)/obs_len > 0:
    for i in range(len(traj)-obs_len):
        dt = traj[i:i+obs_len]
        std = np.sum(np.std(dt[:, :3],axis=0)) #np.sum(np.std(dt,axis=0))
        # dt = np.array([traj[i,:3], traj[obs_len*(i+1)-1,:3]])
        # dist = np.sum(np.linalg.norm(dt, axis=0))
        stds.append(std)
        # if np.all(std < threshold):
        #     cut_ind.append(i)  

    # print('std:', stds)
    cut_ind = np.where(np.array(stds) > threshold)[0]
    # print(cut_ind)
    start_ind = np.min(cut_ind)
    end_ind = np.max(cut_ind)
    cut_traj = traj[start_ind:end_ind]
    print('start/end',start_ind, end_ind)
    plt.figure()
    ax = plt.axes(projection='3d')   
    ax.plot3D(traj[:,0], traj[:,1],traj[:,2], 'o-', c='C0') # blue
    ax.plot3D(cut_traj[:,0], cut_traj[:,1], cut_traj[:,2], 'o-', c='C1')#, markersize=10.) # orange
    plt.savefig('cut_traj3D/threshold_'+str(threshold)+'_traj_'+str(nth_traj)+'.png')    
    return cut_traj



# ! intention starts from 1
data = {}
for i in range(n_intentions):
    data[i] = []
    indx = np.where(intentions==i+1)[0]
    # pdb.set_trace()
    for idx in indx:
        trajs = dataset['trajs'][idx]
        cut_trajs = cut_start_end(trajs, idx)
        data[i].append(torch.Tensor(cut_trajs))
    

dataset_filename = 'multi_intention_labeled.p'
dataset_filepath = join('Dataset', dataset_filename)
with open(dataset_filepath,'wb') as f:
    pickle.dump(data, f)
    # 0: list of (variable_t, 2) tensors with intention 0
    # 1: list of (variable_t, 2) tensors with intention 1
    print(dataset_filepath+" is dumped.")





"""
trajs: list of trajectories with length of 40. trajs correspond to the intentions with same index.
    - trajectory: np array with shape # (T, 4) # (x,y,z,dt)
        # where dt is the current timestep - previous time step.

"""
