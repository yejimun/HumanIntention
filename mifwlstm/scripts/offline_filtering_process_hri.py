from pathhack import pkg_path
import time
import numpy as np
import torch
import argparse
import pickle

from src.mif.intention_particle_filter import IntentionParticleFilter
from src.api.hri_intention_application_interface import HRIIntentionApplicationInterface
# from src.api.pedestrian_intention_application_interface import PedestrianIntentionApplicationInterface
from src.utils import load_preprocessed_train_test_dataset
import time

def arg_parse():
    parser = argparse.ArgumentParser()
    # * Intention LSTM
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--obs_seq_len', default=8, type=int)
    parser.add_argument('--pred_seq_len', default=12, type=int)
    parser.add_argument('--motion_dim', default=3, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    # Scene Options
    parser.add_argument('--num_intentions', default=2, type=int)
    parser.add_argument('--random_seed', default=0, type=int)

    # * Mutable Intention Filter
    # filtering
    parser.add_argument('--num_particles_per_intention', default=100, type=int,\
        help="10, 30, 50, 100.")
    parser.add_argument('--Tf', default=20, type=int, help="should be equal to pred_seq_len")
    parser.add_argument('--T_warmup', default=20, type=int, help="Number of steps for minimum truncated observation. At least 2 for heuristic in ilm. should be equal to obs_seq_len.")
    parser.add_argument('--step_per_update', default=2, type=int, help="1, 2")
    parser.add_argument('--tau', default=10., type=float, help="1., 10.")
    parser.add_argument('--prediction_method', default='ilstm', type=str, help='ilstm, wlstm or ilm.')
    parser.add_argument('--mutable', action='store_true')
    parser.add_argument('--mutation_prob', default=1e-2, type=float)
    parser.add_argument('--scene', default='fit', type=str)
    # evaluation
    parser.add_argument('--num_top_intentions', default=3, type=int, help='Number of top probability intentions. Options: 1, 3.')
    return parser.parse_args()


args = arg_parse()
args.Tf = args.pred_seq_len
args.T_warmup = args.obs_seq_len
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
hri_intention_application_interface = HRIIntentionApplicationInterface(
    method=args.prediction_method,
    scene=args.scene,
    args=args,
    pkg_path=pkg_path,
    device=device,
)

intention_particle_filter = IntentionParticleFilter(
    hri_intention_application_interface.get_num_intentions(),
    args.num_particles_per_intention,
    hri_intention_application_interface,
)

# load trajectory data
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
    data[intention].append(traj*100.) # unit: cm
data[1] = data[2]
intention_split_len = len(data[0])
offline_trajectories_set = data[0] + data[1] # list of len 20.

# _, _, _, _, traj_true_test, _ = load_preprocessed_train_test_dataset(pkg_path, dataset_ver=0)
# offline_trajectories_dataset = np.array([trajectory.numpy() for trajectory in traj_true_test], dtype=object) # np.ndarray of np.ndarray (variable_t, 2)
# offline_trajectories_indices = np.random.randint(0,len(offline_trajectories_dataset),size=10)
# offline_trajectories_set = offline_trajectories_dataset[offline_trajectories_indices] # list of len 10 of np.ndarray (variable_t, 2)
filtering_history_data = []
start_all_time = time.time()
for traj_idx, trajectory in enumerate(offline_trajectories_set):
    # trajectory: (variable_t, 2)
    filtering_history = {}
    if traj_idx < intention_split_len:
        ground_truth_intention = 0.#pedestrian_intention_application_interface.pedestrian_intention_sampler.position_to_intention(trajectory[-1])
    else:
        ground_truth_intention = 1.
    
    intention_particle_filter.reset()
    # import pdb; pdb.set_trace();
    filtering_history['trajectory'] = trajectory
    filtering_history['ground_truth_intention'] = ground_truth_intention
    filtering_history['last_obs_indices'] = list(range(args.Tf+args.T_warmup, len(trajectory)-args.Tf, args.step_per_update))
    filtering_history['intention_prob_dist'] = []
    # filtering_history['fixed_len_predicted_trajectories'] = []
    filtering_history['predicted_trajectories'] = []
    filtering_history['intention'] = []
    for last_obs_index in filtering_history['last_obs_indices']:
        start_time = time.time()
        x_obs = trajectory[:last_obs_index]
        intention_particle_filter.predict(x_obs)
        intention_particle_filter.update_weight(x_obs, tau=args.tau)
        intention_particle_filter.resample()
        if args.mutable:
            intention_particle_filter.mutate(mutation_prob=args.mutation_prob)
        intention_prob_dist = intention_particle_filter.get_intention_probability()
        # fixed_len_predicted_trajectories = hri_intention_application_interface.predict_trajectories(
        #     x_obs,
        #     intention_particle_filter.get_intention(),
        #     truncated=True
        # ) # (num_particles, Tf, 2)
        predicted_trajectories = hri_intention_application_interface.predict_trajectories(
            x_obs,
            intention_particle_filter.get_intention(),
            truncated=True,
        ) # a numpy array of objects. (num_particles, ), where one object is numpy. (variable_t, 2). The predicted trajectories of particles.
        filtering_history['intention_prob_dist'].append(intention_prob_dist)
        # filtering_history['fixed_len_predicted_trajectories'].append(fixed_len_predicted_trajectories)
        filtering_history['predicted_trajectories'].append(predicted_trajectories)
        filtering_history['intention'].append(intention_particle_filter.get_intention())
        # import pdb; pdb.set_trace();
        print("The iteration time: {0:.4f} sec".format(time.time()-start_time))
    filtering_history_data.append(filtering_history)
print('time to filter: {0:.2f} min'.format((time.time()-start_all_time)/60.))
# print('time to filter: {0:.2f} sec'.format(time.time()-start_time))
# print(filtering_history_data)
result_filename = 'hri_filtering_history_data_ilm.p'
with open(result_filename, 'wb') as f:
    pickle.dump(filtering_history_data, f)
    print(result_filename+' is created.')

# logdir = join(pkg_path, 'results', 'mif', args.prediction_method+'_tau_'+str(args.tau))
# result_filename = 'filtering_history.data.p'
# with open(join(logdir, result_filename), 'wb') as f:
#     pickle.dump(infos_list, f)
#     print(result_filename+' is created.')
