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
import rospy
from rosop.msg import HumanSkeletonPosition
from std_msgs.msg import Float32MultiArray
# from tf import transformations as tfs

def arg_parse():
    parser = argparse.ArgumentParser()
    # * Intention LSTM
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=32, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--obs_seq_len', default=20, type=int)
    parser.add_argument('--pred_seq_len', default=20, type=int)
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


class OnlineMIF:
    def __init__(self, args, device, tracking_topic_name, transformation_matrix):
        self.intention_api = HRIIntentionApplicationInterface(
            method=args.prediction_method,
            scene=args.scene,
            args=args,
            pkg_path=pkg_path,
            device=device,
        )
        self.intention_pf = IntentionParticleFilter(
            self.intention_api.get_num_intentions(),
            args.num_particles_per_intention,
            self.intention_api,
        )
        self.args = args
        self.transformation_matrix = transformation_matrix
        # ! without max traj len, trajectory will keep growing.
        # ! Have a reset button for bunch of experiments?
        rospy.init_node('online_mutable_intention_filter', anonymous=True)
        self.intention_prob_dist_pub = rospy.Publisher("hri/human_intention_prob_dist", Float32MultiArray, queue_size=1)
        rospy.Subscriber(tracking_topic_name, HumanSkeletonPosition, self.tracking_callback)
        self.reset()
    
    def reset(self):
        self.intention_pf.reset()
        self.trajectory = np.empty((0,args.motion_dim))
        self.current_filtering_results = {}
        self.current_filtering_results['intention_prob_dist'] = self.intention_pf.get_intention_probability()
        self.current_filtering_results['predicted_trajectories'] = None
        self.current_filtering_results['particle_intentions'] = self.intention_pf.get_intention()
        intention_prob_dist_msg = Float32MultiArray()
        intention_prob_dist_msg.data = list(self.current_filtering_results['intention_prob_dist'].astype(np.float32))
        self.intention_prob_dist_pub.publish(intention_prob_dist_msg)

    def tracking_callback(self, data):
        pos = np.array(data.data).reshape(data.n_humans, data.n_keypoints, data.n_dim) # (n_human, 25, 3)
        wrist_pos = pos[0,4] # (3,) right wrist
        wrist_pos_tf = np.dot(self.transformation_matrix, np.concatenate((wrist_pos,[1.]),axis=0))[:3] # (3,)
        self.trajectory = np.concatenate((self.trajectory, wrist_pos_tf[np.newaxis,:]), axis=0) # (t, 3)
        if len(self.trajectory) < self.args.Tf+self.args.T_warmup:
            return
        if (len(self.trajectory)-(self.args.Tf+self.args.T_warmup)) % self.args.step_per_update == 0:
            x_obs = self.trajectory
            self.intention_pf.predict(x_obs)
            self.intention_pf.update_weight(x_obs, tau=args.tau)
            self.intention_pf.resample()
            if self.args.mutable:
                self.intention_pf.mutate(mutation_prob=args.mutation_prob)
            intention_prob_dist = self.intention_pf.get_intention_probability() # (2,)
            predicted_trajectories = self.intention_api.predict_trajectories(
                x_obs,
                self.intention_pf.get_intention(),
                truncated=True, # does not matter for ilstm
            )
            self.current_filtering_results['intention_prob_dist'] = intention_prob_dist
            self.current_filtering_results['predicted_trajectories'] = predicted_trajectories
            self.current_filtering_results['particle_intentions'] = self.intention_pf.get_intention()
            intention_prob_dist_msg = Float32MultiArray()
            intention_prob_dist_msg.data = list(intention_prob_dist.astype(np.float32))
            self.intention_prob_dist_pub.publish(intention_prob_dist_msg)
    
    def get_current_intention_prob_dist(self):        
        return self.current_filtering_results['intention_prob_dist']

    def run(self):
        while not rospy.is_shutdown():
            command_input = input("\
                \n=====================================================\
                \nReset for next round: r\
                \nGet current intention probability distribution: g\
                \n=====================================================\
                \nCommand: ")
            if command_input == "r":
                self.reset()
            elif command_input == "g":
                curr_intention_prob_dist = self.get_current_intention_prob_dist()
                print()
                print(curr_intention_prob_dist)
                print()
            else:
                print("Th command ["+command_input+"] is invalid.")

if __name__ == '__main__':
    tracking_topic_name = "/hri/human_skeleton_position_tracking_3D"
    transformation_matrix = np.array([[ 0.00622504,  0.99822903, -0.05916131, -0.46120499],
                                    [ 0.99959345, -0.00456556,  0.02814404, -0.00580957],
                                    [ 0.02782409, -0.05931246, -0.99785162,  1.83695167],
                                    [ 0.,          0.,          0.,          1.        ],])
    args = arg_parse()
    args.Tf = args.pred_seq_len
    args.T_warmup = args.obs_seq_len
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    try:
        online_mif = OnlineMIF(
            args,
            device,
            tracking_topic_name,
            transformation_matrix,
        )
        online_mif.run()
    except rospy.ROSInterruptException:
        pass

