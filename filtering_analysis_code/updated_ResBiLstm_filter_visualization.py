# Pedestrian Intention Inference (pii)
# visualization_ipf.py
# ver 1.2
# Author: Zhe Huang
# Date: Nov. 10, 2019

# ver 1.0
# Transformed the code in animation_mmhpf into a class VisualizationMMHPF.
# It is able to plot track data as well as the predicted trajectory during the runtime.
# ver 1.1
# Cleaned the code and integrate with the map information module.
# Set up the plot with variable intent numbers.
# ver 1.2
# Added notification of visualization initialization.
# ver 1.3
# Changed the name to visualization_ipf.
##################################
import os
import sys
import numpy as np
folder_path = os.path.dirname(os.path.abspath(__file__))
pkg_path = os.path.join(folder_path, '../..')
sys.path.append(pkg_path)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mtick

# import src.parameters as params
# from src.ipf.pedestrian_ipf import PedestrianIPF
import pickle
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels

class VisualizationMIF:

    def __init__(self, map_info, particle_num_per_intent, infos, sample_true):
        
        # map_info: contains map information that helps draw the map.
        self.map_info = map_info
        self.particle_num_per_intent = particle_num_per_intent
        # self.intention_dist_hist = intention_dist_hist

        # (correct_intention, percentage_hist, intention_dist_hist, \
        #      particle_intention_hist, long_term_pred_hist, long_term_true_hist)
        (_, _, self.intention_dist_hist, \
             self.particle_intention_hist, self.long_term_pred_hist, self.long_term_true_hist) = infos
        self.intention_dist_hist = np.vstack(self.intention_dist_hist)

        self.sample_true = sample_true


        self.fig = plt.figure(figsize=map_info['fig_size'])
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        self.axs = []
        for i in range(2):
            self.axs.append(plt.subplot(gs[i]))
        self.num_samples = self.particle_num_per_intent * self.map_info['intent_num']
        plt.tight_layout()
        self.init_plot()


    def init_plot(self):

        # plot intents

        for i in range(self.map_info['intent_num']):

            # self.axs[0].plot(self.map_info['intent_mean'][i, 0], self.map_info['intent_mean'][i, 1], \
            #     c=self.map_info['intent_color'][i], marker='.', ms = self.map_info['intent_ms'])
            intent_sq = patches.Rectangle(self.map_info['intent_mean'][i]-self.map_info['intent_radius'], \
                2.*self.map_info['intent_radius'], 2.*self.map_info['intent_radius'], linewidth=1, ls='--', edgecolor=self.map_info['intent_color'][i], facecolor='none')
            self.axs[0].add_patch(intent_sq)

        self.axs[0].set_aspect('equal', 'box')
        # self.axs[0].axis('scaled')
        self.axs[0].set(xlim=(-1, 18), ylim=(-1, 14))
        self.axs[0].set_yticks([0, 4, 8, 12])
        self.axs[0].set_xticks([0, 4, 8, 12, 16])


        # set up probability distribution plot
        self.axs[1].set_xlim([0, self.map_info['intent_num']+1]) # look symmetric
        self.axs[1].set_ylim([0, 1])
        self.axs[1].set_yticks([0, 0.5, 1])
        # ax.set_yticklabels([])
        self.axs[1].set_xticklabels([])

        self.prob_dist = self.axs[1].bar(range(1, self.map_info['intent_num']+1),\
            1. / self.map_info['intent_num'] * np.ones(self.map_info['intent_num']))
        for j, b in enumerate(self.prob_dist): # k is 0, 1, 2
                b.set_color(self.map_info['intent_color'][j])

        # set up pedestrian plot
        # self.ped_track, = self.axs[0].plot([], [], 'b*', ms=5)
        # self.ped_predicted_traj, = self.axs[0].plot([], [], 'c^', ms=5)
        

        # self.ped_predicted_traj, = self.axs[0].plot([], [], 'm^', ms=5)
        
        # self.top_weight_samples = []
        
        self.prediction_samples = []
        for i in range(self.num_samples):
            particle, = self.axs[0].plot([], [])
            self.prediction_samples.append(particle)

        # self.ground_truth_state, = self.axs[0].plot([], [], 'r')
        # self.fig.savefig('init_vis_mif.png')
        self.last_obs_point, = self.axs[0].plot([], [], '.', c='k', ms=30)
        # self.ped_track, = self.axs[0].plot([], [], 'k-')
        
        # self.ped_track, = self.axs[0].plot(sample_true_abnormal[:, 0], sample_true_abnormal[:, 1], 'k-')
        self.ped_track, = self.axs[0].plot(self.sample_true[:, 0], self.sample_true[:, 1], 'k-')


        print('Visualization is initialized.')


    def anime_intention_prob_dist(self, interval=20):

        def init():
            pass

        def update(ts):

            sample_full_pred = self.long_term_pred_hist[ts]
            x_true_remained = self.long_term_true_hist[ts]
            particle_intention = self.particle_intention_hist[ts]
            intention_dist = self.intention_dist_hist[ts]
            max_prob_intention = np.argmax(intention_dist)

            for j, b in enumerate(self.prob_dist): # k is 0, 1, 2
                b.set_height(intention_dist[j])       

            # for i in range(self.num_samples):
            #     particle, = self.axs[0].plot([], [])
            #     self.prediction_samples.append(particle)
            for i in range(self.num_samples):
                self.prediction_samples[i].set_data([], [])
            # self.ped_track.set_data([],[])


            for i, (xy_pred, intention) in enumerate(zip(sample_full_pred, particle_intention)):

                if intention == max_prob_intention:
                    xy_pred = xy_pred - xy_pred[0] + x_true_remained[0]
                    self.prediction_samples[i].set_data(xy_pred[:,0], xy_pred[:,1]) # np
                    # print(type(xy_pred))
                    # raise RuntimeError
                    self.prediction_samples[i].set_color(self.map_info['intent_color'][intention])
                    # plt.plot(xy_pred[:,0], xy_pred[:,1], c=)
            self.last_obs_point.set_data(x_true_remained[0, 0], x_true_remained[0, 1])

            # self.ped_track.set_data(sample_true_abnormal[ts+13:, 0], sample_true_abnormal[ts+13:, 1])
            # self.ped_track.set_data(sample_true_abnormal[:, 0], sample_true_abnormal[:, 1])

        # ani = FuncAnimation(self.fig, update, frames=df.shape[0],
        #                     init_func=init, interval=interval, repeat=True)
        ani = FuncAnimation(self.fig, update, frames=self.intention_dist_hist.shape[0],
                            init_func=init, interval=interval, repeat=True)
        plt.show()






def intention_hist_plot(intention_dist_hist):
    hues = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    
    intention_dist_hist = np.stack(intention_dist_hist)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(intention_dist_hist[:, 0], c=hues[0], lw=3, label='intention 0')
    ax.plot(intention_dist_hist[:, 1], c=hues[1], lw=3, label='intention 1')
    ax.plot(intention_dist_hist[:, 2], c=hues[2], lw=3, label='intention 2')
    # plt.legend()
    ax.axvline(x=30, c='k', ls='--')
    ax.axvline(x=86-12-2, c='k', ls='--')
    # plt.axvline(x=90.5, c='k', ls='--')
    ax.axvline(x=150, c='k', ls='--')

    # gt_intention = np.ones(len(intention_dist_hist))
    # gt_intention[:86-12-2] = 1.
    # gt_intention[86-12-2:] = 0.
    # ax.plot(np.arange(0, 86-12-2+1),  np.ones(86-12-2+1)*1.05,  c=hues[1], lw=2)
    # ax.plot(np.arange(86-12-2, len(intention_dist_hist)),  np.ones(len(intention_dist_hist)-(86-12-2))*1.05,  c=hues[0], lw=2)

    intent_gt_1 = patches.Rectangle((0, 1.05), 86-12-2, 1.1, facecolor=hues[1])
    ax.add_patch(intent_gt_1)
    intent_gt_2 = patches.Rectangle((86-12-2, 1.05), len(intention_dist_hist)-(86-12-2)-0.5, 1.1, facecolor=hues[0])
    ax.add_patch(intent_gt_2)

    # intent_sq = patches.Rectangle(self.map_info['intent_mean'][i]-self.map_info['intent_radius'], \
    #     2.*self.map_info['intent_radius'], 2.*self.map_info['intent_radius'], linewidth=1, ls='--', edgecolor=self.map_info['intent_color'][i], facecolor='none')
    # self.axs[0].add_patch(intent_sq)


    # plt.axhline(y=1, xmin=0, xmax=86-12-2, c=hues[0])
    # plt.axhline(y=1, xmin=86-12-2, xmax=198, c=hues[1])
    # print(len(intention_dist_hist))

    # fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # ax.yaxis.set_major_formatter(yticks)
    
    
    # plt.xlim(-5, 205)
    # plt.ylim(-0.05, 1.05)
    ax.set(xlim=(-5, 205), ylim=(-0.1, 1.1))
    # self.axs[0].set(xlim=(-1, 18), ylim=(-1, 14))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # plt.xticks([0, 4, 8, 12, 16])


    plt.savefig('fit_project_fig.pdf')


def filtering_prediction_plot(infos):
    hues = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
        
    dataset_filename = 'abnormal_3_intentions_Jun_3.p'
    with open(dataset_filename, 'rb') as f:
        abnormal_traj_data = pickle.load(f)
        print(dataset_filename+' is loaded.')

    num_abnormal_samples = 0
    for data_i in abnormal_traj_data:
        num_abnormal_samples = num_abnormal_samples + len(data_i)
    print('Number of abnormal samples: ', num_abnormal_samples)
    abnormal_traj_data_total = abnormal_traj_data[0]+abnormal_traj_data[1]+abnormal_traj_data[2]
    sample_true_abnormal = abnormal_traj_data_total[0]


    # time_idx = 150 # 30
    (correct_intention, percentage_hist, intention_dist_hist, \
            particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos
    for time_idx in [30, 150]:

        sample_full_pred = long_term_pred_hist[time_idx]
        x_true_remained = long_term_true_hist[time_idx]
        particle_intention = particle_intention_hist[time_idx]

        intention_dist = intention_dist_hist[time_idx]
        max_prob_intention = np.argmax(intention_dist)

        for xy_pred, intention in zip(sample_full_pred, particle_intention):
            if intention == max_prob_intention:
                plt.plot(xy_pred[:,0], xy_pred[:,1], c=hues[intention])

        plt.plot(x_true_remained[0, 0], x_true_remained[0, 1], '.', c='k', ms=15, label='last observation') # int
    plt.plot(sample_true_abnormal[:, 0], sample_true_abnormal[:, 1],'k-')
    # plt.plot(x_true_remained[0, 0], x_true_remained[0, 1],'r-')


    time_intention_changing_idx = 86-12-2
    x_true_remained = long_term_true_hist[time_intention_changing_idx]
    plt.plot(x_true_remained[0, 0], x_true_remained[0, 1], '.', c='k', ms=15, label='last observation') # int




    plt.xlim(-1, 18)
    plt.ylim(-1, 14)
    # plt.setxlim=(-1, 18), ylim=(-1, 14))

    # self.axs[0].set(xlim=(-1, 18), ylim=(-1, 14))
    plt.yticks([0, 4, 8, 12])
    plt.xticks([0, 4, 8, 12, 16])

    plt.tight_layout()
    plt.savefig('fit_project_fig_2.pdf')




# def main_all_traj():
#     particle_num_per_intent = 20
#     num_tpp = 6
#     mutation_on = True
    
#     # for particle_num_per_intent in [20, 200]:
#     #     for num_tpp in [6, 12]:
#     #         for mutation_on in [True, False]:
#     #             filter_config = str(particle_num_per_intent)+'_'+str(num_tpp)+'_'+str(mutation_on)
#     #             print('Started processing '+filter_config+' for all abnormal trajectory data.')
#     #             infos_total = {}
#     #             infos_total[filter_config] = []
#     #             for sample_true_abnormal in abnormal_traj_data_total:
#     #                 infos_rlstm_abnormal = IntentionFilteringV4(sample_true_abnormal, particle_num_per_intent, num_tpp, tau, pred_func=rbl_prediction, step_per_update=step_per_update, mutation_on=mutation_on, display_on=False)
#     #                 infos_total[filter_config].append(infos_rlstm_abnormal)

#     #             with open('infos_abnormal_jun_17_'+filter_config+'.p', 'wb') as f:
#     #                 pickle.dump(infos_total, f)
#     #                 print('infos_abnormal_jun_17_'+filter_config+'.p is saved.')



def main_animation():

    # initilize visualization settings
    map_info = {}
    map_info['intent_num'] = 3
    map_info['intent_mean'] = np.array([[ 7.6491633 , 11.74338086],
                                        [ 3.00575615,  0.77987421],
                                        [15.72789116,  7.75681342],])
    map_info['intent_color'] = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(map_info['intent_color'])):
        map_info['intent_color'][i] = np.array(map_info['intent_color'][i])/255.
    map_info['intent_radius'] = 0.8
    map_info['fig_size'] = (12, 7)
    particle_num_per_intent = 200

    # load filtering data
    with open('infos_abnormal_jun_16_tbd.p', 'rb') as f:
        infos_test = pickle.load(f)
        print('infos_abnormal_jun_16_tbd.p is loaded.')
        print(infos_test.keys())

    # load original full trajectory
    dataset_filename = 'abnormal_3_intentions_Jun_3.p'
    with open(os.path.join(dataset_filename), 'rb') as f:
        abnormal_traj_data = pickle.load(f)
        print(dataset_filename+' is loaded.')
    sample_true_abnormal = abnormal_traj_data[0][0]
    # print(len(sample_true_abnormal))
    
    # (correct_intention, percentage_hist, intention_dist_hist, \
    #          particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos_test['200_12_True']

    # print(len(percentage_hist))
    # print(len(intention_dist_hist))
    # print(len(particle_intention_hist))
    # print(len(long_term_pred_hist))
    # print(len(long_term_true_hist))

    vis_mif = VisualizationMIF(map_info, particle_num_per_intent, infos_test['200_12_True'], sample_true_abnormal)

    vis_mif.anime_intention_prob_dist()

    # intention_dist_hist = np.stack(intention_dist_hist)
    # plt.plot(intention_dist_hist[:, 0], c=hues[0], label='intention 0')
    # plt.plot(intention_dist_hist[:, 1], c=hues[1], label='intention 1')
    # plt.plot(intention_dist_hist[:, 2], c=hues[2], label='intention 2')

def main_plot():

    with open('infos_abnormal_jun_16_tbd.p', 'rb') as f:
        infos_test = pickle.load(f)
        print('infos_abnormal_jun_16_tbd.p is loaded.')
        print(infos_test.keys())
    
    # filtering_prediction_plot(infos_test['200_12_True'])
    filtering_prediction_plot(infos_test['200_12_False'])




    # (correct_intention, percentage_hist, intention_dist_hist, \
    #     particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos_test['200_12_True']
    (correct_intention, percentage_hist, intention_dist_hist, \
        particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos_test['200_12_False']
    
    intention_dist_hist = np.vstack(intention_dist_hist)
    intention_hist_plot(intention_dist_hist)




    # tmp = np.argmax(intention_dist_hist, axis=1)





    # print(tmp)
    # cache = tmp[0]
    # for idx, i in enumerate(tmp[1:]):
    #     if cache == 1 and i == 0:
    #         print(idx-1)
    #     cache = i




if __name__ == '__main__':
    main_animation()
    # main_plot()






