#!/usr/bin/env python
r"""Visualize actual and target currents of one joint on ur robot in real time."""
import sys
import argparse
import rospy
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray
import numpy as np

class IntentionVisualizer: 
    def __init__(self):
        # !!! Start here
        self.fig, self.ax = plt.subplots()
        plt.tight_layout()
        self.num_intentions = 2
        self.colors = ['C0', 'C1', 'r']
        self.prob_threshold = 0.9
        # set up probability distribution plot
        self.ax.set_xlim([0, self.num_intentions+1]) # look symmetric
        self.ax.set_ylim([0, 1])
        self.ax.set_yticks([0, 0.5, 1])
        self.ax.set_xticks(range(1, self.num_intentions+1))
        labels = ['L', 'R']
        self.ax.set_xticklabels(labels)
        self.prob_dist = self.ax.bar(range(1, self.num_intentions+1),\
            1. / self.num_intentions * np.ones(self.num_intentions))
        for j, b in enumerate(self.prob_dist):
            b.set_color(self.colors[j])
        rospy.init_node('intention_visualizer', anonymous=True)
        rospy.Subscriber("hri/human_intention_prob_dist", Float32MultiArray, self.callback_realtime_plot, queue_size=1)
    

    def callback_realtime_plot(self, data):
        intention_prob_dist = data.data
        for j, b in enumerate(self.prob_dist):
            b.set_height(intention_prob_dist[j])
            if intention_prob_dist[j] < self.prob_threshold:
                b.set_color(self.colors[j])
            else:
                b.set_color(self.colors[self.num_intentions])
        plt.draw()


if __name__ == '__main__':
    iv = IntentionVisualizer()
    plt.show()
    rospy.spin()