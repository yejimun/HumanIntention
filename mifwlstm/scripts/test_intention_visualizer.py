import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np
def talker():
    pub = rospy.Publisher('hri/human_intention_prob_dist', Float32MultiArray, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(20) # 10hz
    while not rospy.is_shutdown():
        # prob_dist = np.random.rand(2)
        # prob_dist /= prob_dist.sum()
        # msg = Float32MultiArray()
        # msg.data = list(prob_dist.astype(np.float32))
        msg = Float32MultiArray()
        msg.data = [0.7, 0.3]

        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass