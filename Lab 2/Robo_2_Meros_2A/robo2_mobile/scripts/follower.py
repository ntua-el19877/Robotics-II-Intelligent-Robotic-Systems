#!/usr/bin/env python3

"""
Team 50
2023-06-02
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# Ros handlers services and messages
import math
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t
from nav_msgs.msg import Odometry


def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class mymobibot_follower():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # linear and angular velocity
        self.velocity = Twist()
        # joints' states
        self.joint_states = JointState()
        # Sensors
        self.imu = Imu()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.velocity_pub = rospy.Publisher('/mymobibot/cmd_vel', Twist, queue_size=1)
        self.joint_states_sub = rospy.Subscriber('/mymobibot/joint_states', JointState, self.joint_states_callback, queue_size=1)
        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)
        self.odom = Odometry()
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)


        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of the left wheel is stored in :: self.joint_states.position[0])
        # (e.g. the angular velocity of the right wheel is stored in :: self.joint_states.velocity[1])

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        # (e.g. the orientation of the robot wrt the global frome is stored in :: self.imu.orientation)
        # (e.g. the angular velocity of the robot wrt its frome is stored in :: self.imu.angular_velocity)
        # (e.g. the linear acceleration of the robot wrt its frome is stored in :: self.imu.linear_acceleration)

        #quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        #(roll, pitch, self.imu_yaw) = euler_from_quaternion(quaternion)
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        # (e.g. the distance from sonar_front to an obstacle is stored in :: self.sonar_F.range)

    def sonar_frontleft_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FL

        self.sonar_FL = msg
        # (e.g. the distance from sonar_frontleft to an obstacle is stored in :: self.sonar_FL.range)

    def sonar_frontright_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FR

        self.sonar_FR = msg
        # (e.g. the distance from sonar_frontright to an obstacle is stored in :: self.sonar_FR.range)

    def sonar_left_callback(self, msg):
        # ROS callback to get the /sensor/sonar_L

        self.sonar_L = msg
        # (e.g. the distance from sonar_left to an obstacle is stored in :: self.sonar_L.range)

    def sonar_right_callback(self, msg):
        # ROS callback to get the /sensor/sonar_R

        self.sonar_R = msg
        # (e.g. the distance from sonar_right to an obstacle is stored in :: self.sonar_R.range)
    
    def odom_callback(self, msg):
        # ROS callback to get the /odom

        self.odom = msg
        # (e.g., the position of the robot in x-axis is stored in :: self.odom.pose.pose.position.x)
        # (e.g., the position of the robot in y-axis is stored in :: self.odom.pose.pose.position.y)
        # (e.g., the orientation of the robot is stored in :: self.odom.pose.pose.orientation)
        # (e.g., the linear velocity of the robot in x-axis is stored in :: self.odom.twist.twist.linear.x)
        # (e.g., the angular velocity of the robot in z-axis is stored in :: self.odom.twist.twist.angular.z)
    
    def publish(self):

        #
        # used to make the graphs
        # outputFile="/home/angelosros/catkin_ws/src/robo2_mobile/scripts/DataFile.txt"
        # with open(outputFile, "w", encoding="utf-8") as file:
        #     pass
        #

        # set configuration
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()

        #disired distances
        R_dist=0.3
        FR_dist=(R_dist+0.018)*math.sqrt(2)


        # Gains of PD Controller

        K1=0.1
        Kp = 20
        Kd = 10

        Straight=True

        count=-1
        prev_ang_vel=0.0
        Base_distance=0.2

        #init for first run
        previous_FR_error = FR_error=0
        previous_R_error = R_error=0
        while not rospy.is_shutdown():
            count+=1
            if not Straight:
                #Turning motion
                
                self.velocity.linear.x = 0.1
                self.velocity.angular.z = (K1*prev_ang_vel-2*np.pi)/2 
                prev_ang_vel=-2*np.pi
                # 
                if ((self.sonar_R.range + Base_distance < self.sonar_F.range) and 
                    (self.sonar_FR.range + Base_distance< self.sonar_F.range)):
                    FR_error = 0
                    R_error = 0
                    Straight=True
            else: 
                #Straight motion

                #initial velocities  
                if self.sonar_F.range>0.4 and self.sonar_R.range>0.5:
                    self.velocity.linear.x = 0.5
                    self.velocity.angular.z = -0.6
                else:
                    # Proportional Errors
                    FR_error = FR_dist - self.sonar_FR.range
                    R_error = R_dist- self.sonar_R.range
                    P_error = FR_error + R_error
                    # Derivative Errors
                    Derivative_FR_error = FR_error - previous_FR_error
                    Derivative_R_error = R_error - previous_R_error
                    try:
                        D_error = (Derivative_FR_error + Derivative_R_error) / dt
                    except:
                        dt=0.0001
                        D_error = (Derivative_FR_error + Derivative_R_error) / dt

                    self.velocity.linear.x = 0.5 
                    self.velocity.angular.z = -(Kp*P_error+Kd*D_error+K1*prev_ang_vel)/2
                    prev_ang_vel=-Kp*P_error+Kd*D_error

                    # turn if wall is close by
                    if (((self.sonar_R.range + Base_distance >= self.sonar_F.range) 
                        or (self.sonar_FR.range + Base_distance >= self.sonar_F.range))
                        and self.sonar_R.range<1):
                        Straight=False
            if Straight:
                #update error values for next iteration 
                previous_FR_error = FR_error
                previous_R_error = R_error

            # Calculate time interval (in case is needed)
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9

            #
            # used to make the graphs
            # if count<3000:
            #     with open(outputFile, "a", encoding="utf-8") as file:
            #         file.write(str(self.sonar_R.range)+','+str(self.sonar_FR.range)+','+\
            #                    str(self.sonar_F.range)+','+\
            #                    str(self.imu.angular_velocity.z)+','+str(self.velocity.linear.x)+',')
            # elif count ==3000:
            #     print("Done")
            #

            # Publish the new joint's angular positions
            self.velocity_pub.publish(self.velocity)

            self.pub_rate.sleep()
            

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('follower_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_follower(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass
