#!/usr/bin/env python3


"""
Start ROS node to publish angles for the position control of the xArm7.
"""

import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kinematics import xArm7_kinematics

class xArm7_controller():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()
        # joints' angular positions
        self.joint_angpos = [0, 0, 0, 0, 0, 0, 0]
        # joints' angular velocities
        self.joint_angvel = [0, 0, 0, 0, 0, 0, 0]
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)
        # gazebo model's states
        self.model_states = ModelStates()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber('/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher('/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher('/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher('/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher('/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher('/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher('/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher('/xarm/joint7_position_controller/command', Float64, queue_size=1)

        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def model_states_callback(self, msg):
        # ROS callback to get the gazebo's model_states

        self.model_states = msg
        # (e.g. #1 the position in y-axis of GREEN obstacle's center is stored in :: self.model_states.pose[1].position.y)
        # (e.g. #2 the position in y-axis of RED obstacle's center is stored in :: self.model_states.pose[2].position.y)

    def coeff(self, t0, tf, yA, yB, g):
        t1 = t0 + (tf - t0) * 0.1
        t2 = t0 + (tf - t0) * 0.9
        L0=np.array([1,t0,t0**2,t0**3,t0**4,t0**5,0,0,0,0,0,0,0,0])
        L1=np.array([0,1,2*t0,3*t0**2,4*t0**3,5*t0**4,0,0,0,0,0,0,0,0])
        L2=np.array([0,0,2,6*t0,12*t0**2,20*t0**3,0,0,0,0,0,0,0,0])
        L3=np.array([1,t1,t1**2,t1**3,t1**4,t1**5,-1,-t1,0,0,0,0,0,0])
        L4=np.array([0,1,2*t1,3*t1**2,4*t1**3,5*t1**4,0,-1,0,0,0,0,0,0])
        L5=np.array([0,0,2,6*t1,12*t1**2,20*t1**3,0,0,0,0,0,0,0,0])
        L6=np.array([0,0,0,0,0,0,-1,-t2,1,t2,t2**2,t2**3,t2**4,t2**5])
        L7=np.array([0,0,0,0,0,0,0,-1,0,1,2*t2,3*t2**2,4*t2**3,5*t2**4])
        L8=np.array([0,0,0,0,0,0,0,0,0,0,2,6*t2,12*t2**2,20*t2**3])
        L9=np.array([0,0,0,0,0,0,0,0,1,tf,tf**2,tf**3,tf**4,tf**5])
        L10=np.array([0,0,0,0,0,0,0,0,0,1,2*tf,3*tf**2,4*tf**3,5*tf**4])
        L11=np.array([0,0,0,0,0,0,0,0,0,0,2,6*tf,12*tf**2,20*tf**3])
        L12=np.array([-1,-t0,-t0**2,-t0**3,-t0**4,-t0**5,1,0.5*t0 + 0.5*t1,0,0,0,0,0,0])
        L13=np.array([0,0,0,0,0,0,-1,0.5*t0 - 0.5*t1 - t2,1,tf,tf**2,tf**3,tf**4,tf**5])
        L=[L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13]

        b = np.array([yA, 0, g, 0, 0, 0, 0, 0, 0, yB, 0, -g, 0, 0])
        params = np.linalg.inv(L).dot(b)
        return params

    def calc_pva(self,time,A_to_B,period,y_A,y_B):
        #what time the last period started (0 or 1)
        period_start=np.floor(time/period)
        #what time it is based on the A->B->A movement ( 0 to 2)
        if(period_start%2==0):
            _temp=0
        else:
            _temp=period
        period_time=time-period_start+_temp

        if period_time<period :
            #A->B
            y_now=y_A
            y_end=y_B
            moves_to_B=1
            A_to_Bb=False
        else:
            #B->A
            y_now=y_B
            y_end=y_A
            moves_to_B=-1
            A_to_Bb=True
        #check if we are just begining the motion
        if time<period:
            #0->B
            whole_period=0
            y_now=0
        else:
            #A->B or B->A
            whole_period=1
        #find what params[] will stay
        h=[0,0,0]
        if period_time%1<period*0.1:
            h[0]=1
        elif period_time%1<period*0.9:
            h[1]=1
        else:
            h[2]=1
        end_period=8*h[2]
        params = self.coeff(whole_period*period_start,whole_period*period_start+ period, y_now, y_end, moves_to_B)
        pos_y =  h[1]*(params[6]+params[7]* time)+  (h[0]+h[2])*(params[end_period+0] + params[end_period+1] * time + params[end_period+2] * time ** 2 + params[end_period+3] * time**3 + params[end_period+4] * time ** 4 + params[end_period+5] * time ** 5)
        vel_y =  h[1]*(params[7])+                  (h[0]+h[2])*(params[end_period+1] + 2 * params[end_period+2] * time + 3 * params[end_period+3] * time**2 + 4 * params[end_period+4] * time ** 3 + 5 * params[end_period+5] * time ** 4)
        acc_y =  0+                                 (h[0]+h[2])*(2 * params[end_period+2] + 6 * params[end_period+3] * time + 12 * params[end_period+4] * time ** 2 + 20 * params[end_period+5] * time ** 3)
        
        
        return pos_y,vel_y,acc_y,A_to_Bb

    def publish(self):

        # set configuration
        # total pitch: j2-j4+j6+pi (upwards: 0rad)
        j2 = 0.7 ; j4 = np.pi/2
        j6 = - (j2-j4)
        self.joint_angpos = [0, j2, 0, j4, 0, j6, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        #Initialization
        rostime_now = rospy.get_rostime()
        time_now = rostime_now.to_nsec()

        x_A = 0.617
        x_B = 0.617
        y_A = -0.2
        y_B = 0.2
        z_A = 0.199
        z_B = 0.199
        self.x_OBSTACLE1 = 0.3
        self.y_OBSTACLE1 = 0.2
        self.x_OBSTACLE2 = 0.3
        self.y_OBSTACLE2 = -0.2
        rostime_starttime = rospy.get_rostime()
        starttime = rostime_starttime.to_nsec()
        K1 = 150
        K2 = 10
        K3 = 15
        Kc = 10

        p_prev = np.zeros(3)

        x_error_values = []
        y_error_values = []
        z_error_values = []
        y_distance_joint3_green=[]
        y_distance_joint3_red=[]
        y_distance_joint4_green=[]
        y_distance_joint4_red=[]
        distance_joint3_green=[]
        distance_joint3_red=[]
        distance_joint4_green=[]
        distance_joint4_red=[]
        
        vel_previous=0
        count=-1

        dev=False

        pos_z = z_A
        vel_z = 0
        acc_z = 0

        pos_x = x_A
        vel_x = 0
        acc_x = 0  
        
        period = 1

        while not rospy.is_shutdown():
            count+=1
            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A01 = self.kinematics.tf_A01(self.joint_angpos)
            self.A02 = self.kinematics.tf_A02(self.joint_angpos)
            self.A03 = self.kinematics.tf_A03(self.joint_angpos)
            self.A04 = self.kinematics.tf_A04(self.joint_angpos)
            self.A05 = self.kinematics.tf_A05(self.joint_angpos)
            self.A06 = self.kinematics.tf_A06(self.joint_angpos)
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)

            self.l2 = self.kinematics.l2            
            self.l3 = self.kinematics.l3
            self.l4 = self.kinematics.l4
            self.theta1 = self.kinematics.theta1

            s1 = np.sin(self.joint_angpos[0])
            c1 = np.cos(self.joint_angpos[0])
            s2 = np.sin(self.joint_angpos[1])
            c2 = np.cos(self.joint_angpos[1])
            s3 = np.sin(self.joint_angpos[2])
            c3 = np.cos(self.joint_angpos[2])
            s4 = np.sin(self.joint_angpos[3])
            c4 = np.cos(self.joint_angpos[3])
            s5 = np.sin(self.joint_angpos[4])
            c5 = np.cos(self.joint_angpos[4])


            #task 1          

            time =  (time_now - starttime)/1e9
            A_to_B = True 

            pos_y , vel_y , acc_y ,A_to_B= self.calc_pva(time,A_to_B,period,y_A,y_B)

            if vel_previous*vel_y<0  :
                print("Reached Pa / Pb and the x position is: "+str(pos_x))
                print("Reached Pa / Pb and the y position is: "+str(pos_y))
                print("Reached Pa / Pb and the z position is: "+str(pos_z))
                print("====================================================")
            vel_previous=vel_y

            pos = np.matrix([[pos_x],[pos_y],[pos_z]])

            vel = np.matrix([[vel_x],[vel_y],[vel_z]])
            
            pos_real = np.matrix([[self.A07[0, 3]],[self.A07[1, 3]],[self.A07[2, 3]]])

            # Compute jacobian matrix
            J = self.kinematics.compute_jacobian(self.joint_angpos)
            # pseudoinverse jacobian
            pinvJ = pinv(J)


            self.task1 = np.dot(pinvJ,K1*(pos- pos_real)+vel)

            #task 2

            middle_obstacle = (self.model_states.pose[1].position.y + self.model_states.pose[2].position.y) / 2

            if A_to_B:
            	middle_obstacle = middle_obstacle + 0.075
            else:
            	middle_obstacle = middle_obstacle - 0.05

            self.y_OBSTACLE2 = self.model_states.pose[1].position.y
            self.y_OBSTACLE1 = self.model_states.pose[2].position.y
            self.x_OBSTACLE2 = self.model_states.pose[1].position.x
            self.x_OBSTACLE1 = self.model_states.pose[2].position.x

            dist3 = (1/2) * Kc * ((self.A03[1,3] - middle_obstacle) ** 2)
            dist4 = (1/2) * Kc * ((self.A04[1,3] - middle_obstacle) ** 2)
            
            dist3_dot = np.zeros((7,1))
            dist3_dot[0] = -Kc * (self.A03[1,3] - self.y_OBSTACLE2) * self.l2 * c1 * s2
            dist3_dot[1] = -Kc * (self.A03[1,3] - self.y_OBSTACLE2) * self.l2 * c2 * s1                

            dist4_dot = np.zeros((7,1))
            dist4_dot[0] = -Kc * (self.A04[1,3] - self.y_OBSTACLE1) * (self.l2 * c1 * s2 - self.l3 * (s1 * s3 - c1 * c2 * c3))
            dist4_dot[1] = -Kc * (self.A04[1,3] - self.y_OBSTACLE1) * (self.l2 * c2 * s1 - self.l3 * c3 * s1 * s2)
            dist4_dot[2] = -Kc * (self.A04[1,3] - self.y_OBSTACLE1) * (self.l3 * (c1 * c3 - c2 * s1 * s3))

            I = np.eye(7)

            self.task2 =  np.dot( I - np.dot(pinvJ, J) , K2 * dist3_dot + K3 * dist4_dot) 

            maximum = max(dist3, dist4)
            if (maximum >= 0.03):
            	for i in range(7):
            		self.joint_angvel[i] = self.task1[i, 0] + self.task2[i, 0]
            else:
            	for i in range(7):
            		self.joint_angvel[i] = self.task1[i, 0]


            # Convertion to angular position after integrating the angular speed in time
            # Calculate time interval
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9

            # Integration
            self.joint_angpos = np.add( self.joint_angpos, [index * dt for index in self.joint_angvel] )


            # Calculate the error values
            x_error = self.A07[0, 3] - pos_x
            y_error = self.A07[1, 3] - pos_y
            z_error = self.A07[2, 3] - pos_z

            # Append the error values to the lists
            x_error_values.append(x_error)
            y_error_values.append(y_error)
            z_error_values.append(z_error)
            y_distance_joint3_green.append(self.A03[1, 3] - self.y_OBSTACLE2)
            y_distance_joint3_red.append(self.A03[1, 3] - self.y_OBSTACLE1)
            y_distance_joint4_green.append(self.A04[1, 3] - self.y_OBSTACLE2)
            y_distance_joint4_red.append(self.A04[1, 3] - self.y_OBSTACLE1)
            distance_joint3_green.append(np.sqrt((self.A03[0, 3] - self.y_OBSTACLE2) ** 2 + (self.A03[1, 3] - self.x_OBSTACLE2) ** 2))
            distance_joint3_red.append(np.sqrt((self.A03[0, 3] - self.y_OBSTACLE1) ** 2 + (self.A03[1, 3] - self.x_OBSTACLE1) ** 2))
            distance_joint4_green.append(np.sqrt((self.A04[0, 3] - self.y_OBSTACLE2) ** 2 + (self.A04[1, 3] - self.x_OBSTACLE2) ** 2))
            distance_joint4_red.append(np.sqrt((self.A04[0, 3] - self.y_OBSTACLE1) ** 2 + (self.A04[1, 3] - self.x_OBSTACLE1) ** 2))
            #this part was added to graph important values and find the best possible values for Kc,K1,K2,K3
            if dev and count==7000:
                data = np.column_stack((x_error_values, y_error_values, z_error_values,\
                                        y_distance_joint3_green,y_distance_joint3_red,y_distance_joint4_green,y_distance_joint4_red,\
                                            distance_joint3_green,distance_joint3_red,distance_joint4_green,distance_joint4_red))
                np.savetxt('error_values.txt', data, delimiter=',')
                print("Saved Values")

            # Publish the new joint's angular positions
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])

            self.pub_rate.sleep()

    def turn_off(self):
        pass

def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass
