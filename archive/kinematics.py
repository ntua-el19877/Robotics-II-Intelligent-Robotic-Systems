#!/usr/bin/env python3

"""
Compute state space dynamic and kinematic matrices for widowx robot arm (3 links)
"""

#Math stuff
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
from geometry_msgs.msg import Quaternion, Pose, PoseArray
from tf.transformations import quaternion_from_euler


class xArm7_kinematics():
    def __init__(self):

        """ DENAVIT-HARTENBERG CONVENTION """
        self.dh = []
        ''' dh_param = ['theta', 'd', 'a', 'alpha'] # (in radians) '''
        # fill in the dh-table considering that each joint is rotated by 0 rads wrt its initial position (e.g. q1=...=q7=0)
        ## JOINT 1
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 2
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 3
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 4
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 5
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 6
        self.dh.append([ 0 , 0 , 0 , 0 ])
        ## JOINT 7
        self.dh.append([ 0 , 0 , 0 , 0 ])

        pass

    def compute_jacobian(self, r_joints_array):
        q1=(r_joints_array[0])
        q2=(r_joints_array[1])
        q3=(r_joints_array[2])
        q4=(r_joints_array[3])
        q5=(r_joints_array[4])
        q6=(r_joints_array[5])
        q7=(r_joints_array[6])


        J_11 = -9.6978548845024*(((-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*cos(q4) - sin(q1)*sin(q2)*sin(q4))*cos(q5) + (-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(q5))*sin(q6) + 7.598289981128*(((-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*cos(q4) - sin(q1)*sin(q2)*sin(q4))*cos(q5) + (-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(q5))*cos(q6) + 7.598289981128*((-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*sin(q4) + sin(q1)*sin(q2)*cos(q4))*sin(q6) + 9.6978548845024*((-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*sin(q4) + sin(q1)*sin(q2)*cos(q4))*cos(q6) + 34.2534756148792*(-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*sin(q4) + 7.7533095048664*(-sin(q1)*cos(q2)*cos(q3) - sin(q3)*cos(q1))*cos(q4) - 7.7533095048664*sin(q1)*sin(q2)*sin(q4) + 34.2534756148792*sin(q1)*sin(q2)*cos(q4) - 29.3*sin(q1)*sin(q2) - 5.25*sin(q1)*cos(q2)*cos(q3) - 5.25*sin(q3)*cos(q1)
        J_12 = -9.6978548845024*((-sin(q2)*cos(q1)*cos(q3)*cos(q4) + sin(q4)*cos(q1)*cos(q2))*cos(q5) - sin(q2)*sin(q3)*sin(q5)*cos(q1))*sin(q6) + 7.598289981128*((-sin(q2)*cos(q1)*cos(q3)*cos(q4) + sin(q4)*cos(q1)*cos(q2))*cos(q5) - sin(q2)*sin(q3)*sin(q5)*cos(q1))*cos(q6) + 7.598289981128*(-sin(q2)*sin(q4)*cos(q1)*cos(q3) - cos(q1)*cos(q2)*cos(q4))*sin(q6) + 9.6978548845024*(-sin(q2)*sin(q4)*cos(q1)*cos(q3) - cos(q1)*cos(q2)*cos(q4))*cos(q6) - 34.2534756148792*sin(q2)*sin(q4)*cos(q1)*cos(q3) - 7.7533095048664*sin(q2)*cos(q1)*cos(q3)*cos(q4) - 5.25*sin(q2)*cos(q1)*cos(q3) + 7.7533095048664*sin(q4)*cos(q1)*cos(q2) - 34.2534756148792*cos(q1)*cos(q2)*cos(q4) + 29.3*cos(q1)*cos(q2)
        J_13 = -9.6978548845024*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q5) + (-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*cos(q4)*cos(q5))*sin(q6) + 7.598289981128*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q5) + (-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*cos(q4)*cos(q5))*cos(q6) + 7.598289981128*(-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*sin(q4)*sin(q6) + 9.6978548845024*(-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*sin(q4)*cos(q6) + 34.2534756148792*(-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*sin(q4) + 7.7533095048664*(-sin(q1)*cos(q3) - sin(q3)*cos(q1)*cos(q2))*cos(q4) - 5.25*sin(q1)*cos(q3) - 5.25*sin(q3)*cos(q1)*cos(q2)
        J_14 = -9.6978548845024*(-(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + sin(q2)*cos(q1)*cos(q4))*sin(q6)*cos(q5) + 7.598289981128*(-(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + sin(q2)*cos(q1)*cos(q4))*cos(q5)*cos(q6) + 7.598289981128*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*sin(q6) + 9.6978548845024*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q6) - 7.7533095048664*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 34.2534756148792*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + 34.2534756148792*sin(q2)*sin(q4)*cos(q1) + 7.7533095048664*sin(q2)*cos(q1)*cos(q4)
        J_15 = -9.6978548845024*(-((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*sin(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*cos(q5))*sin(q6) + 7.598289981128*(-((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*sin(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*cos(q5))*cos(q6)
        J_16 = -7.598289981128*(((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*sin(q5))*sin(q6) - 9.6978548845024*(((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*sin(q5))*cos(q6) - 9.6978548845024*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4))*sin(q6) + 7.598289981128*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4))*cos(q6)
        J_17 = 0

        J_21 = -9.6978548845024*(((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*sin(q5))*sin(q6) + 7.598289981128*(((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*sin(q5))*cos(q6) + 7.598289981128*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4))*sin(q6) + 9.6978548845024*((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4))*cos(q6) + 34.2534756148792*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) + 7.7533095048664*(-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) - 5.25*sin(q1)*sin(q3) + 7.7533095048664*sin(q2)*sin(q4)*cos(q1) - 34.2534756148792*sin(q2)*cos(q1)*cos(q4) + 29.3*sin(q2)*cos(q1) + 5.25*cos(q1)*cos(q2)*cos(q3)
        J_22 = -9.6978548845024*((-sin(q1)*sin(q2)*cos(q3)*cos(q4) + sin(q1)*sin(q4)*cos(q2))*cos(q5) - sin(q1)*sin(q2)*sin(q3)*sin(q5))*sin(q6) + 7.598289981128*((-sin(q1)*sin(q2)*cos(q3)*cos(q4) + sin(q1)*sin(q4)*cos(q2))*cos(q5) - sin(q1)*sin(q2)*sin(q3)*sin(q5))*cos(q6) + 7.598289981128*(-sin(q1)*sin(q2)*sin(q4)*cos(q3) - sin(q1)*cos(q2)*cos(q4))*sin(q6) + 9.6978548845024*(-sin(q1)*sin(q2)*sin(q4)*cos(q3) - sin(q1)*cos(q2)*cos(q4))*cos(q6) - 34.2534756148792*sin(q1)*sin(q2)*sin(q4)*cos(q3) - 7.7533095048664*sin(q1)*sin(q2)*cos(q3)*cos(q4) - 5.25*sin(q1)*sin(q2)*cos(q3) + 7.7533095048664*sin(q1)*sin(q4)*cos(q2) - 34.2534756148792*sin(q1)*cos(q2)*cos(q4) + 29.3*sin(q1)*cos(q2)
        J_23 = -9.6978548845024*((-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*cos(q4)*cos(q5) + (sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q5))*sin(q6) + 7.598289981128*((-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*cos(q4)*cos(q5) + (sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q5))*cos(q6) + 7.598289981128*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(q4)*sin(q6) + 9.6978548845024*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(q4)*cos(q6) + 34.2534756148792*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*sin(q4) + 7.7533095048664*(-sin(q1)*sin(q3)*cos(q2) + cos(q1)*cos(q3))*cos(q4) - 5.25*sin(q1)*sin(q3)*cos(q2) + 5.25*cos(q1)*cos(q3)
        J_24 = -9.6978548845024*(-(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + sin(q1)*sin(q2)*cos(q4))*sin(q6)*cos(q5) + 7.598289981128*(-(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + sin(q1)*sin(q2)*cos(q4))*cos(q5)*cos(q6) + 7.598289981128*((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*sin(q6) + 9.6978548845024*((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*cos(q6) - 7.7533095048664*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) + 34.2534756148792*(sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + 34.2534756148792*sin(q1)*sin(q2)*sin(q4) + 7.7533095048664*sin(q1)*sin(q2)*cos(q4)
        J_25 = -9.6978548845024*(-((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*sin(q5) + (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*cos(q5))*sin(q6) + 7.598289981128*(-((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*sin(q5) + (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*cos(q5))*cos(q6)
        J_26 = -7.598289981128*(((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*cos(q5) + (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*sin(q5))*sin(q6) - 9.6978548845024*(((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*cos(q5) + (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*sin(q5))*cos(q6) - 9.6978548845024*((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) - sin(q1)*sin(q2)*cos(q4))*sin(q6) + 7.598289981128*((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) - sin(q1)*sin(q2)*cos(q4))*cos(q6)
        J_27 = 0

        J_31 = 0
        J_32 = -9.6978548845024*((-sin(q2)*sin(q4) - cos(q2)*cos(q3)*cos(q4))*cos(q5) - sin(q3)*sin(q5)*cos(q2))*sin(q6) + 7.598289981128*((-sin(q2)*sin(q4) - cos(q2)*cos(q3)*cos(q4))*cos(q5) - sin(q3)*sin(q5)*cos(q2))*cos(q6) + 7.598289981128*(sin(q2)*cos(q4) - sin(q4)*cos(q2)*cos(q3))*sin(q6) + 9.6978548845024*(sin(q2)*cos(q4) - sin(q4)*cos(q2)*cos(q3))*cos(q6) - 7.7533095048664*sin(q2)*sin(q4) + 34.2534756148792*sin(q2)*cos(q4) - 29.3*sin(q2) - 34.2534756148792*sin(q4)*cos(q2)*cos(q3) - 7.7533095048664*cos(q2)*cos(q3)*cos(q4) - 5.25*cos(q2)*cos(q3)
        J_33 = -9.6978548845024*(sin(q2)*sin(q3)*cos(q4)*cos(q5) - sin(q2)*sin(q5)*cos(q3))*sin(q6) + 7.598289981128*(sin(q2)*sin(q3)*cos(q4)*cos(q5) - sin(q2)*sin(q5)*cos(q3))*cos(q6) + 7.598289981128*sin(q2)*sin(q3)*sin(q4)*sin(q6) + 9.6978548845024*sin(q2)*sin(q3)*sin(q4)*cos(q6) + 34.2534756148792*sin(q2)*sin(q3)*sin(q4) + 7.7533095048664*sin(q2)*sin(q3)*cos(q4) + 5.25*sin(q2)*sin(q3)
        J_34 = -9.6978548845024*(sin(q2)*sin(q4)*cos(q3) + cos(q2)*cos(q4))*sin(q6)*cos(q5) + 7.598289981128*(sin(q2)*sin(q4)*cos(q3) + cos(q2)*cos(q4))*cos(q5)*cos(q6) + 7.598289981128*(-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*sin(q6) + 9.6978548845024*(-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*cos(q6) + 7.7533095048664*sin(q2)*sin(q4)*cos(q3) - 34.2534756148792*sin(q2)*cos(q3)*cos(q4) + 34.2534756148792*sin(q4)*cos(q2) + 7.7533095048664*cos(q2)*cos(q4)
        J_35 = -9.6978548845024*(-(-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*sin(q5) - sin(q2)*sin(q3)*cos(q5))*sin(q6) + 7.598289981128*(-(-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*sin(q5) - sin(q2)*sin(q3)*cos(q5))*cos(q6)
        J_36 = -7.598289981128*((-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*cos(q5) - sin(q2)*sin(q3)*sin(q5))*sin(q6) - 9.6978548845024*((-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*cos(q5) - sin(q2)*sin(q3)*sin(q5))*cos(q6) - 9.6978548845024*(-sin(q2)*sin(q4)*cos(q3) - cos(q2)*cos(q4))*sin(q6) + 7.598289981128*(-sin(q2)*sin(q4)*cos(q3) - cos(q2)*cos(q4))*cos(q6)
        J_37 = 0


        J_41 = 0
        J_42 = -sin(q1)
        J_43 = sin(q2)*cos(q1)
        J_44 = sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2)
        J_45 = (-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4)
        J_46 = ((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*sin(q5) - (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*cos(q5)
        J_47 = -(((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*cos(q4) + sin(q2)*sin(q4)*cos(q1))*cos(q5) + (sin(q1)*cos(q3) + sin(q3)*cos(q1)*cos(q2))*sin(q5))*sin(q6) + ((-sin(q1)*sin(q3) + cos(q1)*cos(q2)*cos(q3))*sin(q4) - sin(q2)*cos(q1)*cos(q4))*cos(q6)

        J_51 = 0
        J_52 = cos(q1)
        J_53 = sin(q1)*sin(q2)
        J_54 = sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3)
        J_55 = (sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) - sin(q1)*sin(q2)*cos(q4)
        J_56 = ((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*sin(q5) - (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*cos(q5)
        J_57 = -(((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*cos(q4) + sin(q1)*sin(q2)*sin(q4))*cos(q5) + (sin(q1)*sin(q3)*cos(q2) - cos(q1)*cos(q3))*sin(q5))*sin(q6) + ((sin(q1)*cos(q2)*cos(q3) + sin(q3)*cos(q1))*sin(q4) - sin(q1)*sin(q2)*cos(q4))*cos(q6)

        J_61 = 1
        J_62 = 0
        J_63 = cos(q2)
        J_64 = -sin(q2)*sin(q3)
        J_65 = -sin(q2)*sin(q4)*cos(q3) - cos(q2)*cos(q4)
        J_66 = (-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*sin(q5) + sin(q2)*sin(q3)*cos(q5)
        J_67 = -((-sin(q2)*cos(q3)*cos(q4) + sin(q4)*cos(q2))*cos(q5) - sin(q2)*sin(q3)*sin(q5))*sin(q6) + (-sin(q2)*sin(q4)*cos(q3) - cos(q2)*cos(q4))*cos(q6)

        J = np.matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],\
                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],\
                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ],\
                        [ J_41 , J_42 , J_43 , J_44 , J_45 , J_46 , J_47 ],\
                        [ J_51 , J_52 , J_53 , J_54 , J_55 , J_56 , J_57 ],\
                        [ J_61 , J_62 , J_63 , J_64 , J_65 , J_66 , J_67 ]])

        return J

    def compute_transformation_matrix(dh_param, r_joint):
        theta = dh_param[0] + r_joint
        d = dh_param[1]
        a = dh_param[2]
        alpha = dh_param[3]

        tf_mat = np.matrix([[cos(theta) , -sin(theta)*cos(alpha) , sin(theta)*sin(alpha) , a*cos(theta)],\
                            [sin(theta) , cos(theta)*cos(alpha) , -cos(theta)*sin(alpha) , a*sin(theta)],\
                            [0 , sin(alpha) , cos(alpha) , d],\
                            0 , 0 , 0 , 1])

        return tf_mat

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(R) :

        assert(isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    # compute joints' position wrt the base frame
    def compute_joint_pose_bf(self, r_joints_array):
        dh = self.dh
        num_of_joints = len(r_joints_array)
        j_pose_bf = PoseArray()
        j_pose_bf.header.frame_id = "base_frame"
        j_pose_bf.header.stamp = rospy.Time.now()
        tf_mat = np.identity(4)
        for j in range(num_of_joints):
            tf_mat = np.dot( tf_mat, compute_transformation_matrix(dh[j], r_joints_array[j]) )
            j_pose_bf.poses.append(Pose())
            j_pose_bf.poses[j].position.x = tf_mat[0,3]
            j_pose_bf.poses[j].position.y = tf_mat[1,3]
            j_pose_bf.poses[j].position.z = tf_mat[2,3]
            euler = rotationMatrixToEulerAngles(tf_mat[0:3,0:3])
            q = quaternion_from_euler(euler[0], euler[1], euler[2])
            ori = Quaternion(*q)
            j_pose_bf.poses[j].orientation = ori

        return j_pose_bf
