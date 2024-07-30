#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg           import Int16MultiArray
from controller.mpc_controller import DifferentialDriveRobot, MPCController

import numpy as np
import time

class ControlRobot(Node):
    """Subscribe into detectep points topic \n
    and handles the controller velocity
    """
    def __init__(self):
        super().__init__('control_robot')

        self.get_logger().info('Looking for detected points...')
        self.detected_points_sub = self.create_subscription(Int16MultiArray, "/detected_points", self.callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.detected_points_data = [0]*20

        self.publisher_ = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        
        # Parâmetros do robô
        self.R = 0.1  # Raio da roda em metros
        self.L = 0.5  # Distância entre as rodas em metros
        # Criar instância do robô diferencial
        robot = DifferentialDriveRobot(self.R, self.L)
        
        # Criar instância do controlador MPC
        self.mpc = MPCController(robot, N=10, Q=120, R=0.02, dt=0.1)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.lastrcvtime = time.time() - 10000
        self.rcv_timeout_secs = 1.0

    def timer_callback(self):
        msg = Twist()
        if (time.time() - self.lastrcvtime < self.rcv_timeout_secs):
            try:
                y_ref = np.linspace(0,0.36,20)
                v_R, v_L = self.mpc.solve(y_ref,self.detected_points_data)
                print(f'v_R: {v_R} \t v_L: {v_L}')
                v_linear = self.R / 2 *(v_R + v_L)
                w_angular = self.R / self.L *(v_R - v_L) # correct one
                msg.linear.x = v_linear
                msg.angular.z = w_angular
            except Exception as e:
                print(e)

        self.publisher_.publish(msg)

    def callback(self, msg):
        detected_points = np.linspace(0,msg.data[0]/1000,10)
        for point in msg.data:
            # detected_points.append(point / 1000)
            detected_points = np.append(detected_points, [point/1000])
        self.detected_points_data = detected_points
        # print(f'detected_points_data: {self.detected_points_data}')
        self.lastrcvtime = time.time()


def main(args=None):
    rclpy.init(args=args)
    controlRobot = ControlRobot()
    while rclpy.ok():
        rclpy.spin_once(controlRobot)
        # proc.wait_on_gui()

    controlRobot.destroy_node()
    rclpy.shutdown()

