#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Twist
from std_msgs.msg           import Int16MultiArray, Float32MultiArray, MultiArrayDimension
from detect_lane.mpc_controller import DifferentialDriveRobot, MPCController
from cv_bridge              import CvBridge, CvBridgeError

from scipy.interpolate import UnivariateSpline

from detect_lane.process_image import plot_trajetory

import numpy as np
import time

class ControlRobot(Node):
    """Subscribe into detectep points topic \n
    and handles the controller velocity
    """
    def __init__(self):
        super().__init__('control_robot')

        self.get_logger().info('Looking for detected points...')
        self.detected_points_sub = self.create_subscription(Float32MultiArray, "/detected_points", self.callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.detected_points_data = [[0,0]]*20

        self.publisher_ = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.predicted_trajetory  = self.create_publisher(Float32MultiArray,"/predicted_trajetory",1)
        self.trajetory_data = []

        self.image_sub = self.create_subscription(Image, "/image_out", self.image_callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_trajetory", 1)
        self.bridge = CvBridge()

        # Parâmetros do robô
        self.R = 0.05  # Raio da roda em metros
        self.L = 0.09  # Distância entre as rodas em metros
        # Criar instância do robô diferencial
        robot = DifferentialDriveRobot(self.R, self.L)
        
        # Criar instância do controlador MPC
        self.mpc = MPCController(robot, N=14, Q=1, R=0.0, dt=0.5, max_margin=0.2)

        timer_period = 0.6  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.lastrcvtime = time.time() - 2000
        self.rcv_timeout_secs = 1

    def timer_callback(self):
        msg = Twist()
        if (time.time() - self.lastrcvtime < self.rcv_timeout_secs):
            try:
                if not self.detected_points_data or not len(self.detected_points_data):
                    print('no points', self.detected_points_data)
                    return
                print('*'*10, 'Running Control')
                Xc = []
                Yc = []
                # print('before Xc: ', self.detected_points_data)
                for point in self.detected_points_data:
                    Xc.append(point[1])
                    Yc.append(-point[0])
                
                # print('after Xc: ', Xc)
                x_initial = np.linspace(0, Xc[0], 5)
                y_initial = np.linspace(0, Yc[0], 5)

                x_interpolado = np.concatenate((x_initial, Xc[1:]))
                y_interpolado = np.concatenate((y_initial, Yc[1:]))

                cs = UnivariateSpline(x_interpolado, y_interpolado, s=1.5)

                # # Gera os pontos suavizados
                # x_smooth = np.linspace(0, x_interpolado[-1], num=25)
                x_smooth = np.linspace(0, x_interpolado[-1], num=25)
                y_smooth = cs(x_smooth)

                # x_initial = np.linspace(0, 0.36, 10)
                # y_initial = np.linspace(0, 0, 12)

                # x_traj = x_initial[1:-1]
                # y_traj = y_initial[1:-1]
                # Concatena a trajetória inicial com os pontos de referênci
                x_traj = x_smooth # np.concatenate((x_initial[0:-1]))
                y_traj = y_smooth # np.concatenate((y_initial[0:-1]))
                print(f'x_traj: {(x_smooth)}')
                print(f'y_traj: {(y_smooth)}')

                v_R, v_L, mpc_traj = self.mpc.solve(x_traj, y_traj)
                print(f'v_R: {v_R} \t v_L: {v_L}')
                # print(f'mpc_traj: {mpc_traj}')
                # v_linear = self.R / 2 *(v_R + v_L)
                v_linear = (v_R + v_L) * self.R / 2
                w_angular = (v_R - v_L) * self.R / self.L # correct one
                msg.linear.x = v_linear
                msg.angular.z = w_angular

                # mpc traj publisher
                points_out = Float32MultiArray()
                # points_out.data = detected_points
                trajetory = []
                for i in range(len(mpc_traj[0])):
                    trajetory.append([mpc_traj[0][i], mpc_traj[1][i]])
                self.trajetory_data = trajetory
                # print(f'self.trajetory_data: {self.trajetory_data}')
                # print(f'len trajetory_data: {len(self.trajetory_data)}')
                # points_out.data = self.trajetory_data
                # # Configura o layout para representar a estrutura bidimensional
                # points_out.layout.dim.append(MultiArrayDimension(label="points", size=len(mpc_traj), stride=len(self.detected_points_data) * 2))
                # points_out.layout.dim.append(MultiArrayDimension(label="coordinates", size=2, stride=2))
                # self.predicted_trajetory.publish(points_out) 
            except Exception as e:
                print('Could not detect trajetory', str(e))
            

        self.publisher_.publish(msg)

    def callback(self, msg):
        # print('cb: msg_data', msg.data)
        points = msg.data
        if not len(points):
            return
        detected_points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        # print('cb: detected_points', detected_points)
        detected_points.reverse()
        self.detected_points_data = detected_points
        # print('cb: detected_points_data', self.detected_points_data)
        # print(f'detected_points_data: {self.detected_points_data}')
        self.lastrcvtime = time.time()

    def image_callback(self,data):
        # try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # except CvBridgeError as e:
        #     print(e)
        if not len(self.trajetory_data):
            # print('no trajetory data')
            return
        # try:
        out_image = plot_trajetory(cv_image, self.trajetory_data)
        # keypoints_norm, out_image, tuning_image = proc.find_circles(cv_image, self.tuning_params)
        img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
        img_to_pub.header = data.header
        self.image_out_pub.publish(img_to_pub)
        # except CvBridgeError as e:
        #     print(e)  
        
        # except Exception as e:
        #     print(e)  

def main(args=None):
    rclpy.init(args=args)
    controlRobot = ControlRobot()
    while rclpy.ok():
        rclpy.spin_once(controlRobot)
        # proc.wait_on_gui()

    controlRobot.destroy_node()
    rclpy.shutdown()

