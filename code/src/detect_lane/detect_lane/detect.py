#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from std_msgs.msg           import Int16MultiArray
from cv_bridge              import CvBridge, CvBridgeError
from detect_lane.process_image import detect_lane_image

class Detect_lane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.get_logger().info('Looking for the lane...')
        self.image_sub = self.create_subscription(Image,"/image_in",self.callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.detected_points_pub  = self.create_publisher(Int16MultiArray,"/detected_points",1)
        self.bridge = CvBridge()


    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            
            out_image, detected_points = detect_lane_image(cv_image)
            # keypoints_norm, out_image, tuning_image = proc.find_circles(cv_image, self.tuning_params)
            img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            img_to_pub.header = data.header
            self.image_out_pub.publish(img_to_pub)

            # img_to_pub = self.bridge.cv2_to_imgmsg(tuning_image, "bgr8")
            # img_to_pub.header = data.header
            # self.image_tuning_pub.publish(img_to_pub)
 
            points_out = Int16MultiArray()
            points_out.data = detected_points

            self.detected_points_pub.publish(points_out) 
        except CvBridgeError as e:
            print(e)  


def main(args=None):

    rclpy.init(args=args)

    detect_lane = Detect_lane()
    while rclpy.ok():
        rclpy.spin_once(detect_lane)
        # proc.wait_on_gui()

    detect_lane.destroy_node()
    rclpy.shutdown()

