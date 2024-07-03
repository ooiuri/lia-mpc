#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError
from detect_lane.process_image import detect_lane_image

class Detect_lane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.get_logger().info('Looking for the lane...')
        self.image_sub = self.create_subscription(Image,"/image_in",self.callback,rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.lane_pub  = self.create_publisher(Point,"/detected_lane",1)
        self.bridge = CvBridge()


    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            
            out_image = detect_lane_image(cv_image)
            # keypoints_norm, out_image, tuning_image = proc.find_circles(cv_image, self.tuning_params)
            print("Publishing image")
            img_to_pub = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            img_to_pub.header = data.header
            self.image_out_pub.publish(img_to_pub)

            # img_to_pub = self.bridge.cv2_to_imgmsg(tuning_image, "bgr8")
            # img_to_pub.header = data.header
            # self.image_tuning_pub.publish(img_to_pub)

            # point_out = Point()

            # Keep the biggest point
            # They are already converted to normalised coordinates
            # for i, kp in enumerate(keypoints_norm):
            #     x = kp.pt[0]
            #     y = kp.pt[1];
            #     s = kp.size

            #     self.get_logger().info(f"Pt {i}: ({x},{y},{s})")

            #     if (s > point_out.z):                    
            #         point_out.x = x
            #         point_out.y = y
            #         point_out.z = s

            # if (point_out.z > 0):
            #     self.lane_pub.publish(point_out) 
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

