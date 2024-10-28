#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from visualization_msgs.msg import Marker
from std_msgs.msg           import Int16MultiArray, Float32MultiArray, MultiArrayDimension
from cv_bridge              import CvBridge, CvBridgeError
from detect_lane.process_image import detect_lane_image

class Detect_lane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        self.get_logger().info('Looking for the lane...')
        self.image_sub = self.create_subscription(Image, "/image_in", self.callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_out_pub = self.create_publisher(Image, "/image_out", 1)
        self.image_tuning_pub = self.create_publisher(Image, "/image_tuning", 1)
        self.detected_points_pub  = self.create_publisher(Float32MultiArray,"/detected_points",1)

        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.bridge = CvBridge()

    def publish_marker(self, data):
        # Criação do marker
        marker = Marker()
        marker.header.frame_id = "base_link"  # Defina o frame de referência adequado ao seu sistema
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "lines"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.02  # Largura da linha
        marker.color.a = 1.0   # Transparência
        marker.color.r = 1.0   # Cor vermelha
        marker.color.g = 0.0   # Cor verde
        marker.color.b = 0.0   # Cor azul

        # Adiciona os pontos ao marcador
        for x, y in data:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0  # Z pode ser zero se você estiver trabalhando em 2D
            marker.points.append(point)

        # Publica o marker
        self.marker_pub.publish(marker)

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
            points_out = Float32MultiArray()
            # points_out.data = detected_points
            points_out.data = [item for pair in detected_points for item in pair]

            # Configura o layout para representar a estrutura bidimensional
            points_out.layout.dim.append(MultiArrayDimension(label="points", size=len(detected_points), stride=len(detected_points) * 2))
            points_out.layout.dim.append(MultiArrayDimension(label="coordinates", size=2, stride=2))
            self.detected_points_pub.publish(points_out) 
            self.publish_marker(detected_points)

        except CvBridgeError as e:
            print(e)  
        
        except Exception as e:
            print(e)  



def main(args=None):

    rclpy.init(args=args)

    detect_lane = Detect_lane()
    while rclpy.ok():
        rclpy.spin_once(detect_lane)
        # proc.wait_on_gui()

    detect_lane.destroy_node()
    rclpy.shutdown()

