import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class PixelToRealWorld(Node):
    def __init__(self):
        super().__init__('pixel_to_real_world')

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Altere para o tópico de sua câmera
            self.image_callback,
            10)
        
        self.bridge = CvBridge()
        
        # Matriz intrínseca e coeficientes de distorção fornecidos
        self.intrinsic_matrix = np.array([[774.608099, 0.0, 342.430253], 
                                          [0.0, 774.119900, 264.814194], 
                                          [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.102414, -0.221511, 0.013876, 0.019191, 0.0])

        self.camera_height = 0.12  # Altura da câmera em metros
        self.camera_angle = np.radians(15)  # Inclinação da câmera em radianos
        self.camera_distance = 0.125  # Distância da câmera até a frente do robô
        self.clicked_point = None
        self.image = None

        self.last_click_time = 0  # Tempo do último clique

    def image_callback(self, msg):
        # Converte a mensagem ROS Image para uma imagem OpenCV
        distorted_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Remove a distorção da imagem
        self.image = cv2.undistort(distorted_image, self.intrinsic_matrix, self.dist_coeffs)
        
        # Mostra a imagem corrigida e permite clicar para pegar a posição do pixel
        cv2.imshow("Undistorted Camera Image", self.image)
        cv2.setMouseCallback("Undistorted Camera Image", self.on_mouse_click)
        cv2.waitKey(1)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print(f"Clicked at pixel: {self.clicked_point}")
            Xc, Zc = self.pixel_to_real_world(x, y)
            print(f"Real world position (X: {Xc:.2f}, Z: {Zc:.2f})")
            # Desenhar o ponto clicado na imagem e manter por 3 segundos
            self.draw_click_point(x, y, Xc, Zc)

            # Atualizar o tempo do último clique
            self.last_click_time = time.time()

    def pixel_to_real_world(self, u, v):
        # Desfaz a distorção do ponto
        points = np.array([[u, v]], dtype=np.float32)
        undistorted_points = cv2.undistortPoints(np.expand_dims(points, axis=1), self.intrinsic_matrix, self.dist_coeffs)
        u_distorted, v_distorted = undistorted_points[0][0]

        # Converte o ponto da imagem para coordenadas reais (X, Z)
        fx = self.intrinsic_matrix[0, 0]  # f_x
        fy = self.intrinsic_matrix[1, 1]  # f_y

        # Coordenadas do pixel em relação ao centro da imagem
        dx = (u_distorted * fx)
        dy = (v_distorted * fy)

        # Calcula o ângulo vertical da câmera em relação ao pixel clicado
        alfa_y = np.arctan2(dy, fy)
        total_theta = self.camera_angle + alfa_y

        if total_theta <= 0:
            total_theta = 1e-5  # Evita divisão por zero

        # Calcula a distância no plano XZ (Z positivo no mundo real)
        Zc = self.camera_height / np.tan(total_theta)
        Xc = Zc * dx / fx

        distance_of_robot = Zc - self.camera_distance
        
        return Xc, distance_of_robot

    def draw_click_point(self, x, y, Xc, Zc):
        # Desenhar um círculo no ponto clicado
        cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
        
        # Exibir as coordenadas e a distância na imagem
        textX = f"Xw: {Xc*100:.2f}cm"
        textZ = f"Zw: {Zc*100:.2f}cm"
        cv2.putText(self.image, textX, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 25, 0), 2)
        cv2.putText(self.image, textZ, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 25, 0), 2)
        
        # Manter o texto e o ponto visíveis por 3 segundos
        cv2.imshow("Undistorted Camera Image", self.image)
        cv2.waitKey(30000)  # Exibir por 3000 milissegundos (3 segundos)

def main(args=None):
    rclpy.init(args=args)
    pixel_to_real_world = PixelToRealWorld()
    
    try:
        rclpy.spin(pixel_to_real_world)
    except KeyboardInterrupt:
        pass
    
    pixel_to_real_world.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
