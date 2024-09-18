import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()
        self.image = None
        
        # Parâmetros da câmera
        self.fx = 774.608099  # Valor da matriz intrínseca K
        self.fy = 774.119900  # Valor da matriz intrínseca K
        self.h = 0.12          # Altura da câmera (m) - ajuste conforme necessário
        self.theta = np.radians(15)  # Ângulo da câmera em relação ao chão
        self.camera_distance = 0.10 # Distância da câmera até a frente do robô
        # Criar janela
        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.on_mouse_click)
        self.last_click_time = 0  # Tempo do último clique

    def listener_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.imshow("Camera", self.image)
        cv2.waitKey(1)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Exibe coordenadas (x, y)
            print(f"Coordenadas clicadas: ({x}, {y})")
            
            # Calcula a distância do robô e a distância horizontal
            distance, horizontal_distance = self.calculate_distances(x, y)
            print(f"Distância estimada do robô: {distance*100:.2f} cm")
            print(f"Distância horizontal: {horizontal_distance*100:.2f} cm")

            # Desenhar o ponto clicado na imagem e manter por 3 segundos
            self.draw_click_point(x, y, distance, horizontal_distance)

            # Atualizar o tempo do último clique
            self.last_click_time = time.time()

    def calculate_distances(self, x, y):
        # Tamanho da imagem
        image_width = 640
        image_height = 480
        
        # Centraliza o ponto
        cx = x - (image_width / 2)
        cy = y - (image_height / 2)
        
        # Calcula a distância no eixo z (a partir da altura da câmera e do ângulo theta)
        z = self.h / np.tan(self.theta + np.arctan(cy / self.fy))
        
        # Calcula a distância no plano xz (distância horizontal)
        x_real = cx * z / self.fx
        
        # Distância horizontal e distância total
        distance = np.sqrt(x_real**2 + z**2)
        horizontal_distance = x_real
        
        distance_of_robot = distance - self.camera_distance
        return distance_of_robot, horizontal_distance

    def draw_click_point(self, x, y, distance, horizontal_distance):
        # Desenhar um círculo no ponto clicado
        cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
        
        # Exibir as coordenadas e a distância na imagem
        text = f"Dist: {distance*100:.2f}cm, Hor Dist: {horizontal_distance*100:.2f}cm"
        cv2.putText(self.image, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Manter o texto e o ponto visíveis por 3 segundos
        cv2.imshow("Camera", self.image)
        cv2.waitKey(15000)  # Exibir por 3000 milissegundos (3 segundos)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        camera_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
