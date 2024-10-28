import numpy as np
import cv2

class ImageClass:
    def __init__(self, image):
        self.image = image
        self.image_out = None
        self.contourCenterX = 0
        self.MainContour = None
        self.n_slices = 5
        self.detected_points_center = []
        self.detected_points = []
        # Real robot coordinates
        # self.upper_limit = 235
        # self.bottom_limit = 335

        # Simulated robot coordinates
        self.upper_limit = 100
        self.bottom_limit = 235

        # Matriz intrínseca e coeficientes de distorção fornecidos
        self.intrinsic_matrix = np.array([[774.608099, 0.0, 342.430253], 
                                          [0.0, 774.119900, 264.814194], 
                                          [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.102414, -0.221511, 0.013876, 0.019191, 0.0])

        self.camera_height = 0.12  # Altura da câmera em metros
        self.camera_angle = np.radians(15)  # Inclinação da câmera em radianos
        self.camera_distance = 0.125  # Distância da câmera até a frente do robô
        self.clicked_point = None

    def slice_roi(self, roi, n_slices):
        height, width = roi.shape[:2]
        slice_height = height // n_slices
        slices = []
        for i in range(n_slices):
            start_row = i * slice_height
            end_row = (i + 1) * slice_height if i < n_slices - 1 else height
            roi_slice = roi[start_row:end_row, :]
            slices.append(roi_slice)
        return slices

    def repack_image(self, slices):
        img = slices[0]
        for slice_img in slices[1:]:
            img = np.concatenate((img, slice_img), axis=0)
        img_full = np.concatenate((self.image[:self.upper_limit, :], img), axis=0)
        img_full = np.concatenate((img_full, self.image[self.bottom_limit:, :]), axis=0)
        return img_full

    def process_image(self):
        img_roi = self.image[self.upper_limit:self.bottom_limit, :]
        roi_slices = self.slice_roi(img_roi, self.n_slices)
        
        for i, roi_slice in enumerate(roi_slices):
            imgray = cv2.cvtColor(roi_slice, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY_INV)
            # thresh = cv2.bitwise_not(thresh)  # Inverter pixels para linha preta em fundo claro

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if contours:
                self.MainContour = self.findMainContour(contours)
                contourCenter = self.getContourCenter(self.MainContour)
                if contourCenter:
                    self.contourCenterX = contourCenter[0]
                    width = roi_slice.shape[1]
                    center_offset = contourCenter[0] - width // 2
                    self.detected_points_center.append(center_offset)

                    # Converter as coordenadas do centro do contorno para a imagem completa
                    global_center = (contourCenter[0], contourCenter[1] + self.upper_limit + i * (roi_slice.shape[0]))
                    # Salvar coordenadas do centro da imagem
                    self.detected_points.append(global_center)

                    # Desenhar o contorno e o ponto azul na imagem original
                    cv2.drawContours(self.image, [self.MainContour + np.array([[0, self.upper_limit + i * (roi_slice.shape[0])]])], -1, (0, 255, 0), 2)
                    cv2.circle(self.image, global_center, 3, (255, 0, 0), -1)
                    
                    # Adicionar o texto das coordenadas do ponto na imagem original
                    text_position = (global_center[0] + 5, global_center[1] - 5)
                    cv2.putText(self.image, f"({global_center[0]}, {global_center[1]})", 
                                text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Ponto branco de referência no centro do slice na imagem original
            slice_height = roi_slice.shape[0]
            slice_center = (width // 2, slice_height // 2 + self.upper_limit + i * slice_height)
            cv2.circle(self.image, slice_center, 3, (255, 255, 255), -1)

        self.image_out = self.image
        final_height, final_width = self.image_out.shape[:2]
        cv2.line(self.image_out, (0, self.bottom_limit), (final_width, self.bottom_limit), (0, 0, 255), 2)
        cv2.line(self.image_out, (0, self.upper_limit), (final_width, self.upper_limit), (0, 0, 255), 2)

    def findMainContour(self, contours):
        biggestContour = max(contours, key=cv2.contourArea)

        # Verifique se o contorno está abaixo da linha preta
        contourCenter = self.getContourCenter(biggestContour)
        if contourCenter and contourCenter[1] > self.bottom_limit:  # Certifique-se de que Y está abaixo da linha
            return biggestContour

        if self.detected_points_center:
            if self.getContourCenter(biggestContour):
                biggestContourX = self.getContourCenter(biggestContour)[0]
                if abs((self.contourCenterX - biggestContourX) - self.detected_points_center[-1]) > 50:
                    closest_contour = biggestContour
                    closest_contourX = biggestContourX
                    for contour in contours:
                        contourCenter = self.getContourCenter(contour)
                        if contourCenter and contourCenter[1] > self.bottom_limit:  # Verificar a linha
                            temp_contourX = contourCenter[0]
                            if abs((self.contourCenterX - temp_contourX) - self.detected_points_center[-1]) < \
                               abs((self.contourCenterX - closest_contourX) - self.detected_points_center[-1]):
                                closest_contour = contour
                                closest_contourX = temp_contourX
                    return closest_contour
                else:
                    return biggestContour
        return biggestContour

    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        return [x, y]
    
    def pixel_to_real_world(self, u, v):
        """Function to translate pixel coordinate to real world distance"""

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

def detect_lane_image(image):    
    proc = ImageClass(image)
    proc.process_image()
    processed_points = []
    for point in reversed(proc.detected_points):
        res = proc.pixel_to_real_world(point[0], point[1])
        processed_points.append(res)

    return proc.image_out, processed_points
