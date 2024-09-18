import numpy as np
import cv2

class ImageClass:
    def __init__(self, image):
        self.image = image
        self.image_out = None
        self.contourCenterX = 0
        self.MainContour = None
        self.n_slices = 10
        self.image_slices = []
        self.detected_points = []
        self.upper_limit = 235
        self.bottom_limit = 335
    
    def slice_image(self, n_slices):
        ''' Function to split image into slices \n
        @param n_slices (int): number of desired slices \n
        @return image_slices (list): Array of slices
        '''
        img = self.image
        height, width = img.shape[:2]
        slice_height = self.bottom_limit - self.upper_limit
        slice_size = int(slice_height/n_slices)
        self.image_slices.append(img[0:self.upper_limit, 0:width]) 
        for i in range(n_slices):
            part = self.upper_limit + slice_size * i
            crop_img = img[part:part + slice_size, 0:width]
            self.image_slices.append(crop_img)

        self.image_slices.append(img[self.bottom_limit:height, 0:width]) 
        
        return self.image_slices
    
    def repack_image(self):
        img = self.image_slices[0]
        for i in range(len(self.image_slices)):
            if i == 0:
                img = np.concatenate((img, self.image_slices[1]), axis=0)
            if i > 1:
                img = np.concatenate((img, self.image_slices[i]), axis=0)
        return img

    def process_image(self):
        # start by spliting image into slices
        self.slice_image(self.n_slices)

        for image_slice in reversed(self.image_slices[1:-1]):
            imgray = cv2.cvtColor(image_slice,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
            ret, thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY_INV) #Get Threshold
            # edges = cv2.Canny(blur,15,15,apertureSize=3)
            # blurred = cv2.medianBlur(imgray, 9
            # blur = cv2.GaussianBlur(imgray, (kernel_size, kernel_size), 0)
            # thresh = cv2.bitwise_not(thresh)
            contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Get contour
            
            self.prev_MC = self.MainContour
            self.prev_MCX = self.contourCenterX
            if contours:
                # self.MainContour = max(contours, key=cv2.contourArea)
                self.MainContour = self.findMainContour(contours)
                height, width  = image_slice.shape[:2]

                self.middleX = int(width/2) #Get X coordenate of the middle point
                self.middleY = int(height/2) #Get Y coordenate of the middle point
                contourCenter = self.getContourCenter(self.MainContour)
                if contourCenter:    
                    self.contourCenterX = contourCenter[0]
                    cv2.drawContours(image_slice,self.MainContour,-1,(0,255,0),1) #Draw Contour GREEN
                    cv2.circle(image_slice, (self.contourCenterX, self.middleY), 3, (255,255,255), -1) #Draw dX circle WHITE
                    cv2.circle(image_slice, (self.middleX, self.middleY), 1, (0,0,255), -1) #Draw middle circle RED
                    
                    point_offset = self.middleX - self.contourCenterX
                    
                    cv2.putText(image_slice, str(point_offset), (self.contourCenterX + 10, self.middleY + 5), cv2.FONT_HERSHEY_COMPLEX, .5, (255,255,255))
                    self.detected_points.append(self.middleX - self.contourCenterX)

        self.image_out = self.repack_image()
        final_height, final_width = self.image_out.shape[:2]
        cv2.line(self.image_out, (0, self.bottom_limit), (final_width, self.bottom_limit), (0,0,255), 2)
        cv2.line(self.image_out, (0, self.upper_limit), (final_width, self.upper_limit), (0,0,255), 2)
    
    def findMainContour(self, contours):
        biggestContour = max(contours, key=cv2.contourArea)
        if len(self.detected_points):
            if self.getContourCenter(biggestContour):    
                biggestContourX = self.getContourCenter(biggestContour)[0]
                if (abs((self.middleX - biggestContourX)- self.detected_points[-1]) > 50):
                    contour = biggestContour
                    contourX = biggestContourX
                    for tmp_contour in contours:
                        if (self.getContourCenter(tmp_contour)):
                            temp_contourX = self.getContourCenter(tmp_contour)[0]
                            if (abs((self.middleX - temp_contourX) - self.detected_points[-1]) < 
                                abs((self.middleX - contourX) - self.detected_points[-1])):
                                contour = tmp_contour
                                contourX = temp_contourX
                    return contour
                else:
                    return biggestContour
        else:
            return max(contours, key=cv2.contourArea)
         
        

    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return 0
        
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        
        return [x,y]
                                                    

def detect_lane_image(image):            
    # Read image
    img = image
    processImage = ImageClass(img)
    processImage.process_image()             
    image_out = processImage.image_out
    detected_points = processImage.detected_points
    # Save the result image
    # cv2.imwrite('contours.png',processImage.image_out)
    return image_out, detected_points
                            
                                
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            