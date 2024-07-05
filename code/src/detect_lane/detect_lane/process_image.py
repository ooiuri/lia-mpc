import numpy as np
import cv2

upper_limit = 300
bottom_limit = 480

# --------------------------------
def SliceImage(img, n_slices):
    ''' Function to split image into slices \n
    @param img (int): original image \n
    @param n_slices (int): number of desired slices \n
    @return image_slices (list): Array of slices
    '''
    image_slices = []
    height, width = img.shape[:2]
    slice_height = bottom_limit - upper_limit
    slice_size = int(slice_height/n_slices)
    image_slices.append(img[0:upper_limit, 0:width]) 
    for i in range(n_slices):
        part = upper_limit + slice_size * i
        crop_img = img[part:part + slice_size, 0:width]
        image_slices.append(crop_img)

    image_slices.append(img[bottom_limit:height, 0:width]) 

    return image_slices

def RepackImage(image_slices):
    img = image_slices[0]
    for i in range(len(image_slices)):
        if i == 0:
            img = np.concatenate((img, image_slices[1]), axis=0)
        if i > 1:
            img = np.concatenate((img, image_slices[i]), axis=0)
    return img

class ImageClass:
    def __init__(self, image):
        self.image = image
        self.contourCenterX = 0
        self.MainContour = None
        
    def Process(self):
        imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY) #Convert to Gray Scale
        ret, thresh = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY_INV) #Get Threshold
        # edges = cv2.Canny(imgray,50,150,apertureSize=3)
        # blurred = cv2.medianBlur(imgray, 9)
        thresh = cv2.bitwise_not(thresh)
        self.contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #Get contour
        
        self.prev_MC = self.MainContour
        if self.contours:
            self.MainContour = max(self.contours, key=cv2.contourArea)
            self.height, self.width  = self.image.shape[:2]

            self.middleX = int(self.width/2) #Get X coordenate of the middle point
            self.middleY = int(self.height/2) #Get Y coordenate of the middle point
        
            self.contourCenterX = self.getContourCenter(self.MainContour)[0]
            
            cv2.drawContours(self.image,self.MainContour,-1,(0,255,0),3) #Draw Contour GREEN
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 3, (255,255,255), -1) #Draw dX circle WHITE
            cv2.circle(self.image, (self.middleX, self.middleY), 1, (0,0,255), -1) #Draw middle circle RED
                
            
    def getContourCenter(self, contour):
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return 0
        
        x = int(M["m10"]/M["m00"])
        y = int(M["m01"]/M["m00"])
        
        return [x,y]
                            

def detect_lane_image(image):            
    # Read image
    # img = cv2.imread('assets/01/frame0001.jpg')
    img = image
    parts = SliceImage(img, 12)

    part_list = []
    part_list.append(parts[0])
    for part in parts[1:-1]:
        partImage = ImageClass(image=part)
        if partImage.image is not None:
            partImage.Process()
        part_list.append(partImage.image)
    part_list.append(parts[-1])
    final_img = RepackImage(part_list)
    final_height, final_width = final_img.shape[:2]
    cv2.line(final_img, (0, bottom_limit), (final_width, bottom_limit), (0,0,255), 2)
    cv2.line(final_img, (0, upper_limit), (final_width, upper_limit), (0,0,255), 2)

    # Save the result image
    # cv2.imwrite('contours.png',final_img)
    return final_img
                            
                                
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            