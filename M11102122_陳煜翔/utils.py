import cv2
import numpy as np
import matplotlib.pyplot as plt

class cv_image:
    def __init__(self, Image):
        self.image = cv2.resize(Image, (1024, 1024))
        self.new_image = cv2.resize(Image, (1024, 1024))
    
    def convert2gray(self): # Convert Image from BGR to GRAY
        self.new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def gamma_correction_image(self, gamma): # Use gamma correction to change illumination of image
        inverse_gamma = 1.0 / gamma
        self.new_image = self.image.astype(np.float32) / 255.0 # Normalize image to [0, 1]
        self.new_image = ((self.new_image ** inverse_gamma) * 255.0 ).astype(np.uint8)

    def plt_histogram(self, img, name, save): # Get histogram value
        hist = plt.hist(img.flatten(), 256, [0, 255], label = name)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        
        if save == True:
            plt.legend([name], loc ="lower right")
            plt.savefig(name + ".png")
            plt.close("all")
            
        return hist
    
    def histogram_equalization(self, q_k, q_o, hist):
        width, heigh = self.new_image.shape # Save the orignal shape for the resize to same size.

        img = self.new_image.flatten() # Convert to 1-D vector
        cp_img = img.copy()

        fq_sigma = 0
        for i in range(img.size):
            fq_sigma = hist[:cp_img[i]+1].sum() # Calculate sigma(H(p))
            img[i] = (((q_k - q_o) / img.size) * fq_sigma) + q_o # histogram_equalization function
        
        img = np.resize(img, (width, heigh)) #　Resize to the original size
        return img