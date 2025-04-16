import cv2
import numpy as np
import glob


class medianBlending:

    def blend(self,images):
        #load the images from where we stored them
        # image_files = glob.glob("Images/*.jpg")
        # images = [cv2.imread(img) for img in image_files]

        stack = np.array(images)

        median_image = np.median(stack, axis=0).astype(np.uint8)

        cv2.imwrite("output.jpg", median_image)
       
        filename = f"Converted/converted.jpg"
        cv2.imwrite(filename, median_image)
        return median_image