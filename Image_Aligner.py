import cv2
import numpy as np
import glob




class imageAligner:

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    def align_images(self):

        
        
        image_files = glob.glob("Images/*.jpg")  
        images = [cv2.imread(img) for img in image_files]
        
        gray_ref = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        

        # Detect keypoints and descriptors for the reference image
        # orb is used for redcognation
        # It uses corner detection with binary string created around the corners
       

        kp1, des1 = self.orb.detectAndCompute(gray_ref, None)

        # First image is the reference immage. all the remaining images will be aligned according to this image
        aligned_images = [images[0]]  

        # now that we have initilised the reference we will compute the remaining images.

        for i in range(1, len(images)):
            # Convert the current image to grayscale
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors for the current image
            kp2, des2 = self.orb.detectAndCompute(gray, None)

            # Match descriptors using Brute Force Matcher
            
            matches = self.bf.match(des1, des2)

            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            # Warp the current image to align with the reference image
            h, w = gray_ref.shape
            aligned = cv2.warpPerspective(images[i], matrix, (w, h))

            # Append to the list of aligned images
            aligned_images.append(aligned)

        
        
        return aligned_images