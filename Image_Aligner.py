import cv2
import numpy as np
import glob

class imageAligner:
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_images(self):
        image_files = glob.glob("Images/*.jpg")  
        images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]

        if not images:
            raise ValueError("No images loaded. Check your folder and file extension.")

        gray_ref = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray_ref, None)

        height, width = gray_ref.shape
        aligned_images = [images[0]]  # First image is already aligned

        for i in range(1, len(images)):
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            kp2, des2 = self.orb.detectAndCompute(gray, None)

            if des1 is None or des2 is None:
                print(f"Descriptors not found in image {i}. Skipping.")
                continue

            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            # good_matches = matches[:int(len(matches) * 0.2)]  # Use top 20%
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if matrix is None:
                print(f"Homography failed for image {i}. Skipping.")
                continue

            aligned = cv2.warpPerspective(images[i], matrix, (width, height))

            # Optional: resize to match just in case
            aligned = cv2.resize(aligned, (width, height))

            aligned_images.append(aligned)
            for idx, aligned_img in enumerate(aligned_images):
                filename = f"Aligned/aligned_{idx+1}.jpg"
                cv2.imwrite(filename, aligned_img)

        return aligned_images
