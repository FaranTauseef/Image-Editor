from Image_Aligner import imageAligner
from Median_Blending import medianBlending
import cv2

def main():
    ImageA = imageAligner()
    ImageB = medianBlending()
    print ("start")
    images = ImageA.align_images()
    converted_image = ImageB.blend(images)
    cv2.namedWindow("Median Blended Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Median Blended Image", converted_image)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

  

if __name__ == "__main__":
    main()