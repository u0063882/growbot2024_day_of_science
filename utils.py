
import cv2 
import numpy as np 


def create_mask(image): 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for masking (adjust as necessary)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Create an inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment the plant
    plant = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Create a white background
    background = np.full(image.shape, 255, dtype=np.uint8)
    background = cv2.bitwise_and(background, background, mask=mask_inv)

    # Combine plant with white background
    result = cv2.add(plant, background)
    return result, mask 


 