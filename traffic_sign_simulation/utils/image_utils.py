import numpy as np
import cv2

def carla_to_opencv(image):

    img = np.frombuffer(image.raw_data, dtype=np.uint8)

    img = img.reshape((image.height, image.width, 4))

    img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img