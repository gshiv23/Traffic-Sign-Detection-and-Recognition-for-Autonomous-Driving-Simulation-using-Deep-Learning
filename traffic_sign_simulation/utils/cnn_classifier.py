import tensorflow as tf
import cv2
import numpy as np
import os

class CNNClassifier:

    def __init__(self):

        model_path = os.path.join(r"C:\Users\gshiv\Desktop\Project Internship\Preprocessing Code\traffic_sign_simulation\models\traffic_sign_final_model.h5")

        model_path = os.path.abspath(model_path)

        print("Loading model from:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = tf.keras.models.load_model(model_path)

    def classify(self, img):

        img = cv2.resize(img,(224,224))
        img = img/255.0

        img = np.expand_dims(img,axis=0)

        prediction = self.model.predict(img)

        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return class_id, confidence