import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
import pathlib
import cv2 as cv

img_height : int = 250
img_width : int = 180
org = (50, 50)

loaded_model = tf.keras.models.load_model('modelo')
print(loaded_model.summary())

class_names : list = ['GRUPO_6', 'GRUPO_7']

camara : cv.VideoCapture = cv.VideoCapture(0)

while True:
    _, matrix = camara.read()
    
    matrix_to_model = matrix
    
    matrix_to_model = cv.resize(matrix_to_model, (img_width, img_height))
    matrix_to_model = tf.expand_dims(matrix_to_model, 0)
    predictions = loaded_model.predict(matrix_to_model)
    
    score = tf.nn.softmax(predictions[0])
    label = class_names[np.argmax(score)]
    
    cv.putText(matrix,
               label,
               org,
               cv.FONT_HERSHEY_SIMPLEX,
               2,
               (255,0,0),
               3,
               cv.LINE_AA)
    
    cv.imshow('Camara', matrix)
    
    if cv.waitKey(10) == ord('x'):
        break
    
camara.release()
cv.destroyAllWindows()