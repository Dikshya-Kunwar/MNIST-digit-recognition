import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model("mnist.model")
# loss, accuracy = model.evaluate(x_test,y_test)

image_number=1
while os.path.isfile(f"dataset\{image_number}.png"):
    try:
        img = cv2.imread(f"dataset\{image_number}.png")[:,:,0]
        invert = cv2.bitwise_not(img)
        img = np.invert(np.array([invert]))
        prediction = model.predict(img)
        print(f"This digit is {np.argmax(prediction)}")              
        plt.imshow(img[0],cmap= "Greys")
        plt.show()
    except:
        print("Error")

    finally:
        image_number+=1