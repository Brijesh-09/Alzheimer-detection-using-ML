import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('Alzheimer10EpochsCategorical.h5')

<<<<<<< HEAD
image=cv2.imread('C:\\Users\\brijesh\\OneDrive\\Desktop\\Major project\\Alzheimer\\datasets\\non_demented\\non_3064.jpg')
# C:\Users\brijesh\OneDrive\Desktop\Major project\Alzheimer\datasets\demented\mild_440.jpg
=======
image=cv2.imread('datasets/demented/verymild_619.jpg')

>>>>>>> 5c45c0898a4bea319c4321d6561d2bfdb70b4d8d

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)
print(result)




