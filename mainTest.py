import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('Alzheimer10EpochsCategorical.h5')

image=cv2.imread('C:\\Users\\brijesh\\OneDrive\\Desktop\\Major project\\Alzheimer\\datasets\\non_demented\\non_3064.jpg')
# C:\Users\brijesh\OneDrive\Desktop\Major project\Alzheimer\datasets\demented\mild_440.jpg

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)
print(result)




