import numpy as np
import cv2
import scipy.io
import matplotlib.pyplot as plt
from keras.models import load_model,Input,Model
from PIL import Image
import pandas as pd
import skvideo
plt.ion()
def show_image(img,title=""): 
	if(img.shape[2]==3):
		temp = img
	else:
		temp = img.transpose((2,1,0))
	plt.figure()
	plt.title(title)
	plt.imshow(temp)
	plt.show()
def noisy(level,image): 
	#accepts a last channel image and returns the same 
	row,col,ch= image.shape
	mean = 0
	# var = 0.1
	sigma = level
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	return noisy

def fun_mscn(temp):
	#first channel input and return by this function
    dummy=np.zeros(shape=temp.shape) 
    x,y,z = skvideo.utils.compute_image_mscn_transform(temp[0,:,:])
    dummy[0,:,:] = x
    x,y,z = skvideo.utils.compute_image_mscn_transform(temp[1,:,:])
    dummy[1,:,:] = x
    x,y,z = skvideo.utils.compute_image_mscn_transform(temp[2,:,:])
    dummy[2,:,:] = x
    return dummy

dist_model = load_model('./DistNet_model/mscn_v1.37-0.09254.h5',compile=False)
model_path = './Denoise_model/weight_cont.01-0.00124.h5'
model = load_model(model_path)

# impath = '/home/sathya/Downloads/NEW_DATASET_V2/img62.bmp'
impath = './test_images/img_uniform.png'
img = cv2.imread(impath)
#img = img[:,:,:3]
#show_image(img,'Clean Image')
# img_noisy = noisy(10,img)
img_noisy = img[:224,:224,:]
# show_image(img_noisy,'Noisy image')
img_mscn = fun_mscn(np.transpose(img_noisy,(2,0,1)))
# show_image(img_mscn,'MSCN noisy image')
img_mscn = np.expand_dims(img_mscn,axis=0)
dist_map = dist_model.predict(img_mscn)
show_image(np.squeeze(dist_map),'Predicted Distortion map')

img = Image.open(impath)
img.load()
img = np.asarray( img, dtype="uint8" )
img_noisy = img[:224,:224,:]
show_image(img_noisy,'Noisy image')
img_noisy = np.expand_dims(img_noisy,axis=0)

dist_map = np.transpose(dist_map,(0,2,3,1))

restor = model.predict([img_noisy/255.0,dist_map])
pred = np.squeeze(restor)
show_image(pred,'Restored image')
plt.imsave('./output/img_uniform_restored.png',pred)

