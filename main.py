import numpy as np
import cv2
from pydicom import dcmread
from matplotlib import pyplot as plt 
from PIL import Image, ImageFilter

# def get_neighboring_pixels(image,R,y,x,isFloat):
# 	#create an array with dimension of kernel, same type as image
# 	#neighbor = np.zeros((R,R))
# 	#neighbor = neighbor.astype(image.dtype)
# 	#copy value inside kernel to created array
# 	neighbor = image[y-R:y+R+1,x-R:x+R+1]
# 	return neighbor

def calculate_contextual(image):
	
	#kernel size = 2R+1
	R = 1

	h = image.shape[0]
	w = image.shape[1]
	# new_image = cv2.blur(image,(R,R))
	#create variance matrix
	contextual = np.zeros_like(image,dtype=float)
	for row in range(R,h-1-R):
		for col in range(R,w-1-R):
			neighbor = np.empty((2*R+1,2*R+1))
			neighbor = image[row-R:row+R+1,col-R:col+R+1]
			contextual[row,col] = calculate_variance(neighbor)
	#normalize the variance matrix
	min_variance = contextual[R:h-1-R,R:w-1-R].min()
	max_variance = contextual[R:h-1-R,R:w-1-R].max()
	difference = max_variance - min_variance

	contextual = (contextual - min_variance)/difference
	# print(contextual[100,100])
	# print(contextual[0,0])
	# print(new_image.type)
	# print(new_image.shape)
	# print(image.shape)
	# print(h+w)
	# plt.figure(figsize=(11,6))
	# plt.subplot(121), plt.imshow(image),plt.title('Original')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(122), plt.imshow(new_image),plt.title('Mean filter')
	# plt.xticks([]), plt.yticks([])
	# plt.show()
	return contextual

def calculate_variance(image):
	# dimension = image.ndim
	# if dimension == 2:
	# 	variance = (pow(np.sum((np.sum(image,axis=1)),axis = 0),2)/image.size - 
	# 		pow(np.sum((np.sum(image,axis=1)),axis = 0)/image.size,2))
	# else dimension == 3:
	# 	variance = (pow(np.sum((np.sum(image,axis=1)),axis = 0),2)/image.size - 
	# 		pow(np.sum((np.sum(image,axis=1)),axis = 0)/image.size,2))
	variance = (pow(np.sum((np.sum(image,axis=1)),axis = 0),2)/image.size - 
			pow(np.sum((np.sum(image,axis=1)),axis = 0)/image.size,2))
	return variance


if __name__ == '__main__':
	# image = cv2.imread('scene.png')
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	dicomfile = dcmread('Image (0001).dcm')
	image = dicomfile.pixel_array
	print(image)
	print(calculate_contextual(image))