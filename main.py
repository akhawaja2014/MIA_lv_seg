import numpy as np
import cv2
from pydicom import dcmread
from matplotlib import pyplot as plt 
from PIL import Image, ImageFilter
import math
import imageio
# def get_neighboring_pixels(image,R,y,x,isFloat):
# 	#create an array with dimension of kernel, same type as image
# 	#neighbor = np.zeros((R,R))
# 	#neighbor = neighbor.astype(image.dtype)
# 	#copy value inside kernel to created array
# 	neighbor = image[y-R:y+R+1,x-R:x+R+1]
# 	return neighbor
def calculate_local(image):
	
	h = image.shape[0]
	w = image.shape[1]

	local_dist = np.zeros_like(image, dtype=float)
	# print(local_dist.shape)
	for row in range(1,h-1-1):
		for col in range(1,w-1-1):
			# sum = abs(image[row+1,col] - image[row-1,col]) + abs(image[row,col+1]-image[row,col-1]) + \
			# 	abs(image[row+1,col+1] - image[row-1,col-1]) + abs(image[row+1,col-1]-image[row-1,col+1])
			
			H = abs(int(image[row+1,col]) - int(image[row-1,col]))
			V = abs(int(image[row,col+1]) - int(image[row,col-1]))
			C = abs(int(image[row+1,col+1]) - int(image[row-1,col-1]))
			D = abs(int(image[row+1,col-1]) - int(image[row-1,col+1]))
			sum = H + V + C + D
			local_dist[row,col] = sum/4
			# print(str(H) + ' ' + str(V) + ' ' + str(C) + ' ' + str(D) + ' ' +  str(sum)+ ' '+ str(local_dist[row,col]))
	return local_dist

def calculate_contextual(image, R, theta):
	
	#kernel size = 2R+1
	# R = 1

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

	contextual = np.where(contextual < theta, 0, contextual)
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

def smoothing(image, theta = 0.1, S = 8, R = 2, alpha = 10 , n_iter = 5):
	
	contextual_dist = calculate_contextual(image, R, theta)
	# local_dist = calculate_local(image)
	contextual_eff = np.exp(contextual_dist*(-1)*alpha)
	# local_eff = np.exp((-1)*local_dist/S)
	# out = np.zeros_like(image)
	# out = image
	h = image.shape[0]
	w = image.shape[1]
	center = R + 1
	for i in range(n_iter):
		local_dist = calculate_local(image)
		local_eff = np.exp((-1)*local_dist/S)
		smoothing_matrix = calculate_smoothing_matrix(image, contextual_eff, local_eff,R)
		image = image + smoothing_matrix
		print('Iteration ' + str(i))
	return image

def calculate_smoothing_matrix(image, contextual_eff, local_eff, R):
	
	h = image.shape[0]
	w = image.shape[1]

	smoothing_matrix = np.zeros_like(image,dtype=float)
	
	for row in range(R,h-1-R):
		for col in range(R,w-1-R):
			
			kernel_img = np.zeros((2*R+1,2*R+1))
			kernel_img = image[row-R:row+R+1,col-R:col+R+1]
			
			kernel_contextual = np.zeros((2*R+1,2*R+1))
			kernel_contextual = contextual_eff[row-R:row+R+1,col-R:col+R+1]
			
			kernel_local = np.zeros((2*R+1,2*R+1))
			kernel_local = local_eff[row-R:row+R+1,col-R:col+R+1]
			
			
			center_value = float(kernel_img[R,R])
			center_matrix = np.full((2*R+1,2*R+1),center_value)

			difference = kernel_img - center_matrix

			numerator = np.sum((np.sum(kernel_local*kernel_contextual*difference,axis=1)),axis=0)

			denumerator = np.sum((np.sum(kernel_local*kernel_contextual,axis=1)),axis=0) - float(kernel_local[R,R])*float(kernel_contextual[R,R])
			if denumerator == 0:
				smoothing_matrix[row,col] = 0
			else: 
				smoothing_matrix[row,col] = (float(kernel_contextual[R,R])*float(numerator))/float(denumerator)
			# print("Kernel Img")
			# print(kernel_img)
			# print(kernel_local)
			# print(kernel_contextual)
			# print(center_matrix)
			# print(difference)
			# print(numerator)
			# print(denumerator)
			# print(smoothing_matrix[row,col])
	return smoothing_matrix


if __name__ == '__main__':
	# image = cv2.imread('scene.png')
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# dicomfile = dcmread('Image (0001).dcm')
	# image = dicomfile.pixel_array
	# print(image)
	# print(calculate_contextual(image))
	# np.savetxt('test.txt',out)
	image = imageio.imread('original_2D_02_28.png')
	out = smoothing(image)
	# contextual_dist = calculate_contextual(image,2,0.2)
	# out = np.exp(contextual_dist*(-1)*50)
	plt.figure(figsize=(11,6))
	plt.subplot(121), plt.imshow(image,cmap='gray'),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(out,cmap='gray'),plt.title('Adaptive smoothing')
	plt.xticks([]), plt.yticks([])
	plt.show()