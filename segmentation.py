import numpy as np
import cv2
from pydicom import dcmread
from matplotlib import pyplot as plt 
from PIL import Image, ImageFilter
import math
import imageio
from sklearn.metrics import confusion_matrix
import nibabel as nib
import os
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

def smoothing(image, choice = 1):
	
	# theta = 0.1,S = 8,R = 2,alpha = 10,n_iter = 5
	# theta = 0.25,S = 8, R = 1, alpha = 15 , n_iter = 5
	# theta = 0.1,S = 6,R = 2,alpha = 15,n_iter = 5
	# theta = 0.25,S = 8,R = 2,alpha = 10,n_iter = 5
	theta = 0.1
	S = 6
	R = 2 
	alpha = 15 
	n_iter = 5
	if choice == 2:
		theta = 0.25
		S = 8
		R = 2
		alpha = 10
		n_iter = 5

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

def cluster(image):
	
	# reshape image

	pixel_values = image.reshape((image.shape[0]*image.shape[1],1))

	# convert to float
	
	pixel_values = np.float32(pixel_values)
	
	# define stopping criteria
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)	

	# number of clusters (K)
	
	k = 3

	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


	# convert back to 8 bit values
	
	centers = np.uint8(centers)

	# flatten the labels array
	
	labels = labels.flatten()

	min_center = np.amin(centers)
	max_center = np.amax(centers)

	centers = np.where(centers == min_center, 0 , centers)
	centers = np.where(centers == max_center, 255, centers)
	
	# convert all pixels to the color of the centroids
	
	segmented_image = centers[labels.flatten()]
	# reshape back to the original image dimension
	
	segmented_image = segmented_image.reshape(image.shape)
	return segmented_image, centers 

def cluster_2(image):
	
	# reshape image

	pixel_values = image.reshape((image.shape[0]*image.shape[1],1))

	# convert to float
	
	pixel_values = np.float32(pixel_values)
	
	# define stopping criteria
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)	

	# number of clusters (K)
	
	k = 2

	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


	# convert back to 8 bit values
	
	centers = np.uint8(centers)

	# flatten the labels array
	
	labels = labels.flatten()

	min_center = min(centers[0],centers[1])
	# max_center = np.amax(centers)

	centers = np.where(centers == min_center, 0 , 255)
	# centers = np.where(centers == max_center, 255, centers)
	
	# convert all pixels to the color of the centroids
	
	segmented_image = centers[labels.flatten()]
	# reshape back to the original image dimension
	
	segmented_image = segmented_image.reshape(image.shape)
	return segmented_image

def compute_iou(y_pred, y_true):
	# ytrue, ypred is a flatten vector
	y_pred = y_pred.flatten()
	y_true = y_true.flatten()
	current = confusion_matrix(y_true, y_pred, labels=[0, 1])
	# compute mean iou
	intersection = np.diag(current)
	ground_truth_set = current.sum(axis=1)
	predicted_set = current.sum(axis=0)
	union = ground_truth_set + predicted_set - intersection
	IoU = intersection / union.astype(np.float32)
	return np.mean(IoU)

def run_with_png(image, ground_truth_img):
	
	# ground_truth_img = imageio.imread(mask_path)
	ground_truth_img_copy = ground_truth_img.copy()
	ground_truth_img_copy = np.where(ground_truth_img_copy < 255, 0,255)
	ground_truth_img = np.where(ground_truth_img == 255, 1 , 0)
	# ground_truth_img = np.where(ground_truth_img == 3, 1 , 0)
	# image = imageio.imread(img_path)
	# print(image.dtype)


	#Contrast stretching
	# min_pix = np.amin(image)
	# max_pix = np.amax(image)
	# image = ((image-min_pix)/(max_pix - min_pix))*255
	
	smoothed_img = smoothing(image)
	min_pix = np.amin(smoothed_img)
	max_pix = np.amax(smoothed_img)
	smoothed_img = ((smoothed_img-min_pix)/(max_pix - min_pix))*255

	clustered_img, centers = cluster(smoothed_img)
	centers = np.sort(centers)
	centers = np.delete(centers,0)
	clustered_img = cv2.convertScaleAbs(clustered_img)
	image_contour = image.copy()
	image_contour = cv2.convertScaleAbs(image_contour)
	
	clustered_slices = np.zeros((image.shape[0],image.shape[1],len(centers)))
	clustered_slices_copy = clustered_slices.copy()
	#k > 2
	for i in range(len(centers)):
		clustered_slices[:,:,i] = np.where(clustered_img == centers[i], clustered_img, 0)
		clustered_slices_copy[:,:,i] = np.where(clustered_img == centers[i], 1, 0)

	#k = 2
	# clustered_slices = np.where(clustered_img == centers, clustered_img, 0)
	# clustered_slices_copy = np.where(clustered_img == centers, 1, 0)


	# circles = cv2.HoughCircles(clustered_img, cv2.HOUGH_GRADIENT,1,20,param1=20,param2=10,minRadius=14,maxRadius=25)
	# if circles is not None:
	# 	circles = np.round(circles[0,:]).astype("int")
	# 	for (x,y,r) in circles:
	# 		cv2.circle(clustered_img, (x,y),r, (25,127,125),2)
	# 		cv2.rectangle(clustered_img, (x - 50, y - 30), (x + 40, y + 30), (0, 128, 255))
	# 		slicedImage = clustered_img[y-30:y+30, x-50:x+40]
	

	# print(clustered_img.shape)
	# copy_image = clustered_img.copy()
	contours_list = []
	img_center_x = int(image.shape[1]/2)
	img_center_y = int(image.shape[0]/2)
	min_dist = 60
	min_dist_fix = 1000
	clustered_slices_copy = cv2.convertScaleAbs(clustered_slices_copy)
	# print(clustered_slices.shape)
	#for k > 2
	for i in range(len(centers)):
		kernel = np.ones((3,3),np.uint8)
		clustered_slices_copy[:,:,i] = cv2.morphologyEx(clustered_slices_copy[:,:,i], cv2.MORPH_CLOSE, kernel)
		cnt, _ = cv2.findContours(clustered_slices_copy[:,:,i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
		contours_list.append(cnt)
	
	#for k = 2
	# kernel = np.ones((3,3),np.uint8)
	# clustered_slices_copy = cv2.morphologyEx(clustered_slices_copy, cv2.MORPH_CLOSE, kernel)
	# cnt, _ = cv2.findContours(clustered_slices_copy, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	# contours_list.append(cnt)
	
	print(len(contours_list))
	# print(cnt.dtype)
	
	lv_index_fix = 0
	lv_area_fix = 0
	im = cv2.cvtColor(image_contour,cv2.COLOR_GRAY2BGR)
	
	list_lv_index = []
	list_min_dist = []
	# list_lv_area = []
	for j in range(len(contours_list)):
		min_dist = 60
		lv_index = 0
		cnt = contours_list[j]
		if not cnt:
			list_lv_index.append(lv_index_fix)
			list_min_dist.append(min_dist_fix)
			# list_lv_area.append(lv_area_fix)
			continue
		print("Length contour " + str(j) + " " + str(len(cnt)))
		for i in range(len(cnt)):
			M = cv2.moments(cnt[i])
			# print(M)
			if 100 < M['m00'] < 3000:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				# print(str(cx) + ":" + str(cy))
				dist = abs(cx - img_center_x) + abs(cy - img_center_y)
			else:
				continue
			if dist < min_dist:
				min_dist = dist
				lv_index = i
				# lv_area = M['m00']
		list_lv_index.append(lv_index)
		list_min_dist.append(min_dist)
		# list_lv_area.append(lv_area)
	mask = np.zeros_like(image)
	mask = cv2.convertScaleAbs(mask)
	print("mask dtype "+str(mask.dtype))

	
	min_index = list_min_dist.index(min(list_min_dist))
	print(min_index)
	final_lv_index = list_lv_index[min_index]
	# print(final_lv_index)

	# print(contours_list)
	final_contour = contours_list[min_index]
	# print(final_contour)


	# poligon_mask = np.zeros(image.shape)
	# epsilon = 0.001
	# approx = cv2.approxPolyDP(cnt,epsilon,True)

	
	# cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),cv2.FILLED)

	# cv2.imshow('Draw Contours',im)
	# cv2.waitKey(0)
	# lv_contour = cnt[lv_index].reshape(-1,2)
	# for (x,y) in lv_contour:
		# cv2.circle(im,(x,y),1,(0,255,0),cv2.FILLED)

	if final_contour:
		
		cv2.drawContours(mask,final_contour,final_lv_index,color=1,thickness=-1)
		area = cv2.contourArea(final_contour[final_lv_index])
		area1 = np.sum(mask)
		print(area1)
		# cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),cv2.FILLED)
		print(area)
	# test = contours_list[1]
	# if test:
	# 	print("true")
	# cv2.drawContours(im, contours_list[1], -1 , (0,255,0),1)
	#k > 2
	cv2.drawContours(im, final_contour[final_lv_index], -1 , (0,255,0),1)
	
	# k = 2
	# cv2.drawContours(im, contours_list[0], -1 , (0,255,0),1)
	#Calculate RMS errors
	# N = image.shape[0]*image.shape[1]
	# sq_diff = np.zeros_like(image)
	# sq_diff = pow(mask - ground_truth_img,2)
	# rms = math.sqrt(np.sum((np.sum(sq_diff,axis=1)),axis = 0)/N)
	# print("RMS error: " + str(rms))

	iou = compute_iou(mask,ground_truth_img)
	print("IOU: " + str(iou))

	# return iou
	# params = cv2.SimpleBlobDetector_Params()
	# params.filterByCircularity = True
	# params.minCircularity = 0.9
	# ver = (cv2.__version__).split('.')
	# if int(ver[0]) < 3 :
	# 	detector = cv2.SimpleBlobDetector(params)
	# else : 
	# 	detector = cv2.SimpleBlobDetector_create(params)
	
	# # detector.empty()
	# keypoints = detector.detect(image)
	# print(len(keypoints))
	# blank = np.zeros((1, 1))
	# im_with_keypoints = cv2.drawKeypoints(image, keypoints, blank ,(0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imshow("Keypoints", im_with_keypoints)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# cv2.imwrite('clustered_img.png',im_with_keypoints)

	# contextual_dist = calculate_contextual(image,2,0.2)
	# out = np.exp(contextual_dist*(-1)*50)

	plt.figure(figsize=(11,6))
	plt.subplot(231), plt.imshow(image,cmap='gray'),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(232), plt.imshow(smoothed_img,cmap='gray'),plt.title('Adaptive smoothing')
	plt.xticks([]), plt.yticks([])
	plt.subplot(233), plt.imshow(clustered_img,cmap='gray'),plt.title('Cluster')
	plt.xticks([]), plt.yticks([])
	plt.subplot(234), plt.imshow(mask),plt.title('LV cavity segmented')
	plt.xticks([]), plt.yticks([])
	plt.subplot(235), plt.imshow(im),plt.title('Contours image')
	plt.xticks([]), plt.yticks([])
	plt.subplot(236), plt.imshow(ground_truth_img_copy),plt.title('Ground truth image')
	plt.xticks([]), plt.yticks([])
	plt.show()

def segment_img(image):
	smoothed_img = smoothing(image)
	min_pix = np.amin(smoothed_img)
	max_pix = np.amax(smoothed_img)
	smoothed_img = ((smoothed_img-min_pix)/(max_pix - min_pix))*255

	clustered_img, centers = cluster(smoothed_img)
	centers = np.sort(centers)
	centers = np.delete(centers,0)
	clustered_img = cv2.convertScaleAbs(clustered_img)
	image_contour = image.copy()
	image_contour = cv2.convertScaleAbs(image_contour)
	
	clustered_slices = np.zeros((image.shape[0],image.shape[1],len(centers)))
	clustered_slices_copy = clustered_slices.copy()
	
	for i in range(len(centers)):
		clustered_slices[:,:,i] = np.where(clustered_img == centers[i], clustered_img, 0)
		clustered_slices_copy[:,:,i] = np.where(clustered_img == centers[i], 1, 0)
	
	contours_list = []
	img_center_x = int(image.shape[1]/2)
	img_center_y = int(image.shape[0]/2)
	min_dist = 60
	min_dist_fix = 1000
	clustered_slices_copy = cv2.convertScaleAbs(clustered_slices_copy)
	# print(clustered_slices.shape)
	for i in range(len(centers)):
		kernel = np.ones((3,3),np.uint8)
		clustered_slices_copy[:,:,i] = cv2.morphologyEx(clustered_slices_copy[:,:,i], cv2.MORPH_CLOSE, kernel)
		cnt, _ = cv2.findContours(clustered_slices_copy[:,:,i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
		contours_list.append(cnt)

	lv_index_fix = 0
	lv_area_fix = 0
	im = cv2.cvtColor(image_contour,cv2.COLOR_GRAY2BGR)
	
	list_lv_index = []
	list_min_dist = []
	# list_lv_area = []
	for j in range(len(contours_list)):
		min_dist = 60
		lv_index = 0
		cnt = contours_list[j]
		if not cnt:
			list_lv_index.append(lv_index_fix)
			list_min_dist.append(min_dist_fix)
			# list_lv_area.append(lv_area_fix)
			continue
		# print("Length contour " + str(j) + " " + str(len(cnt)))
		for i in range(len(cnt)):
			M = cv2.moments(cnt[i])
			# print(M)
			if 100 < M['m00'] < 3000:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				# print(str(cx) + ":" + str(cy))
				dist = abs(cx - img_center_x) + abs(cy - img_center_y)
			else:
				continue
			if dist < min_dist:
				min_dist = dist
				lv_index = i
				# lv_area = M['m00']
		list_lv_index.append(lv_index)
		list_min_dist.append(min_dist)
		# list_lv_area.append(lv_area)
	mask = np.zeros_like(image)
	mask = cv2.convertScaleAbs(mask)
	# print("mask dtype "+str(mask.dtype))

	
	min_index = list_min_dist.index(min(list_min_dist))
	# print(min_index)
	final_lv_index = list_lv_index[min_index]
	# print(final_lv_index)

	# print(contours_list)
	final_contour = contours_list[min_index]

	if final_contour:
		
		cv2.drawContours(mask,final_contour,final_lv_index,color=1,thickness=-1)
		area = cv2.contourArea(final_contour[final_lv_index])
		# cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),cv2.FILLED)
		# print(area)
	# test = contours_list[1]
	# if test:
	# 	print("true")
	# cv2.drawContours(im, contours_list[1], -1 , (0,255,0),cv2.FILLED)
	cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),2)
	return im, area

def segment_img_with_center(image, lv_center_x, lv_center_y):
	smoothed_img = smoothing(image)
	min_pix = np.amin(smoothed_img)
	max_pix = np.amax(smoothed_img)
	smoothed_img = ((smoothed_img-min_pix)/(max_pix - min_pix))*255

	clustered_img, centers = cluster(smoothed_img)
	centers = np.sort(centers)
	centers = np.delete(centers,0)
	clustered_img = cv2.convertScaleAbs(clustered_img)
	image_contour = image.copy()
	image_contour = cv2.convertScaleAbs(image_contour)
	
	clustered_slices = np.zeros((image.shape[0],image.shape[1],len(centers)))
	clustered_slices_copy = clustered_slices.copy()
	
	for i in range(len(centers)):
		clustered_slices[:,:,i] = np.where(clustered_img == centers[i], clustered_img, 0)
		clustered_slices_copy[:,:,i] = np.where(clustered_img == centers[i], 1, 0)
	
	contours_list = []
	img_center_x = lv_center_x
	img_center_y = lv_center_y
	if lv_center_x == 0 and lv_center_y == 0:

		img_center_x = int(image.shape[1]/2)
		img_center_y = int(image.shape[0]/2)
	min_dist = 60
	min_dist_fix = 1000
	clustered_slices_copy = cv2.convertScaleAbs(clustered_slices_copy)
	# print(clustered_slices.shape)
	for i in range(len(centers)):
		kernel = np.ones((3,3),np.uint8)
		clustered_slices_copy[:,:,i] = cv2.morphologyEx(clustered_slices_copy[:,:,i], cv2.MORPH_CLOSE, kernel)
		cnt, _ = cv2.findContours(clustered_slices_copy[:,:,i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
		contours_list.append(cnt)

	lv_index_fix = 0
	lv_area_fix = 0
	im = cv2.cvtColor(image_contour,cv2.COLOR_GRAY2BGR)
	
	list_lv_index = []
	list_min_dist = []
	# list_lv_area = []
	for j in range(len(contours_list)):
		min_dist = 60
		lv_index = 0
		cnt = contours_list[j]
		if not cnt:
			list_lv_index.append(lv_index_fix)
			list_min_dist.append(min_dist_fix)
			# list_lv_area.append(lv_area_fix)
			continue
		# print("Length contour " + str(j) + " " + str(len(cnt)))
		for i in range(len(cnt)):
			M = cv2.moments(cnt[i])
			# print(M)
			if 100 < M['m00'] < 3000:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				# print(str(cx) + ":" + str(cy))
				dist = abs(cx - img_center_x) + abs(cy - img_center_y)
			else:
				continue
			if dist < min_dist:
				min_dist = dist
				lv_index = i
				# lv_area = M['m00']
		list_lv_index.append(lv_index)
		list_min_dist.append(min_dist)
		# list_lv_area.append(lv_area)
	mask = np.zeros_like(image)
	mask = cv2.convertScaleAbs(mask)
	# print("mask dtype "+str(mask.dtype))

	
	min_index = list_min_dist.index(min(list_min_dist))
	# print(min_index)
	final_lv_index = list_lv_index[min_index]
	# print(final_lv_index)

	# print(contours_list)
	final_contour = contours_list[min_index]

	if final_contour:
		
		cv2.drawContours(mask,final_contour,final_lv_index,color=1,thickness=-1)
		area = cv2.contourArea(final_contour[final_lv_index])

		# cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),cv2.FILLED)
		# print(area)
	# test = contours_list[1]
	# if test:
	# 	print("true")
	# cv2.drawContours(im, contours_list[1], -1 , (0,255,0),cv2.FILLED)
	cv2.drawContours(im, final_contour, final_lv_index , (0,255,0),2)
	return im, area

def lv_localize(image):

	# ground_truth_img = np.where(ground_truth_img == 3, 1, 0)

	smoothed_img = smoothing(image, choice = 2)
	clustered_img = cluster_2(smoothed_img)	
	# images_copy1 = image.copy()
	clustered_img = cv2.convertScaleAbs(clustered_img)
	# images_copy1 = cv2.cvtColor(images_copy1,cv2.COLOR_GRAY2BGR)
	center_x = 0
	center_y = 0

	width, height = image.shape
	img_cut = np.zeros((width,height))
	new_width1 = int(width/4)
	new_width2 = int(3*width/4)
	new_height1 = int(height/4)
	new_height2 = int(3*height/4)
	img_cut[new_width1:new_width2,new_height1:new_height2] = clustered_img[new_width1:new_width2,new_height1:new_height2]
	img_cut = cv2.convertScaleAbs(img_cut)

	circles = cv2.HoughCircles(img_cut, cv2.HOUGH_GRADIENT,1,200,param1=25,param2=10,minRadius=8,maxRadius=25)
	if circles is not None:
		for j in circles[0,:]:
			# cv2.circle(images_copy1,(j[0],j[1]),j[2],color=1,thickness=-1)
			# cv2.circle(images_copy1,(j[0],j[1]),2,(0,255,0),3)
			center_x = j[0]
			center_y = j[1]
	# # iou = compute_iou(mask,ground_truth_img)
	# plt.figure(figsize=(11,6))
	# plt.subplot(231), plt.imshow(ground_truth_img,cmap='gray'),plt.title('Original')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(232), plt.imshow(images_copy1,cmap='gray'),plt.title('Adaptive smoothing')
	# plt.xticks([]), plt.yticks([])
	# # plt.subplot(233), plt.imshow(ed_slice_variance,cmap='gray'),plt.title('Cluster')
	# # plt.xticks([]), plt.yticks([])
	# # plt.subplot(234), plt.imshow(mask),plt.title('LV cavity')
	# # plt.xticks([]), plt.yticks([])
	# # plt.subplot(235), plt.imshow(im),plt.title('Contours image')
	# # plt.xticks([]), plt.yticks([])
	# # plt.subplot(236), plt.imshow(ground_truth_img_copy),plt.title('Ground truth image')
	# # plt.xticks([]), plt.yticks([])
	# plt.show()

	return center_x, center_y

if __name__ == '__main__':
	

	img_path = '../training/patient025/original_2D_ED_slice/original_2D_05.png'
	mask_path = "../training/patient025/original_2D_ED_mask/original_2D_05.png"
	sa_zip_file = '../training/patient068/patient068_4d.nii.gz'

	image = imageio.imread(img_path)
	ground_truth_img = imageio.imread(mask_path)
	run_with_png(image, ground_truth_img)
	# lv_localize(image)

	#k > 2 and separate cluster
	# smoothed_img = smoothing(image)
	# clustered_img, centers = cluster(smoothed_img)
	# centers = np.sort(centers)
	# centers = np.delete(centers,0)		
	# clustered_slices = np.zeros((image.shape[0],image.shape[1],len(centers)))
	# images_copy1 = image.copy()
	# images_copy2 = image.copy()
	# clustered_slices = cv2.convertScaleAbs(clustered_slices)
	# images_copy1 = cv2.cvtColor(images_copy1,cv2.COLOR_GRAY2BGR)
	# images_copy2 = cv2.cvtColor(images_copy2,cv2.COLOR_GRAY2BGR)

	# for i in range(len(centers)):
	# 	clustered_slices[:,:,i] = np.where(clustered_img == centers[i], clustered_img, 0)
	# 	circles = cv2.HoughCircles(clustered_slices[:,:,i], cv2.HOUGH_GRADIENT,1,200,param1=20,param2=10,minRadius=8,maxRadius=25)
	# 	if circles is not None:
	# 		if i == 0:
	# 			for j in circles[0,:]:
	# 				cv2.circle(images_copy1,(j[0],j[1]),j[2],color=1,thickness=-1)
	# 				cv2.circle(images_copy1,(j[0],j[1]),2,(0,0,255),3)
	# 		if i == 1:
	# 			for j in circles[0,:]:
	# 				cv2.circle(images_copy2,(j[0],j[1]),j[2],color=1,thickness=-1)
	# 				cv2.circle(images_copy2,(j[0],j[1]),2,(0,0,255),3)
	
	#k = 2
	# smoothed_img = smoothing(image)
	# clustered_img = cluster_2(smoothed_img)	
	# images_copy1 = image.copy()
	# clustered_img = cv2.convertScaleAbs(clustered_img)
	# images_copy1 = cv2.cvtColor(images_copy1,cv2.COLOR_GRAY2BGR)

	# circles = cv2.HoughCircles(clustered_img, cv2.HOUGH_GRADIENT,1,200,param1=20,param2=10,minRadius=8,maxRadius=25)
	# if circles is not None:
	# 	for j in circles[0,:]:
	# 		cv2.circle(images_copy1,(j[0],j[1]),j[2],color=1,thickness=-1)
	# 		cv2.circle(images_copy1,(j[0],j[1]),2,(0,0,255),3)

	# plt.figure(figsize=(11,6))
	# plt.subplot(231), plt.imshow(image,cmap='gray'),plt.title('Original')
	# plt.xticks([]), plt.yticks([])
	# # plt.subplot(232), plt.imshow(smoothed_img,cmap='gray'),plt.title('Adaptive smoothing')
	# # plt.xticks([]), plt.yticks([])
	# plt.subplot(233), plt.imshow(clustered_img,cmap='gray'),plt.title('Cluster')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(234), plt.imshow(images_copy1),plt.title('LV cavity')
	# plt.xticks([]), plt.yticks([])
	# # plt.subplot(235), plt.imshow(images_copy2),plt.title('Contours image')
	# # plt.xticks([]), plt.yticks([])
	# plt.subplot(236), plt.imshow(ground_truth_img),plt.title('Ground truth image')
	# plt.xticks([]), plt.yticks([])
	# plt.show()

	#k=3
	# smoothed_img = smoothing(image)
	# clustered_img,centers = cluster(image)	
	# images_copy1 = image.copy()
	# smoothed_img = cv2.convertScaleAbs(smoothed_img)
	# clustered_img = cv2.convertScaleAbs(clustered_img)
	# images_copy1 = cv2.cvtColor(images_copy1,cv2.COLOR_GRAY2BGR)

	# circles = cv2.HoughCircles(smoothed_img, cv2.HOUGH_GRADIENT,1,200,param1=20,param2=15,minRadius=8,maxRadius=25)
	# if circles is not None:
	# 	for j in circles[0,:]:
	# 		cv2.circle(images_copy1,(j[0],j[1]),j[2],(0,255,0),thickness=-1)
	# 		cv2.circle(images_copy1,(j[0],j[1]),2,(0,0,255),3)

	# plt.figure(figsize=(11,6))
	# plt.subplot(231), plt.imshow(image,cmap='gray'),plt.title('Original')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(232), plt.imshow(smoothed_img,cmap='gray'),plt.title('Adaptive smoothing')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(233), plt.imshow(clustered_img,cmap='gray'),plt.title('Cluster')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(234), plt.imshow(images_copy1),plt.title('LV cavity')
	# plt.xticks([]), plt.yticks([])

	# plt.subplot(236), plt.imshow(ground_truth_img),plt.title('Ground truth image')
	# plt.xticks([]), plt.yticks([])
	# plt.show()



	# image = nib.load(sa_zip_file)
	# image_data = image.get_fdata()
	# for i in range(image_data[:,:,:,0].shape[2]):

	# 	f = np.fft.fft2(image_data[:,:,i,])
	# 	# fh = np.absolute(np.fft.)
	# 	print(f.shape)




	# clustered_img = cluster(image)

	# plt.imshow(clustered_img)
	# plt.show()
	# a = nib.load(sa_zip_file)
	# img = np.array(a.dataobj)

	# plt.hist(img[:,:,9])






	# np.seterr(divide='ignore', invalid='ignore')
	# data_dir = '/home/bao/Documents/vibot_m1_slides/s2/medical image analysis/training/{}'
	# subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 101)]
	# iou_lists = []

	# count_non_lv_ed = 0
	# count_non_lv_es = 0

	# count_lv_ed = 0
	# count_lv_es = 0

	# for subject in subjects:
	# 	subject_dir = data_dir.format(subject)
	# 	sa_zip_file = os.path.join(subject_dir,'{}_4d.nii.gz'.format(subject))
		
	# 	nifty_img = nib.load(sa_zip_file)
	# 	img = np.array(nifty_img.dataobj)

	# 	cfg_file = os.path.join(subject_dir,'Info.cfg')
	# 	slice_info = []
	# 	with open(cfg_file,"r") as data:
	# 		r_count = 0
	# 		for line in data:
	# 			if r_count == 2:
	# 				break

	# 			a = line.split(":")
	# 			slice_info.append(a)
	# 			r_count += 1
		
	# 	ED_slice = int(slice_info[0][1])
	# 	ES_slice = int(slice_info[1][1])


	# 	ED_imgs = img[:,:,:,ED_slice]
	# 	ES_imgs = img[:,:,:,ES_slice]

	# 	ED_mask_zip_file = os.path.join(subject_dir,'{}_frame{}_gt.nii.gz'.format(subject,str(ED_slice).zfill(2)))
	# 	ES_mask_zip_file = os.path.join(subject_dir,'{}_frame{}_gt.nii.gz'.format(subject,str(ES_slice).zfill(2)))
		
	# 	ED_nifty_masks = nib.load(ED_mask_zip_file)
	# 	ED_masks = np.array(ED_nifty_masks.dataobj)
		
	# 	ES_nifty_masks = nib.load(ES_mask_zip_file)
	# 	ES_masks = np.array(ES_nifty_masks.dataobj)


	# 	count1 = 0
	# 	count2 = 0
	# 	for j in range(ED_masks.shape[2]):
	# 		ED_mask = ED_masks[:,:,j]
	# 		max_gt_val = np.amax(ED_mask)
	# 		if max_gt_val != 3:
	# 			count1 += 1

	# 	for j in range(ES_masks.shape[2]):
	# 		ES_mask = ES_masks[:,:,j]
	# 		max_gt_val = np.amax(ES_mask)
	# 		if max_gt_val != 3:
	# 			count2 += 1

	# 	count_non_lv_ed += count1
	# 	count_non_lv_es += count2

	# 	count_lv_ed = count_lv_ed + ED_masks.shape[2] - count1
	# 	count_lv_es = count_lv_es + ED_masks.shape[2] - count2


	# N = 2
	# lv_means = (count_lv_ed, count_lv_es)
	# non_lv_means = (count_non_lv_ed, count_non_lv_es)

	# ind = np.arange(N) 
	# width = 0.15       
	# # fig, ax = plt.subplots()  
	# plt.figure(figsize=(6,9))  
	# rect1 = plt.bar(ind, lv_means, width, label='Contain Left Ventricle')

	# rect2 = plt.bar(ind + width, non_lv_means, width,
	#     label='Not contain Left Ventricle')

	# plt.ylabel('Number of images')
	# plt.title('Number of images in ED slices and ES slices')

	# plt.xticks(ind + width / 2, ('ED slice', 'ES slice'))
	# plt.legend(loc='best')
	# for rect in rect1:
	# 	height = rect.get_height()
	# 	plt.text(rect.get_x() + rect.get_width()/2., height+0.1,
	# 	'%d' % int(height),
	# 	ha='center', va='bottom')
	# for rect in rect2:
	# 	height = rect.get_height()
	# 	plt.text(rect.get_x() + rect.get_width()/2., height+0.1,
	# 	'%d' % int(height),
	# 	ha='center', va='bottom')
	# # plt.show()
	# plt.savefig('chart.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
	# 	iou_list = []
		
	# 	for j in range(ED_imgs.shape[2]):
	# 		ED_img = ED_imgs[:,:,j]
	# 		ED_mask = ED_masks[:,:,j]
	# 		max_gt_val = np.amax(ED_mask)
	# 		if max_gt_val != 3:
	# 			continue
	# 		iou = run_with_png(ED_img,ED_mask)
	# 		iou_list.append(iou)
		
	# 	for j in range(ES_imgs.shape[2]):
	# 		ES_img = ES_imgs[:,:,j]
	# 		ES_mask = ES_masks[:,:,j]
	# 		max_gt_val = np.amax(ES_mask)
	# 		if max_gt_val != 3:
	# 			continue
	# 		iou = run_with_png(ES_img,ES_mask)
	# 		iou_list.append(iou)

	# 	mIOU = sum(iou_list)/len(iou_list)
	# 	iou_lists.append(mIOU)
	# 	iou_list.clear()
	# 	slice_info.clear()	
	# mIOUs = sum(iou_lists)/len(iou_lists)
	# print(mIOUs) 

	
