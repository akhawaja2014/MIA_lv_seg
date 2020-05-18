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



if __name__ == '__main__':
	

	img_path = '../training/patient020/original_2D_ED_slice/original_2D_01.png'
	mask_path = "../training/patient020/original_2D_ED_mask/original_2D_01.png"
	# sa_zip_file = '../training/patient068/patient068_4d.nii.gz'

	ground_truth_img = imageio.imread(img_path)
	width, height = ground_truth_img.shape
	img_cut = np.zeros((width,height))
	new_width1 = int(width/4)
	new_width2 = int(3*width/4)
	new_height1 = int(height/4)
	new_height2 = int(3*height/4)
	img_cut[new_width1:new_width2,new_height1:new_height2] = ground_truth_img[new_width1:new_width2,new_height1:new_height2]


	img_cut = cv2.convertScaleAbs(img_cut)
	circles = cv2.HoughCircles(img_cut, cv2.HOUGH_GRADIENT,1,200,param1=25,param2=10,minRadius=8,maxRadius=25)

	if circles is not None:
		for j in circles[0,:]:
			cv2.circle(img_cut,(j[0],j[1]),j[2],color=1,thickness=-1)
			cv2.circle(img_cut,(j[0],j[1]),2,(0,0,255),3)

	# image = nib.load(sa_zip_file)
	# image_data = image.get_fdata()
	# width, height, frame, slice_num = image_data.shape
	
	# ed_slice = image_data[:,:,:,0]

	# sum_image = np.sum(ed_slice, axis = 2)

	# mean_image = sum_image/frame

	# diff = np.zeros((width,height,frame))
	
	# for i in range(frame):
	# 	diff[:,:,i] = ed_slice[:,:,i] - mean_image
	# 	diff[:,:,i] = np.square(diff[:,:,i])

	# sum_variance = np.sum(diff, axis = 2)

	# ed_slice_variance = sum_variance/(frame-1)

	plt.figure(figsize=(11,6))
	plt.subplot(231), plt.imshow(ground_truth_img,cmap='gray'),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(232), plt.imshow(img_cut,cmap='gray'),plt.title('Adaptive smoothing')
	plt.xticks([]), plt.yticks([])
	# plt.subplot(233), plt.imshow(ed_slice_variance,cmap='gray'),plt.title('Cluster')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(234), plt.imshow(mask),plt.title('LV cavity')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(235), plt.imshow(im),plt.title('Contours image')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(236), plt.imshow(ground_truth_img_copy),plt.title('Ground truth image')
	# plt.xticks([]), plt.yticks([])
	plt.show()

