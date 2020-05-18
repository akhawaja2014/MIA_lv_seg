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
	

	# img_path = '../training/patient001/original_2D_ED_slice/original_2D_01.png'
	# mask_path = "../training/patient001/original_2D_ED_mask/original_2D_01.png"
	sa_zip_file = '../training/patient068/patient068_4d.nii.gz'

	image = nib.load(sa_zip_file)
	image_data = image.get_fdata()
	width, height, frame, slice_num = image_data.shape
	
	ed_slice = image_data[:,:,:,0]

	sum_image = np.sum(ed_slice, axis = 2)

	mean_image = sum_image/frame

	print(mean_image.shape)