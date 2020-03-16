import nibabel as nib
import os
from nibabel.testing import data_path
import numpy as np

#import cv2
#example_filename = os.path.join('patient001_4d.nii')
img = nib.load('/home/arsalan/Desktop/hello/MIA/CMD03Gate1.nii')
#print(img)
data = img.get_data()
#print(data.shape)
#%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.figure()
plt.imshow(img.get_data()[:,:,2])
#plt.show()

array_img = np.array(img.dataobj)
print(array_img)



sz = array_img.shape
Edirs = np.array([[1, 0,-1, 0],[0, 1, 0, -1],[1, 1, -1, -1],[1, -1, -1, 1]])
print(Edirs)
localDiscont = np.zeros(sz)
#print(localDiscont)
#print(localDiscont[125,100,:])

for row in range(1,sz[0]-1):
    #print(row)
    for col in range(1,sz[1]-1):
        #print(col)
        for e in range(0,4):
            #print(e)
            localDiscont[row,col,:]    =     localDiscont[row,col,:]    +     abs(   array_img[row+Edirs[e,0] , col+Edirs[e,1], :]      -      array_img[row+Edirs[e,2] , col+Edirs[e,3], :] )
            

localDiscont = localDiscont/4
print(localDiscont)
fig = plt.figure()
plt.imshow(localDiscont[:,:,2])
plt.show()


"""
#def smothin:
#    pass

def convolution:
    pass


def measuring_local_discontinuity:
    E_H = abs(Ixp1_y - Ixm1_y)
    E_V = abs(Ix_yp1 -Ix_ym1)
    E_C = abs(Ixp1_yp1 - Ixm1_ym1)
    E_D = abs(Ixp1_ym1 - Ixm1_yp1)
    
    E_xy = E_H + E_V + E_C + E_D
    E_xy = E_xy/4
"""
