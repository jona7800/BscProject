"""
Script for writing digital twin as .tif.
For later 3d volumetric viewing.
"""
# Necessary packages:
import cv2 as cv
import tifffile as tff
import numpy as np
import os
import glob
from TwinLib import filenum


#Directory of digital twin
path1 = 'C:/Users/Jonathan/Documents/6.Semester/BachelorProject/Code/PreviousVprints/Atos_scan/*.jpg' 
#Directory of new tiff file
path2 = 'C:/Users/Jonathan/Documents/6.Semester/BachelorProject/Code'
#Name of twin
TwinName = 'Atos_scan'

scale_percent = 200
twinpaths = glob.glob(path1)
tiff_list = None
count = 0

img = cv.imread(twinpaths[0], cv.IMREAD_GRAYSCALE)
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)



for fname in sorted(twinpaths, key=filenum):
    
    if count != 0:
        #Open image
        twin = cv.imread(fname,  cv.IMREAD_GRAYSCALE)
        
        #Scale to correct viewing dimensions
        twin = cv.resize(twin, dim, interpolation = cv.INTER_AREA)
        twin = cv.flip(twin, 0)
         
        twin = twin[np.newaxis, ::]
    
        #Append image
        if count == 1:
            tiff_list = twin
        else:
            tiff_list = np.append(tiff_list, twin, axis=0)
        
    
    #Print saved images (for every tenth frame)
    if count%10 == 0 and count != 0:
        print('Saved layers: %i-%i' %(count-9,count))
    count += 1
    
tff.imsave(os.path.join(path2, TwinName + '.tif'), tiff_list)   
    

    
    