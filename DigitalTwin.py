"""
Main script for generating digital twin.

OBS. Set directories appropriately before running. 
"""
# Necessary packages:
import numpy as np
import cv2 as cv
import glob
import os
import TwinLib as TL
import time

"""
Initialization:
"""
#Main directory
path = 'C:/Users/Jonathan/Documents/6.Semester/BachelorProject/Code'
#Directory of perspective square image
perspectiveSquares = os.path.join(path, 'PiCamCal/Perspectives/PerspectiveSquares.jpg')
#Calibration images
calImgs = glob.glob(os.path.join(path, 'PiCamCal/*.jpg'))
#Images of 3d print
imgs = glob.glob(os.path.join(path, 'Prints/Cube_LEXT/*.jpg'))
#Set accordingly if Vprint is generated or not
perfectTwinExists = False
layerCheck = 70


"""
Create Perfect Twin
"""
if perfectTwinExists == False:
    t0 = time.time() #Time Vprint
    brushSize = 0.401
    Gcode = os.path.join(path, 'Gcode/Cube_LEXT.gcode')
    SqSize = [10.3, 14.3]
    pixelSize = [515, 715]
    bedSize = [22.3, 22.3]
    print("Printing virtual print...")
    ROIMatrices ,nofLayers = TL.Vprinter(brushSize, Gcode, SqSize, pixelSize, bedSize)
    for i in range(nofLayers+1):
        img=ROIMatrices[i]
        cv.imwrite(os.path.join(path,'Vprint/layer%i.jpg' %i),img)
    t1 = time.time()
    timeVprint = t1-t0
else:
    timeVprint = 0
    nofLayers = np.size(imgs)


#Printmasks from virtual print
masks = glob.glob(os.path.join(path, 'Vprint/*jpg'))
masks = sorted(masks, key=TL.filenum)


#Inisialize calibration parameters
gridsize = (7,9)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

"""
Camara preperation
"""
t0 = time.time() #Time Calibration
#Calibration
mtx, dist, newcameramtx, roi = TL.CalibratePiCam(calImgs, gridsize, objp)
#Compute perspective transform Ultimaker 2+
M = TL.GetPerspectiveU2(perspectiveSquares, mtx, dist, newcameramtx, roi)
t1 = time.time()
timeCal = t1-t0


"""
Main loop for digital twin:
"""
print('Computing twin for %i layers:' % np.size(imgs) )
#Loop over all printed layers, to create twin
t0 = time.time() #Time Digital twin
timesPerLayer = []

i = 0
for fname in sorted(imgs, key=TL.filenum ):
    t00 = time.time() #Time Digital twin per layer
    
    img = cv.imread(fname)
    img = cv.resize(img, (2028, 1520), interpolation=cv.INTER_AREA)    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    #Warp perspective    
    dst = cv.warpPerspective(dst,M, (515, 715))
    dst = cv.resize(dst, (515, 715), interpolation=cv.INTER_AREA)
    
    #Save warped
    if i == layerCheck:
        cv.imwrite(os.path.join(path , 'FigureStuff/Temp0/Warped'+str(layerCheck)+'.jpg'), dst)
        cv.waitKey(0)
    
    #Segmentation
    printmask = cv.imread(masks[i], cv.IMREAD_UNCHANGED)
    printmask = cv.resize(printmask, (515, 715), interpolation=cv.INTER_AREA)
    
    #Set printmask to none if running on LPBF printer
    #printmask = None
    
    ###### Comment if running time code ######
    if i == layerCheck:        
        cv.namedWindow('Check for correct distortion', cv.WINDOW_NORMAL)
        cv.imshow('Check for correct distortion', dst)
        cv.waitKey()
    
    dst = TL.SegmentPrint(dst, printmask)
    
    ###### Comment if running time code ######
    if i == layerCheck:
        cv.namedWindow('Check for correct segmentation', cv.WINDOW_NORMAL)
        cv.imshow('Check for correct segmentation', dst)
        cv.waitKey()

    #Save output to folder
    cv.imwrite(os.path.join(path , 'Twin/twin%i.jpg' % i), dst)
    cv.waitKey(0)
    
    #Print saved images (for every tenth frame)
    if i%10 == 0 and i != 0:
        print('Saved layers: %i-%i' %(i-9,i))
    i = i+1
    
    t01 = time.time()
    timesPerLayer.append(t01-t00)

t1 = time.time()
timeTwin = t1-t0
timesPerLayer = np.array(timesPerLayer)
timeMean = np.mean(timesPerLayer)
timeVar = np.var(timesPerLayer)


#Print running time data
print('\n', end='')
print('Running times for: ' + os.path.basename(imgs[0]))
print('\n', end='')
print('Generated complete virtual print in: %f s' %(timeVprint))
print('Generated virtual print at %f seconds per layer' %((timeVprint)/(nofLayers+1)))
print('\n', end='')
print('Calibration took: %f s' %(timeCal))
print('\n', end='')
print('Generated complete digital twin in: %f s' %(timeTwin))
print('Generated digital twin at: %f seconds per layer' %((timeTwin)/(nofLayers+1)))
print('\n', end='')
print('Twin time per layer mean:      %f' %(timeMean))
print('Twin time per layer variance:  %f' %(timeVar))
print('Number of layers/Sample size:  %i' %(nofLayers+1))
print('\n', end='')
print('-----------------------------------------------------------')

