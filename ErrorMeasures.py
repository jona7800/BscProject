"""
Main script for computing error of 3d print.
Using digital twin and 'perfect twin' in directory (image folder).

OBS. Set directories appropriately before running. 
"""
# Necessary packages:
import numpy as np
import cv2 as cv
import os as os
import glob
import tifffile as tff
import TwinLib as TL
from matplotlib import pyplot as plt


path1 = 'C:/Users/Jonathan/Documents/6.Semester/BachelorProject'
path2 = 'C:/Users/Jonathan/Documents/6.Semester/BachelorProject/Code/Vprint'

perfectTwinExists = True
LocalError = True

Twin = glob.glob(os.path.join(path1, 'Code/Twin/*jpg'))
Twin = sorted(Twin, key=TL.filenum)

"""
Create perfect twin from Gcode, if it doesn't exist already
"""
if perfectTwinExists == False:
    brushSize = 0.4
    Gcode = os.path.join(path1, 'Gcode/UM2_Tie1.gcode')
    SqSize = [10.3, 14.3]
    pixelSize = [515, 715]
    bedSize = [22.3, 22.3]
    print("Saving 'perfect' twin...")
    ROIMatrices ,nofLayers = TL.Vprinter(brushSize, Gcode, SqSize, pixelSize, bedSize)
    for i in range(nofLayers+1):
        img=ROIMatrices[i]
        cv.imwrite(os.path.join(path2,'layer%i.jpg' %i),img)
    
Vprint = glob.glob(os.path.join(path2, '*jpg'))
Vprint = sorted(Vprint, key=TL.filenum)

sumerror=0
maxerror=0
sumgcode=0
errorarr=[]
tiff_list = None



"""
Compute both global and local error for each layer:
"""
for i in range(len(Twin)):
    vprint = cv.imread(Vprint[i], cv.IMREAD_UNCHANGED)
    twin = cv.imread(Twin[i], cv.IMREAD_UNCHANGED)
    
    diff=cv.absdiff(twin, vprint)
    
    #Global error
    error=((np.sum(diff)/255)/(np.sum(vprint)/255))*100
    errorarr.append(error)
    
    if maxerror <= error:
        maxerror = error
        idx=i
        
    sumerror = (sumerror + np.sum(diff)/255)
    sumgcode = sumgcode + (np.sum(vprint)/255)
    
    if LocalError:        
        tiff = cv.imread(Vprint[i])
        
        #Color local areas red
        tiff[:,:,0] = tiff[:,:,0]
        tiff[:,:,1] = tiff[:,:,1]-diff
        tiff[:,:,2] = tiff[:,:,2]-diff

        #Append local error to tiff list
        tiff = tiff[np.newaxis, ::]
        if i == 0:
            tiff_list = tiff
        else:
            tiff_list = np.append(tiff_list, tiff, axis=0)

    if i%10 == 0 and i != 0:
        print('Computed errors for layers: %i-%i' %(i-9,i))


avgerror=sumerror/sumgcode*100
print('Average error: %.2f' % avgerror, '%')
print('MAX error is: %.2f' % maxerror,'%' + ', at layer: %i' % idx)


#PLOT
layer=range(i+1)
plt.plot(layer, errorarr)
plt.xlabel('Layer')
plt.ylabel('Relative error')
plt.title('Layer by layer error')
plt.show()


if LocalError:
    #save local errors as tiff
    tff.imsave(os.path.join(path1,'Code/LocalErrorTwin.tif'), tiff_list)








