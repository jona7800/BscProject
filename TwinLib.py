"""
Functions for 3d printing error estimation. 
To be used with either:
    - Ultimaker 2+ and Raspberry Pi HQ camera (Wide angle lens).
    - LPBF-printer, made by Magnus Bolt (AM Group at DTU)
"""

# Necessary packages:
import cv2 as cv
import os
import re
import numpy as np
import math


"""
Finds filenumber for sorting purposes.
"""
def filenum(filename):
    #Finds framenumber for sorting
    return int(re.findall(r'\d+',filename)[-1])


"""
Function for calibrating camera, undistorting.

images:         All file paths for calibration images
gridsize:       Gridsize of calibration checkerboard
objp:           Real world objectpoints of calibration checkerboard

Returning:      mtx, dist, newcameramtx, roi

mtx:            input camera matrix
dist:           distortion coefficients
newcameramtx:   refined camera matrix
roi:            region of interrest 
"""
def CalibratePiCam(images, gridsize, objp):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    print('Calibrating camera...')
    
    for fname in sorted(images, key=filenum):
        img = cv.imread(fname)
        img = cv.resize(img, (2028, 1520), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, gridsize, None)
        # If found, add object points, image points
        if ret == True:
            print('Found grid on: ',os.path.basename(fname))
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = \
        cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    print('\n')
    
    return mtx, dist, newcameramtx, roi


"""
Function to find perspective transform for Ultimaker 2+.
OBS. Uses four points for perspective transform, if three are used used, use GetPerspectiveLPBF()

perspectiveSquares:        File path for perspective image with printed squares

Returning:                 M (Perspective transformation matrix)
"""
def GetPerspectiveU2(perspectiveSquares, mtx, dist, newcameramtx, roi):
    squares = []
    
    #Define perspective using printed squares
    PerspectiveImg = cv.imread(perspectiveSquares)
    PerspectiveImg = cv.resize(PerspectiveImg, 
                           (2028, 1520), interpolation=cv.INTER_AREA)
    PerspectiveImg = cv.cvtColor(PerspectiveImg, cv.COLOR_BGR2GRAY)
    
    # Undistort, to find actual perspective
    PerspectiveImg = cv.undistort(PerspectiveImg, mtx, dist, None, newcameramtx)
    # crop the image with roi from calibration
    x, y, w, h = roi
    PerspectiveImg = PerspectiveImg[y:y+h, x:x+w]
   
    # Roi (new) allows to search for squares in only relevant area.
    x, y, w, h = 250, 120, 1300, 1100
    
    # Threshold to find squares
    thr = 68
    white = 255*np.ones_like(PerspectiveImg)
    ret, mask1 = cv.threshold(PerspectiveImg,thr,255,cv.THRESH_BINARY)

    roi = mask1[y:y + h, x:x + w]
    white[y:y+h, x:x+w] = roi
    mask1 = white
    mask1 = cv.bitwise_not(mask1)
    
    # Find contours and filter using threshold mask
    cnts, hierarchy = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # minimum and maximum accepted area of squares
    min_area = 50
    max_area = 4000
    
    #cv.drawContours(PerspectiveImg, cnts, -1, (0,255,0), 3)
    cv.rectangle(PerspectiveImg,(int(x), int(y)),(int(x+w), int(y+h)),(0,255,0),2)
    
    square_number = 0
    for c in cnts:
        area = cv.contourArea(c)
        if area > min_area and area < max_area:
            
            # Save square
            (x,y),radius = cv.minEnclosingCircle(c)
            squares.append((int(x), int(y)))
            
            #Draw circle around square candidate
            center = (int(x),int(y))
            radius = int(radius)
            cv.circle(PerspectiveImg,center,radius,(0,255,0),5)
        
            print('Found square %i in perspective image'  % square_number)
        
            square_number += 1
            
    
    # If exactly four squares candidates have not been found, return typeError
    if  square_number != 4:
        # Show square candidates
        cv.namedWindow('Found wrong number of squares, check ROI', cv.WINDOW_NORMAL)
        cv.imshow('Found wrong number of squares, check ROI', PerspectiveImg)
        cv.waitKey()
        raise TypeError('Failed to find all four squares in' + 
                        os.path.basename(perspectiveSquares) +
                        '. Capture new perspective image or redefine Roi for image or')
    
    squares = np.array(squares)
    
    #Change squares orientation accordingly
    squares[[0,2]] = squares[[2,0]]
    
    ###### Comment if running time code ######
    # Show square candidates
    cv.namedWindow('Check wether squares has been found correctly'
                   , cv.WINDOW_NORMAL)
    cv.imshow('Check wether squares has been found correctly', PerspectiveImg)
    cv.waitKey()
    
    pts1 = np.float32(squares)
    pts2 = np.float32([[0,0],[0,715],
                       [515,715],[515,0]])
    
    #Compute perspective transformation matrix
    M = cv.getPerspectiveTransform(pts1,pts2)
    
    return M

"""
Function to find perspective transform for LPBF printer.
OBS. Uses three points for perspective transform, if four are used used, use GetPerspectiveU2() 

perspectiveHoles:          File path for perspective image with clear view of buildplate holes

Returning:                 M (Perspective transformation matrix)
"""
def GetPerspectiveLPBF(perspectiveHoles, mtx, dist, newcameramtx, roi):
    holes = []
    
    #Define perspective using printed squares
    PerspectiveImg = cv.imread(perspectiveHoles)
    PerspectiveImg = cv.resize(PerspectiveImg, 
                           (2028, 1520), interpolation=cv.INTER_AREA)
    PerspectiveImg = cv.cvtColor(PerspectiveImg, cv.COLOR_BGR2GRAY)
    
    # Undistort, to find actual perspective
    PerspectiveImg = cv.undistort(PerspectiveImg, mtx, dist, None, newcameramtx)
    # crop the image with roi from calibration
    x, y, w, h = roi
    PerspectiveImg = PerspectiveImg[y:y+h, x:x+w]
   
    # Roi, allows to search for squares in only relevant area. (crude)
    x, y, w, h = 370, 240, 1378, 1280
    
    # Threshold to find squares
    thr = 68
    white = 255*np.ones_like(PerspectiveImg)
    ret, mask1 = cv.threshold(PerspectiveImg,thr,255,cv.THRESH_BINARY)
    
    roi = mask1[y:y + h, x:x + w]
    white[y:y+h, x:x+w] = roi
    mask1 = white
    mask1 = cv.bitwise_not(mask1)
    
    # Find contours and filter using threshold mask
    cnts, hierarchy = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # minimum and maximum accepted area of squares
    min_area = 50
    max_area = 3000
    
    cv.drawContours(PerspectiveImg, cnts, -1, (0,255,0), 3)
    
    hole_number = 0
    for c in cnts:
        area = cv.contourArea(c)
        if area > min_area and area < max_area:
            
            # Save square
            (x,y),radius = cv.minEnclosingCircle(c)
            holes.append((int(x), int(y)))
            
            center = (int(x),int(y))
            radius = int(radius)
            cv.circle(PerspectiveImg,center,radius,(0,255,0),5)
        
            print('Found hole %i in perspective image'  % hole_number)
        
            hole_number += 1
    
    # Return typeerror if less than four squares candidates have been found
    if  hole_number != 3:
        raise TypeError('Failed to find all four squares in' + 
                        os.path.basename(perspectiveHoles) +
                        '. Capture new perspective image or redefine Roi for image')
    
    
    holes = np.array(holes)
    holes[[0,2,1]] = holes[[1,0,2]]

    
    # Show square candidates
    cv.namedWindow('Check wether holes has been found correctly'
                   , cv.WINDOW_NORMAL)
    cv.imshow('Check wether holes has been found correctly', PerspectiveImg)
    cv.waitKey()

    pts1 = np.float32(holes)
    pts2 = np.float32([[0,0],[0,715],
                       [515,715],[515,0]])
    
    #Compute perspective transformation matrix
    M = cv.getPerspectiveTransform(pts1,pts2)
    
    return M
    

   
"""
Function for segmenting printed top layer out of image:

img:                Input image
prevmask0:          Unedited mask from previous layer, 
                    (set to None if at first layer)
prevmask:           Edited mask from previous layer, 
                    (set to None if at first layer)
                    
Returning:          Edited and unedited mask of input image
"""
def SegmentPrint(img, printmask):   
     # Threshold using Otsu's method
    ret, mask1 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    mask1 = cv.bitwise_not(mask1)
    
    # Apply printmask if given
    if printmask is not None:
        mask1 = cv.bitwise_and(mask1, printmask)
    
    return mask1
    
    
"""
Function for creating "perfect" digital twin.

brushsize:      Size of extrusion head (of actual printer)
filename:       Filepath of GCode
SqSize:         Height and width of calibration squares
pixelSize:      Pixel dimensions of output
bedSize:        Bedsize in x and y direction

Returning:      ROIMatrices, nofLayers
"""
def Vprinter(brushSize, filename, SqSize, pixelSize, bedSize):
    # Initialization
    layer=-1                                    #starting in layer -1
    xpos=0                                      #current x position
    ypos=0                                      #current y position
    realBedSizeX=int(bedSize[0]*100)            #conversion from cm to 0.1 mm
    realBedSizeY=int(bedSize[1]*100)            #conversion from cm to 0.1 mm
    nofLayers=-1                                #starting in layer -1 for sake of continuity

    bedSizeX=int((realBedSizeX/(SqSize[0]*100))*pixelSize[0])
    bedSizeY=int((realBedSizeY/(SqSize[1]*100))*pixelSize[1])
    countline=0                                 #Line count used for anomaly examination
    
    bedSizeRatio=bedSizeX/realBedSizeX          #Ratio for scaling

    brushSize=brushSize*bedSizeRatio*10
    
    #Ensure we terminate after the movement is complete
    M107count=0
    M104count=0
    
    #function used to avoid bankers rounding
    def round_school(x):
        i, f = divmod(x, 1)
        return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))
    
    
    #open file and save it as datafile
    with open(filename, "r") as f:
        datafile = f.readlines()
    
    """
    Main loop:
    """
    for line in datafile:
        if ';LAYER:' in line:
            nofLayers=nofLayers+1 
            nofLayers=int(nofLayers)

    #pre designation of location of 3d matrix
    ImgMatrices = []
    ROIMatrices = []
    #2d matrix for layer 0 of the digitalTwin
    ImgMatrix = np.zeros((bedSizeX,bedSizeY), dtype=np.uint8)
    ROIMatrix = np.zeros(pixelSize, dtype=np.uint8)
    ## LOCATING & RECORDING X & Y LOCATION & PRINTING OF IMAGES ##
    #runs through each line in the datafile and further down determines what
    #action shall or shant be taken.
    for line in datafile:
        countline=countline+1 #adds 1 to the count of lines used for anomaly exam
        
        ## New Layer Detection ##
        #checks wether we've reached a new layer & keeps count of layer & resets
        #the count of lines in layer (countline is primarily used for anomaly examination)
        if ';LAYER:' in line:
            layer=layer+1
            countline=0
            
            #appends the previous layer before creating a clean slate (right after) 
            if layer != 0:
                # by 90 degrees clockwise
                ImgMatrix = cv.rotate(ImgMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ROIMatrix = cv.rotate(ROIMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ImgMatrices.append(ImgMatrix)
                ROIMatrices.append(ROIMatrix)
            #Creating a clean 2d matrix for information storage for the layer
            ImgMatrix = np.zeros((bedSizeX,bedSizeY), dtype=np.uint8)
            ROIMatrix = np.zeros(pixelSize, dtype=np.uint8)
            
            #status update.
            if layer%10 == 0 and layer != 0:
                print('Generated Layers: %i-%i' %(layer-9,layer))
            
            
        #'M107' check for stopping program before encouring unnecesary G0 & G1
        #so to avoid bad X & Y positions which clutter image. added due to octopi. 
        if 'M107' in line:
            M107count=M107count+1
            if M107count==2:
                # by 90 degrees clockwise
                ImgMatrix = cv.rotate(ImgMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ROIMatrix = cv.rotate(ROIMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ImgMatrices.append(ImgMatrix)
                ROIMatrices.append(ROIMatrix)
                return ROIMatrices ,nofLayers 
        
        if 'M104' in line:
            M104count=M104count+1
            if M104count==2:
                # by 90 degrees clockwise
                ImgMatrix = cv.rotate(ImgMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ROIMatrix = cv.rotate(ROIMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
                ImgMatrices.append(ImgMatrix)
                ROIMatrices.append(ROIMatrix)
                return ROIMatrices ,nofLayers 
        
        ## X & Y POSITION CHECKS ##
        ### Checking if G0 is in line so new X and Y positions can be recorded ###
        if 'G0 ' in line:
            curLine=line                               #curLine: Current Line
            
            if 'X' in line:
                xStrLoc=curLine.find('X')       #xStrLoc: X String Location
                XYexsistance=1
                
                #No action if layer 0 is not yet reached.
                if layer<0:
                    pass
                elif 'Y' in line:
                    yStrLoc=curLine.find('Y')        #yStrLoc: y String Location
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:yStrLoc-1]    
                elif 'Z' in line:
                    zStrLoc=curLine.find('Z')        #zStrLoc: Z String Location
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:zStrLoc-1]
                else:                                
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:-1]
                    
            if 'Y' in line:
                yStrLoc=curLine.find('Y')        #yStrLoc: y String Location
                XYexsistance=1
                
                #No action if layer 0 is not yet reached.
                if layer<0:
                    pass     
                elif 'Z' in line:
                    zStrLoc=curLine.find('Z')        #zStrLoc: Z String Location
                    yprev=ypos
                    ypos=curLine[yStrLoc+1:zStrLoc-1]
                else:                          
                    yprev=ypos    
                    ypos=curLine[yStrLoc+1:-1]
        
        ### Checking if G1 is in line so new X and Y positions can be recorded ###
        if 'G1 ' in line:
            curLine=line                        #curLine = Current Line
  
            if 'X' in line:
                xStrLoc=curLine.find('X')       #xStrLoc = X String Location
                XYexsistance=1
                
                #No action if layer 0 is not yet reached.
                if layer<0:
                    pass
                elif 'Y' in line:
                    yStrLoc=curLine.find('Y')        #yStrLoc: y String Location
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:yStrLoc-1] 
                elif 'Z' in line:
                    zStrLoc=curLine.find('Z')        #zStrLoc: Z String Location
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:zStrLoc-1]
                elif 'E' in line:
                    eStrLoc=curLine.find('E')       #eStrLoc: E string Location
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:eStrLoc-1]
                else:                                
                    xprev=xpos
                    xpos=curLine[xStrLoc+1:-1]
            
 
            if 'Y' in line:
                yStrLoc=curLine.find('Y')        #yStrLoc: y String Location
                XYexsistance=1
                
                #No action if layer 0 is not yet reached.
                if layer<0:
                    pass
                elif 'Z' in line:
                    zStrLoc=curLine.find('Z')        #zStrLoc: Z String Location
                    yprev=ypos
                    ypos=curLine[yStrLoc+1:zStrLoc-1]
                elif 'E' in line:
                    eStrLoc=curLine.find('E')        #eStrLoc: E string Location
                    yprev=ypos
                    ypos=curLine[yStrLoc+1:eStrLoc-1]
                else:                                
                    yprev=ypos    
                    ypos=curLine[yStrLoc+1:-1]
                    
            ### PRINTING SEGMENT ###
            if 'E' in line:
            
                # check of wether X and Y are NOT in line.
                if not 'X' in line:
                    if not 'Y' in line:                
                        XYexsistance=0
                
                #No action if layer 0 is not yet reached.
                if layer<0:
                    pass
                        
                #check if 1 since that means X or Y (or both) exists.
                elif XYexsistance==1: 
                    xprev = float(xprev)
                    yprev = float(yprev)
                    xpos = float(xpos)
                    ypos = float(ypos)
                    x1 = min(xprev,xpos)
                    x2 = max(xprev,xpos)
                    
                    if x1==xprev:
                        y1=yprev
                        y2=ypos
                    else:
                        y1=ypos
                        y2=yprev
                        
                    #string to integer & rounding & multiplying by Ratio
                    x1=int(round_school(x1*10*bedSizeRatio))
                    x2=int(round_school(x2*10*bedSizeRatio))
                    y1=int(round_school(y1*10*bedSizeRatio))
                    y2=int(round_school(y2*10*bedSizeRatio))
            
                    ## generating points for infill, 3 cases""
                    #case 1: check if x-direction doesnt change
                    if (x1==x2):
                        k=abs(y2-y1)+1   
                        xpoints=np.linspace(x1,x2,k)
                        ypoints=np.linspace(y1,y2,k)
                        
                    #case 2: check if y-direction doesnt change
                    if (y1==y2):
                        k=abs(x2-x1)+1   
                        xpoints=np.linspace(x1,x2,k)
                        ypoints=np.linspace(y1,y2,k)
                        
                    #case 3: check if x changes and y changes
                    if (x1!=x2) and (y1!=y2):
                        a=(y2-y1)/(x2-x1)
                        b=y1-a*x1
                        linelength=math.sqrt((x2-x1)**2+(y2-y1)**2)
                        linelength=math.ceil(linelength)
                        xpoints=np.linspace(int(x1),int(x2),int(linelength))
                        k=len(xpoints)
                        ypoints=np.ones(k)+a*(xpoints)+b
                    
                    #conversion from floats to integers concerning use as indexes
                    for i in range(0,k-1):
                        ypoints[i]=round_school(ypoints[i])
                        xtemp=int(xpoints[i])
                        ytemp=int(ypoints[i])
                
                        #brushing around the center-dot using euclidian distance 
                        for j in range (math.floor(-brushSize/2),math.ceil(brushSize/2)):
                            for k in range (math.floor(-brushSize/2),math.ceil(brushSize/2)):
                                if (math.sqrt((j)**2+(k)**2)<=brushSize):
                                    ImgMatrix[xtemp+j][ytemp+k]=255
                    
                    Xlow=(math.ceil(bedSizeX/2)-math.ceil(pixelSize[0]/2))
                    Xhigh=(math.ceil(bedSizeX/2)+math.floor(pixelSize[0]/2))
                    Ylow=(math.ceil(bedSizeY/2)-math.ceil(pixelSize[1]/2))
                    Yhigh=(math.ceil(bedSizeY/2)+math.floor(pixelSize[1]/2))
                    ROIMatrix=ImgMatrix[Xlow:Xhigh,Ylow:Yhigh]
    
    #appending the last 2d matrix 
    # by 90 degrees clockwise
    ImgMatrix = cv.rotate(ImgMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)
    ROIMatrix = cv.rotate(ROIMatrix, cv.ROTATE_90_COUNTERCLOCKWISE)              
    ImgMatrices.append(ImgMatrix)
    ROIMatrices.append(ROIMatrix)

    #returning the 3d matrix and number of Layers 
    return ROIMatrices ,nofLayers       


