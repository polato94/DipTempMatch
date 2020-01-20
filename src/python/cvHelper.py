# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 



def bilinear(img, pix):
    # calculate neighborhood−> integer
    x1 = math.floor(pix[1])
    y1 = math.floor(pix[0])
    x2 = math.floor(pix[1] + 1)
    y2 = math.floor(pix[0] + 1)
    # check if pix is out of range
    if (x1 < 0 or x1 > img.shape[1] - 1) or (x2 < 0 or x2 > img.shape[1] - 1) \
            or (y1 < 0 or y1 > img.shape[0] - 1) or (y2 < 0 or y2 > img.shape[0] - 1):
        return 0
    # do interpolation−> 1. get pixel values
    q11 = img[y1, x1]
    q12 = img[y2, x1]
    q21 = img[y1, x2]
    q22 = img[y2, x2]
    # interpolation in x−direction
    f_r1 = q11 * (x2 - pix[1]) / (x2 - x1) + q21 * (pix[1] - x1) / (x2 - x1)
    f_r2 = q12 * (x2 - pix[1]) / (x2 - x1) + q22 * (pix[1] - x1) / (x2 - x1)
    # interpolation in y−direction
    f_p = f_r1 * (y2 - pix[0]) / (y2 - y1) + f_r2 * (pix[0] - y1) / (y2 - y1)
    return f_p


def bilinear_scale(img_src, img_dst, s):
    T = np.array([[s, 0], [0, s]])  # e.g. s = 0.5
    for y in range(img_src.shape[0] - 1):
        for x in range(img_src.shape[1] - 1):
            img_dst[y, x] = bilinear(img_src, T.dot([y, x]))

def plothist(img,title):
# plot the default histogram of an image
    # calculate histogram
    plt.figure()
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # plot the histogram
    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.grid()
    #plt.show(block=False)

def plothist_normalized(img):
# plot the normalized histogram using matplotlib
    plt.figure()
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist.max() / img.size, cv2.NORM_MINMAX)
    hist = hist / img.size
    plt.plot(hist)

def plothist_channels(img):
# plot the rgb colour channels using matplotlib
    plt.figure()
    chans = cv2.split(img)  # split color channels
    color = ('b', 'g', 'r') # ID for colors
    for (chans, color) in zip(chans, color):
        hist = cv2.calcHist([chans], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.xlim([0, 256])
    plt.show(block=False)

def gray_spread(img_src):
# image histogram spreading. Only works when there are no histogram peaks near the extremal edges
    gmin = img_src.min()
    gmax = img_src.max()
    wmax = 255
    wmin = 0
    g = img_src
    f = (wmax-wmin)/(gmax-gmin)
    gp = (g - gmin).dot(f) + wmin
    gp = gp.astype(np.uint8)
    return gp

def imadjust(x,a,b,c,d,gamma=1):
# Similar to imadjust in MATLAB: Converts an image range from [a,b] to [c,d]. Used for overexposed/underexposed images
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def imgplot(img,title):
# plot an image using matplotlib in grayscale
    plt.figure()
    plt.imshow(img, vmax=255, vmin=0, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show(block=False)

################################
#adopted from https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python#5849861
import time
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
#####################################

def image(gray_img, title=None):
    # for double images
    vvmin=0 
    vvmax=1 
  
    # for uint8 images
    if hasattr(gray_img,'dtype') and gray_img.dtype == np.uint8:  
        vvmin=0
        vvmax=255
    plt.figure()  
    if title:
        plt.title(title)
    plt.imshow(gray_img, cmap='gray', vmin=vvmin, vmax=vvmax)
    plt.colorbar()
    plt.show()

def imagesc(img, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img,  vmin=np.min(img), vmax=np.max(img), cmap='hot')
    plt.colorbar() 
    plt.show()  # display it
    
def imhist(gray_img):
    #hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    plt.figure()
    plt.hist(gray_img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale image')
    plt.show()

def imHough(h, theta, rho, aspectRatio=1/5):
    # show accumlator cells

    from matplotlib import cm
    plt.figure()
    plt.imshow(np.log(1 + h),
                extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
                cmap=cm.gray, aspect=aspectRatio)
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')
    plt.title('Hough accumlator cells')

def displayInliers(img, B):
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    B[:,[0, 1]] = B[:,[1, 0]]
    img3=cv2.polylines(img_c, [B], True, (0,255,255), 3)
    imagesc(img3)

def plot3d(img, width=100):
    print("plot3d: use '%matplotlib auto' for interactive mode")
    ratio = img.shape[0] /img.shape[1]
    dim = (width, int(ratio*width))
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
 
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:resized.shape[0], 0:resized.shape[1]]
    
    # create the figure
    fig = plt.figure()
   
    ax = Axes3D(fig)
    #ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, resized ,rstride=1, cstride=1, cmap=plt.cm.seismic, linewidth=0)
    
    # show it
    plt.show()

def showRectangle(img, point, height, width):
    displayimage = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)       
    cv2.rectangle(displayimage, point, (point[0] + width, 
                                point[1] + height), (0,255,255), 2) 
    image(displayimage)    
    
def closeFigures():
    plt.close("all")
    
def drawAxis(img, rvec, tvec, K, dist, length=3, linewidth=20, flipZ=True):   
    xlength=length
    ylength=length
    zlength=length
    if flipZ:
        zlength=-1*zlength
    points = np.float32([[0, 0, 0], 
                         [xlength, 0, 0], 
                         [0, ylength, 0], 
                         [0, 0, zlength]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
   
    img = cv2.line(img, tuple(axisPoints[0].ravel()), tuple(axisPoints[1].ravel()), (255,0,0), linewidth)
    img = cv2.line(img, tuple(axisPoints[0].ravel()), tuple(axisPoints[2].ravel()), (0,255,0), linewidth)
    img = cv2.line(img, tuple(axisPoints[0].ravel()), tuple(axisPoints[3].ravel()), (0,0,255), linewidth)
    return img
    
    
    
