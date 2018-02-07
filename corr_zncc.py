#!/usr/bin/env python

from math import sqrt, log
import time
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Function : get_mean(image, x_start, y_start, window_size)
# -----------------------------------------------------------
# Description : compute mean of an image or sub-image
# -----------------------------------------------------------
# Input :
#      image : image in which the calculation takes place
#      x_start : x coordinates of upper left corner of image/sub-image
#      y_start : y coordinates of upper left corner of image/sub-image
#      window_size : half size of image/sub-image, the actual size is (2*window_size+1)
# -----------------------------------------------------------
# Output : mean of image/sub-image
# -----------------------------------------------------------

def get_mean(img, x, y, n):
    somme = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            somme += img[x+i][y+j]
    return float(somme)/((2*n+1)**2)

def paraboloid(x, y, a, b, c, d, e, f):
    return a*(x**2) + b*(y**2) + c*x*y + d*x + e*y * f


# -----------------------------------------------------------
# Function :sub_pixel_interp(interpolationMatrix, coefficientMatrix, similarity_peak, posX, posY)
# -----------------------------------------------------------
# Description : compute sub-pixel displacement residuals with
# paraboloid interpolation
# -----------------------------------------------------------
# Input :
#      interpolationMatrix : 
#      coefficientMatrix : 
#      similarity_peak : 
#      posX : 
#      posY : 
# -----------------------------------------------------------
# Output : X and Y sub-pixel displacements
# -----------------------------------------------------------
def sub_pixel_interp(A, B, sim_peak, posX, posY):
    
    X = np.matrix([[sim_peak[posX-1][posY-1]],
                   [sim_peak[posX-1][posY]],
                   [sim_peak[posX-1][posY+1]],
                   [sim_peak[posX][posY-1]],
                   [sim_peak[posX][posY]],
                   [sim_peak[posX][posY+1]],
                   [sim_peak[posX+1][posY-1]],
                   [sim_peak[posX+1][posY]],
                   [sim_peak[posX+1][posY+1]]])

    
    # system resolution to find constants : B = (At*A)^-1 * At * X
    B = (((A.transpose())*A).getI())*(A.transpose())*X
    
    # residuals to be added to X and Y displacements
    if (B[2]*B[2] - 4*B[0]*B[1] != 0 and B[0] < 0 and B[1] < 0) :
        XO = (2*B[1]*B[3] - B[2]*B[4])/(B[2]*B[2] - 4*B[0]*B[1]) # (2*b*d - c*e)/(c**2 - 4*a*b)
        YO = (2*B[0]*B[4] - B[2]*B[3])/(B[2]*B[2] - 4*B[0]*B[1]) # (2*a*e - c*d)/(c**2 - 4*a*b)
        if (XO < -1 or XO > 1 or YO < -1 or YO > 1) :
            XO = YO = 0
    else :
        XO = YO = 0

    return XO, YO
        
# -----------------------------------------------------------
# Function : correlation_ZNCC()
# -----------------------------------------------------------

def correlation_ZNCC(masterImage, slaveImage, corr_window, research_window, line_start, col_start, line_width, col_width, A, B) :

    im_step = 1
    research_step = 1
    savefig = 0

    compute_similarity = 'py'
    #compute_similarity ='c'

    coordinates = {'argentiere':{'line_start':1020, 'col_start':990, 'line_width':410, 'col_width':260},
               'mer_glace' :{'line_start':880, 'col_start':760, 'line_width':70, 'col_width':130},
               'test':{'line_start':880, 'col_start':760, 'line_width' : 20, 'col_width' : 30}}

    peak = np.ndarray((line_width, col_width))
    disX = np.ndarray((line_width, col_width))
    disY = np.ndarray((line_width, col_width))
    sim_peak = np.ndarray((2*research_window+1,2*research_window+1))
    displacementVector = np.ndarray((line_width, col_width))
    residualMapX = np.ndarray((line_width, col_width))
    residualMapY = np.ndarray((line_width, col_width))
    
    start_time = time.time()
    
    # faugeras (1993) approach
    # correlation window has dimensions (2n+1)(2m+1) if rectangular
    # in our case n = m (squared window)
    for x in range(line_start, line_start + line_width, im_step):
        print("progressing in line... %d" %x)
        for y in range(col_start, col_start + col_width, im_step):
        
            sim = 0

            # compute master correlation window average centered at (x,y)
            avgMaster = get_mean(masterImage, x, y, corr_window)
        
            # slide in research window
            for p in range(-research_window, research_window + 1, research_step):
                for q in range(-research_window, research_window + 1, research_step):

                    # correlation terms initialization 
                    c1, c2, c3 = (0,)*3

                    # compute average of slave correlation window centered at (x+p,y+q)
                    avgSlave = get_mean(slaveImage, x+p, y+q, corr_window)
                
                    # compute correlation between master & slave correlation windows
                    for i in range(-corr_window, corr_window + 1):
                        for j in range(-corr_window, corr_window + 1):
                                                
                            c1 += (masterImage[x+i][y+j] - avgMaster)*(slaveImage[x+p+i][y+q+j] - avgSlave)
                            c2 += (masterImage[x+i][y+j] - avgMaster)**2
                            c3 += (slaveImage[x+p+i][y+q+j] - avgSlave)**2

                    if (c2 != 0 and c3 != 0) :
                        sim_peak[p+research_window][q+research_window] = float(c1)/((c2*c3)**0.5)
                    else :
                        sim_peak[p+research_window][q+research_window] = None
                        print('Invalid value encountered in similarity peak')

                    # recherche du maximum de similarite par comparaisons successives (style code C)
                    if compute_similarity == 'c' :
                        if sim_peak[p+research_window][q+research_window] >= sim:
                            sim = sim_peak[p+research_window][q+research_window]
                        else:
                            sim = sim
                        peak[x-line_start][y-col_start] = sim

            # recherche du maximum de similarite avec fonction max() (style code python)
            if compute_similarity == 'py' :
                peak[x-line_start][y-col_start] = np.max(sim_peak)
            
                posX = np.argmax(sim_peak)/len(sim_peak) - len(sim_peak)/2
                posY = np.argmax(sim_peak)%len(sim_peak) - len(sim_peak)/2
            
                residualX = sub_pixel_interp(A, B, sim_peak, posX, posY)[0]
                residualY = sub_pixel_interp(A, B, sim_peak, posX, posY)[1]
                
                residualMapX[x-line_start][y-col_start] = residualX
                residualMapY[x-line_start][y-col_start] = residualY

                posX += residualX
                posY += residualY
            
                disX[x-line_start][y-col_start] = posX
                disY[x-line_start][y-col_start] = posY

                # displacement vector calculation
                displacementVector[x-line_start][y-col_start] = sqrt(posX**2 + posY**2)
            
            
    print("computation time: %s s" %round(time.time() - start_time))

    #
    # graphics
    #
    plt.figure(1)
    im = plt.imshow(peak)
    plt.colorbar(im)
    plt.title("image size=(%d*%d) ; n=%d ; m=%d" %(line_width, col_width, 2*corr_window+1, 2*research_window+1))

    if savefig :
        plt.savefig('/home/hipperta/nas_mastodons/hipperta/script/img/simpeak_xy%d%d_n%d_m%d.png' %(line_width, col_width, 2*corr_window+1, 2*research_window+1))

    plt.figure(2)
    im1 = plt.imshow(sim_peak)
    plt.colorbar(im1)

    plt.figure(3)
    im2 = plt.imshow(disX)
    plt.colorbar(im2)
    plt.title('Displacement x')

    plt.figure(4)
    im3 = plt.imshow(disY)
    plt.colorbar(im3)
    plt.title('Displacement y')

    plt.figure(5)
    im4 = plt.imshow(displacementVector)
    plt.colorbar(im4)
    plt.title('Displacement vector')
    
    plt.figure(6)
    im5 = plt.imshow(residualMapX)
    plt.colorbar(im5)
    plt.title('residual X')
    
    plt.figure(7)
    im6 = plt.imshow(residualMapY)
    plt.colorbar(im6)
    plt.title('residual y')

    plt.show()
