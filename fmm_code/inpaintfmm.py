import numpy as np
import cv2
import sys
from inpaint_fmm import fast_marching_method




boxes = []
drawing=False

def inpaint(input_image, inpaint_mask, radius=5):    
    
    h, w = input_image.shape
    painted = np.zeros((h + 2, w + 2), np.float)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    inner = (slice(1, -1), slice(1, -1))
    painted[inner] = input_image
    mask[inner] = inpaint_mask

    fast_marching_method(painted, mask, radius=radius)

    return painted[inner]

def on_mouse(event, x, y, flags, params):
    global ix,iy,drawing
    
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            
            cv2.line(image,(ix,iy),(x,y),(1,1,1),5)
            cv2.line(mask,(ix,iy),(x,y),(1),5)
            ix=x
            iy=y
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        
    cv2.imshow('masked image',image)

    return x,



if len(sys.argv) == 2:
    
    filename = sys.argv[1]
    image = cv2.imread(filename)
else:
    print "name of image to be inpainted is not given"
    print "use: 'python inpaintfmm.py [image name]'"




rows,cols,_ = image.shape
mask = np.zeros((rows,cols), np.uint8)
cv2.imshow('input image', image)
cv2.imshow('masked image', image)
while(1):
    
    
    key = cv2.waitKey(1)
    if key == 27: #esccape 
        break
    if key == ord('o'):
        mask2 = mask
        image[mask == 1] = 0
        b,g,r = cv2.split(image)
        b_i = inpaint(b, mask)
        g_i = inpaint(g, mask)
        r_i = inpaint(r, mask)
        
        painted = cv2.merge((b_i,g_i,r_i))
#        painted = inpaint(image, mask)
        painted =cv2.convertScaleAbs(painted)
        cv2.imshow('inpainted_image',painted)


    if key == ord('m'):
        cv2.setMouseCallback('masked image', on_mouse, 0)
        
    elif key == ord('h'):
        print "program description:"
        print "This program performs image inpainting using FMM"
        print "python inpaintfmm.py [image name]' to run program with image input"
        print "press 'm' to start detecting mouse actions. After this, click anywhere on 'masked image' window and start drawing mask on it" 
        print "Press 'o' to display ouput"
        
        
cv2.destroyAllWindows()

