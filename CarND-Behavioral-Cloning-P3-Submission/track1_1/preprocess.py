import cv2
import numpy as np

def img_rgb2Lab(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    return labImg

def img_rgb2HSV(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return labImg

def img_rgb2HLS(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return labImg

_resize_factor = 4

def img_resize(img):
    img = cv2.resize(img, (int(img.shape[1]/_resize_factor), int(img.shape[0]/_resize_factor)))
    return(img)

def img_normalize(img):
    return(img/255.0)

#percent of image
_top_crop = 0.35
_bottom_crop = 0.15
#_bottom_crop = 0.0

def img_crop(img):
    height = int((1.0-_top_crop-_bottom_crop)*img.shape[0])
    origin_y = int(_top_crop*img.shape[0])
    return(img[origin_y:origin_y+height,:])

def preprocess(img):
    img = img_resize(img)
    img = img_crop(img)
    img = img_rgb2HLS(img)
    img = img_normalize(img)
    img=img-0.5
    if len(img.shape)==2:
        img = img[:,:,None]
    return(img)

if __name__=='__main__':
    img = cv2.imread('test.jpg')
    print('input image shape: ',img.shape)

    img = img_resize(img)
    img = img_crop(img)
    img = img_rgb2HLS(img)

    print('output image shape: ',img.shape)
    cv2.imshow('test image', img)
    cv2.waitKey(0)
