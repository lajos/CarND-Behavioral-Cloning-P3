import cv2
import numpy as np
import skimage
import skimage.transform

def img_rgb2Lab(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    return labImg

def img_rgb2HSV(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return labImg

def img_rgb2HLS(img):
    labImg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return labImg

_resize_factor = 2

def img_resize(img):
    img = cv2.resize(img, (int(img.shape[1]/_resize_factor), int(img.shape[0]/_resize_factor)))
    return(img)

def img_normalize(img):
    return(img/255.0)

#percent of image
#_top_crop = 0.35
_top_crop = 0.45
_bottom_crop = 0.15
#_bottom_crop = 0.0

def img_crop(img):
    height = int((1.0-_top_crop-_bottom_crop)*img.shape[0])
    origin_y = int(_top_crop*img.shape[0])
    return(img[origin_y:origin_y+height,:])

def img_sharpen(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def img_unsharp_mask(img):
    kernel = np.array([[1,4,6,4,1],
        [4,16,24,16,4],
        [6,24,-476,24,6],
        [4,16,24,16,4],
        [1,4,6,4,1]])/-256.0
    return cv2.filter2D(img, -1, kernel)

def img_pad(img, wt=200, ht=66):
    w=img.shape[1]
    h=img.shape[0]
    wp = int((wt-w)/2)
    hp = int((ht-h)/2)
    padded = np.zeros((ht,wt,3),dtype=np.uint8)
    padded[hp:h+hp, wp:w+wp,:]=img
    return(padded)

def img_untilt(img, distance=5):
    w=img.shape[1]
    h=img.shape[0]
    m1 = np.array(((0,0),(0,h),(w,h),(w,0)))
    m2 = np.array(((-distance,0),(0,h),(w,h),(w+distance,0)))
    projection = skimage.transform.ProjectiveTransform()
    projection.estimate(m2,m1)
    img = skimage.transform.warp(img, projection)
    return(img)

def img_untilt2(img, distance=5):
    w=img.shape[1]
    h=img.shape[0]
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([[-distance,0],[w+distance,0],[0,h],[w,h]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    return cv2.warpPerspective(img,M,(w,h), flags=cv2.INTER_NEAREST)

def preprocess(img):
    # img = img_resize(img)
    # img = img_crop(img)
    # img = img_rgb2HLS(img)
    # img = img_normalize(img)
    # img_unsharp_mask(img)

    # img = img_resize(img)
    # img = img_crop(img)
    # b,g,r = cv2.split(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    # #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    # b = clahe.apply(b)
    # g = clahe.apply(g)
    # r = clahe.apply(r)
    # img=cv2.merge((b,g,r))
    # img = img_rgb2HLS(img)
    # img = img_normalize(img)
    # img_unsharp_mask(img)

    img = img_resize(img)
    img = img_crop(img)
    img = img_untilt2(img,distance=80)
    img = img_rgb2HLS(img)
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    g = clahe.apply(g)
    img=cv2.merge((b,g,r))
    img = img_normalize(img)
    img_unsharp_mask(img)

    img=img-0.5

    if len(img.shape)==2:
        img = img[:,:,None]
    return(img)

if __name__=='__main__':
    img = cv2.imread('test2.jpg')
    print('input image shape: ',img.shape)


#     img = img_resize(img)
#     img = img_crop(img)
#     img = img_untilt2(img,distance=50)
#     img = img_rgb2HLS(img)
#     b,g,r = cv2.split(img)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
# #    b = clahe.apply(b)
#     g = clahe.apply(g)
# #    r = clahe.apply(r)
#     img=cv2.merge((g,g,g))
# #    img =img_pad(img)

    img = np.fliplr(img)

 #   img_unsharp_mask(img)

    print('output image shape: ',img.shape)
    cv2.imshow('test image', img)
    cv2.waitKey(0)
