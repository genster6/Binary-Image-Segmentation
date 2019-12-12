
import matplotlib.pylab as plt
from ipywidgets import interact

from see import Segmentors

def showtwo(img, img2):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig.add_subplot(1,2,2)
    ax.imshow(img2)
    
def showthree(im, img, img2):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(im)
    ax = fig.add_subplot(1,3,2)
    ax.imshow(img)
    ax = fig.add_subplot(1,3,3)
    ax.imshow(img2)
    
def showSegment(im, mask):
    im1 = im.copy()
    im2 = im.copy()
    im1[mask>0,:] = 0
    im2[mask==0,:] = 0
    showtwo(im1,im2)


def paramsearch(im,mask, dx = 110):
    showtwo(im,mask)
    return 'test'


def makewidget(im,mask, params=110):
    return interact(paramsearch, im=im, mask=mask, dx=(-400,400)); 