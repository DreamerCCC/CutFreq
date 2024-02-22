import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from albumentations import *
  
IMAGES_PATH5 = 'test-images/'
SAVE_PATH = 'frequency-outputs/'
paths5 = glob.glob(os.path.join(IMAGES_PATH5, '*.jpg'))
paths5.sort()

def imread(image):
    image = cv.imread(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image.astype(np.uint8)
    return np.array(image)
    
def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def showgray(image):
    plt.imshow(image, 'gray')
    plt.axis('off')
    plt.show()

#as cv.dft api
def fft(img):
    return np.fft.fft2(img)

# move low-frequency to the center
def fftshift(img):
    return np.fft.fftshift(fft(img))

# 2d inverse transformation
def ifft(img):
    return np.fft.ifft2(img)

# recover high and low frequency position
def ifftshift(img):
    return ifft(np.fft.ifftshift(img))

def generateDataWithDifferentFrequencies_3Channel(Images):
    fd = fftshift(Images)
    img = ifftshift(fd)
    
    return np.array(img)
    
def frequency1DDomain(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift))
    return fimg
    
def frequency3DDomain(img):
    freq = []
    tmp = np.zeros([img.shape[0], img.shape[1], 3])
    for j in range(3):
        f = np.fft.fftshift(np.fft.fft2(img[ :, :, j]))
        s = np.log(np.abs(f))
        tmp[:,:,j] = s
    freq.append(tmp)
    return np.array(freq)

def test_for_gray(): 
    for image in paths1:
        test = cv.imread(image, 0)
        print('Processing: ' + str(image) + '...')
        frequency = frequency1DDomain(test)
        plt.imsave(SAVE_PATH + str(image[-8:-4]) + '_' + 'canon_gray_image.jpg', frequency)
        #show(frequency)
    print('Processing gray images successfully!')
    
def test_for_color(): 
    for image in paths1:
        test = imread(image)
        print('Processing: ' + str(image) + '...')
        frequency = frequency3DDomain(test)
        frequency = np.squeeze(frequency, axis=(0,))
        plt.imsave(SAVE_PATH + str(image[-8:-4]) + '_' + 'deepspark_full_image.jpg', frequency.astype(np.uint8))
        #show(frequency.astype(np.uint8))
    print('Processing color images successfully!')
    
def test_for_gray_resize(): 
    for image in paths5:
        test = cv.imread(image, 0)
        print('Processing: ' + str(image) + '...')
        frequency = frequency1DDomain(test)
        plt.imsave(SAVE_PATH + 'freq_' + image.replace(IMAGES_PATH5, ''), frequency)
        #plt.imsave(SAVE_PATH + str(image[-13:-4]) + '_restormer_freq.jpg', frequency)
    print('Processing resized images successfully!')

def test_for_sample(): 
    for i in range(0, len(paths5), 2):
        test1 = cv.imread(paths5[i], 0)
        test2 = cv.imread(paths5[i+1], 0)
        print('Processing: ' + str(paths5[i]) + '...')
        frequency1 = frequency1DDomain(test1)
        frequency2 = frequency1DDomain(test2)
        frequency_minus = abs(frequency1-frequency2)
        
        fig = plt.figure()
        ax_1 = plt.subplot(131)
        ax_1.imshow(frequency1)
        plt.title('(a) Former')
        ax_2 = plt.subplot(132)
        ax_2.imshow(frequency2)
        plt.title('(b) Latter')
        ax_3 = plt.subplot(133)
        ax3 = ax_3.imshow(frequency_minus)
        plt.title('(c) Difference')
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(ax3, cax=cax)
        
        fig.suptitle('Comparison with Samples ID: ' + str(i+1), y=0.8)
        plt.tight_layout()
    
        plt.savefig(SAVE_PATH + 'sample_gap_' + paths5[i].replace(IMAGES_PATH5, ''))
        #plt.imsave(SAVE_PATH + str(image[-13:-4]) + '_freq.jpg', frequency)
    print('Processing resized images successfully!')
    
if __name__ == '__main__':
    test_for_sample()