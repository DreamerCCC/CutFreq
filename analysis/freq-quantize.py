import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors, cm, ticker
from scipy import signal
from PIL import Image
import glob
from albumentations import *

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
    ps = np.absolute((fshift))**2
    average_psd = 10 * np.log10(ps)
    return fimg, average_psd
    
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

def test_for_hist():
    img1=np.array(Image.open(SAVE_PATH + '0005-0007_restormer_freq.jpg').convert('L'))
    img2=np.array(Image.open(SAVE_PATH + '0005-0007_ref_freq.jpg').convert('L'))
    bins = np.linspace(50, 175, 50)
    plt.figure("Hist")
    arr1=img1.flatten()
    arr2=img2.flatten()
    plt.hist(arr1, bins, alpha=0.5, label='Ours')
    plt.hist(arr2, bins, alpha=0.5, label='Ground Truth')
    plt.legend(loc='upper left')
    plt.xlabel('Value') 
    plt.ylabel('Amount') 
    plt.title(r'Histogram of Frequency Map')
    plt.savefig(SAVE_PATH1 + 'rebuttal_ours.png')
    #plt.show()
    print('Printing histogram figure successfully!')

if __name__ == '__main__':
    test_for_hist()