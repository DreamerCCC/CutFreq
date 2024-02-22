import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors, cm, ticker
from torch.utils.data import DataLoader
from dataloaders.data_rgb import get_training_data, get_validation_data
from scipy import signal
from PIL import Image
import glob
import torch
from torchvision import transforms
from albumentations import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from load_data import LoadData, LoadVisualData

SAVE_PATH = 'frequency-outputs/denoise-nips/'


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

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

def test_for_single_v2(i, im1, im2): 
    print('Processing the test sample ' + str(i) + ': ...')
    im1 = transform(torch.squeeze(im1))
    im2 = transform(torch.squeeze(im2))
    frequency_im1, average_psd1 = frequency1DDomain(torch.squeeze(im1))
    frequency_im2, average_psd2 = frequency1DDomain(torch.squeeze(im2))
    frequency_minus = abs(frequency_im1-frequency_im2)
    '''
    fig = plt.figure()
    ax_1 = plt.subplot(131)
    ax_1.imshow(frequency_im1)
    plt.title('(a) Corruption')
    ax_2 = plt.subplot(132)
    ax_2.imshow(frequency_im2)
    plt.title('(b) Real')
    ax_3 = plt.subplot(133)
    ax3 = ax_3.imshow(frequency_minus)
    plt.title('(c) Difference')
    
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(ax3, cax=cax)
    
    fig.suptitle('Frequency Comparison with Derain Samples: ' + str(i), y=0.8)
    plt.tight_layout()
    #plt.show()
    plt.savefig("/home/chen/detection/frequency-dis/corruption-frequency/rain13k/Test1200/comparison_test_" + str(i) + ".png")
    '''
    print('Processing samples successfully!')
    return frequency_minus, frequency_im2

def pattern():

    val_dataset = get_training_data("./datasets/NH-HAZE/", {'patch_size': 128})
    data_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    SUM_temp = np.zeros(shape=(1200,1600))
    SUM_minus = np.zeros(shape=(1200,1600))
    SUM_input = np.zeros(shape=(1200,1600))
    fig = plt.figure()
    
    for i, data in enumerate((data_loader), 0):
        print(len(data_loader))
        target = data[0]
        input_ = data[1]
        frequency_minus, frequency_input = test_for_single_v2(i, target, input_)
        print(frequency_minus.shape)
        if frequency_minus.shape[0] > frequency_minus.shape[1]:
            frequency_minus = frequency_minus.T
            frequency_input = frequency_input.T
        
        SUM_minus = SUM_minus + frequency_minus
        SUM_input = SUM_input + frequency_input
        SUM_temp = SUM_temp + frequency_minus

        if (i+1)%11 == 0:
            SUM_temp = SUM_temp / int(11)
            plt.subplot(2, 3, int((i+1)/11))
            im = plt.imshow(SUM_temp)
            plt.title('(' + str(int((i+1)/11)) + ') part')
            plt.xticks(())
            plt.yticks(())
            plt.imsave("./corruption-frequency/nhhaze/comparison_trainpart_" + str(int((i+1)/11)) + ".png", SUM_temp)
            SUM_temp = np.zeros(shape=(1200,1600))
            print("~~~Subfigure processing~~~")
    
    SUM_minus = SUM_minus / len(data_loader)
    SUM_input = SUM_input / len(data_loader)
    
    plt.subplot(2, 3, 6)
    im = plt.imshow(SUM_minus)
    plt.title('(6) ALL')
    plt.xticks(())
    plt.yticks(())
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.92, 0.12, 0.015, .78])
    cb = fig.colorbar(im, cax = position)
    plt.savefig("./corruption-frequency/nhhaze/comparison_test_subsets" + ".png")
    plt.imsave("./corruption-frequency/nhhaze/comparison_test_minus" + ".png", SUM_minus)
    plt.imsave("./corruption-frequency/nhhaze/comparison_test_input" + ".png", SUM_input)
    

if __name__ == '__main__':
    pattern()
        