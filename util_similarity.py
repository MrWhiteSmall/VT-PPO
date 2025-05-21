
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cv2
import os
import hashlib
import math

'''
    直方图相似度
    相关性比较 cv2.HISTCMP_CORREL：值越大，相似度越高
    相交性比较 cv2.HISTCMP_INTERSECT：值越大，相似度越高
    卡方比较 cv2.HISTCMP_CHISQR ：值越小，相似度越高
    巴氏距离比较 cv2.HISTCMP_BHATTACHARYYA ：值越小，相似度越高
'''

def normalize(data):
    return data / np.sum(data)
def hist_similarity(img1, img2, hist_size=256,compare=cv2.HISTCMP_CORREL):
    imghistb1 = cv2.calcHist([img1], [0], None, [hist_size], [0, 256])
    imghistg1 = cv2.calcHist([img1], [1], None, [hist_size], [0, 256])
    imghistr1 = cv2.calcHist([img1], [2], None, [hist_size], [0, 256])
 
    imghistb2 = cv2.calcHist([img2], [0], None, [hist_size], [0, 256])
    imghistg2 = cv2.calcHist([img2], [1], None, [hist_size], [0, 256])
    imghistr2 = cv2.calcHist([img2], [2], None, [hist_size], [0, 256])
 
    distanceb = cv2.compareHist(normalize(imghistb1), normalize(imghistb2), compare)
    distanceg = cv2.compareHist(normalize(imghistg1), normalize(imghistg2), compare)
    distancer = cv2.compareHist(normalize(imghistr1), normalize(imghistr2), compare)
    meandistance = np.mean([distanceb, distanceg, distancer])
    return meandistance

def PSNR(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def SSIM(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 计算两个灰度图像之间的结构相似度
    score, diff = structural_similarity(gray1, gray2, win_size=None, full=True)
    # diff = (diff * 255).astype("uint8")
    # print("SSIM:{}".format(score))
    return score, diff

def MSE(img1,img2):
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    return mse