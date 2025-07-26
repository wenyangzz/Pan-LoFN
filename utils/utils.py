#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2020-11-25 23:00:57
@Description: file content
'''
import os, math, torch,cv2
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F

from scipy import ndimage
from scipy import signal
import scipy.misc as misc
#from model.deepfuse import MEF_SSIM_Loss

def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(reduction='sum')
    elif loss_type == "L1":
        loss = nn.L1Loss(reduction='sum')
    elif loss_type == "MEF_SSIM":
        loss = MEF_SSIM_Loss()
    elif loss_type == "VGG22":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "VGG54":
        loss = VGG(loss_type[3:], rgb_range=255)
    else:
        raise ValueError
    return loss

def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/records.txt')) else 'w'
    log_file = open(get_path('./log/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/net.txt')) else 'w'
    log_file = open(get_path('./log/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')

def cpsnr(img1, img2, dynamic_range=255):
    """PSNR metric, img uint8 if 225; uint16 if 2047"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_)**2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))

def _ssim(img1, img2, dynamic_range=255):
    """SSIM for 2D (one-band) image, shape (H, W); uint8 if 225; uint16 if 2047"""
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  # kernel size 11
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def cssim(img1, img2, dynamic_range=255):
    """SSIM for 2D (H, W) or 3D (H, W, C) image; uint8 if 225; uint16 if 2047"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [_ssim(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(img1, img2, pixel_range=255, color_mode='rgb'):
    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2
    # img1 and img2 have range [0, pixel_range]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(pixel_range / math.sqrt(mse))

def ssim(img1, img2, pixel_range=255, color_mode='rgb'):
    C1 = (0.01 * pixel_range)**2
    C2 = (0.03 * pixel_range)**2

    # transfer color channel to y
    if color_mode == 'rgb':
        img1 = (img1 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
        img2 = (img2 * np.array([0.256789, 0.504129, 0.097906])).sum(axis=2) + 16 / 255 * pixel_range
    elif color_mode == 'yuv':
        img1 = img1[:, 0, :, :]
        img2 = img2[:, 0, :, :]
    elif color_mode == 'y':
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, pixel_range=255):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2, pixel_range)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, pixel_range))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), pixel_range)
    else:
        raise ValueError('Wrong input image dimensions.')

def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    # print(img_fake)
    return QNR_idx
def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """Spectral distortion
    img_fake, generated HRMS
    img_lm, LRMS"""
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # D_lambda
    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1/p)


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size ** 2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu2_sq
    #    print(mu1_mu2.shape)
    # print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0

    #    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))

    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) > 1e-8) * ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    #    print(np.mean(qindex_map))

    #    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    #    # sigma !=0 and mu == 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    #    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    #    # sigma != 0 and mu != 0
    #    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
    #    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
    #        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])

    return np.mean(qindex_map)

def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        #print(band1.shape)
        #print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)

def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_
def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)
def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2)
    return w
def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h
################################################################################
def upsample_bilinear(image, ratio):
    
    h,w,c = image.shape
    re_image = cv2.resize(image, (w*ratio, h*ratio), cv2.INTER_LINEAR)
    
    return re_image

def upsample_bicubic(image, ratio):
    
    h,w,c = image.shape
    re_image = cv2.resize(image, (w*ratio, h*ratio), cv2.INTER_CUBIC)
    
    return re_image

def upsample_interp23(image, ratio):

    image = np.transpose(image, (2, 0, 1))
    
    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    for z in range(1,np.int(np.log2(ratio))+1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image = I1LRU
        
    re_image=np.transpose(I1LRU, (1, 2, 0))
        
    return re_image

def upsample_mat_interp23(image, ratio=4):
    
    '''2 pixel shift compare with original matlab version'''
    
    shift=2
        
    h,w,c = image.shape
    
    basecoeff = np.array([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])
    
    coeff = np.dot(basecoeff.T, basecoeff)
    
    I1LRU = np.zeros((ratio*h, ratio*w, c))
    
    I1LRU[shift::ratio, shift::ratio, :]=image
    
    for i in range(c):
        temp = I1LRU[:, :, i]
        temp = ndimage.convolve(temp, coeff, mode='wrap')
        I1LRU[:, :, i]=temp
        
    return I1LRU

def gaussian2d (N, std):
    
    t=np.arange(-(N-1)/2,(N+2)/2)
    t1,t2=np.meshgrid(t,t)
    std=np.double(std)
    w = np.exp(-0.5*(t1/std)**2)*np.exp(-0.5*(t2/std)**2) 
    return w
    
def kaiser2d (N, beta):
    
    t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1,t2=np.meshgrid(t,t)
    t12=np.sqrt(t1*t1+t2*t2)
    w1=np.kaiser(N,beta)
    w=np.interp(t12,t,w1)
    w[t12>t[-1]]=0
    w[t12<t[0]]=0
    
    return w

def fir_filter_wind(Hd,w):
    """
	compute fir filter with window method
	Hd: 	desired freqeuncy response (2D)
	w: 		window (2D)
	"""
	
    hd=np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h=np.fft.fftshift(np.fft.ifft2(hd))
    h=np.rot90(h,2)
    h=h*w
    h=h/np.sum(h)
    
    return h

def no_ref_evaluate(pred, pan, hs):
    #no reference metrics
    c_D_lambda = D_lambda(pred, hs)
    c_D_s = D_s(pred, hs, pan)
    c_qnr = qnr(pred, hs, pan)
    c_qnr = (1-c_D_s)*(1-c_D_lambda)
    return [c_D_lambda, c_D_s, c_qnr]

def downgrade_images(I_MS, I_PAN, ratio, sensor=None):
    """
    downgrade MS and PAN by a ratio factor with given sensor's gains
    """
    I_MS=np.double(I_MS)
    I_PAN=np.double(I_PAN)
    
    I_MS = np.transpose(I_MS, (2, 0, 1))
    I_PAN = np.squeeze(I_PAN)
    
    ratio=np.double(ratio)
    flag_PAN_MTF=0
    
    if sensor=='QB':
        flag_resize_new = 2
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif sensor=='IKONOS':
        flag_resize_new = 2             #MTF usage
        GNyq = np.asarray([0.26,0.28,0.29,0.28],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.17;
    elif sensor=='GeoEye1':
        flag_resize_new = 2             # MTF usage
        GNyq = np.asarray([0.23,0.23,0.23,0.23],dtype='float32')    # Band Order: B,G,R,NIR
        GNyqPan = 0.16     
    elif sensor=='WV2':
        flag_resize_new = 2             # MTF usage
        GNyq = [0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.27]
        GNyqPan = 0.11
    elif sensor=='WV3':
        flag_resize_new = 2             #MTF usage
        GNyq = 0.29 * np.ones(8)
        GNyqPan = 0.15
    else:
        '''the default way'''
        flag_resize_new = 1
    
    '''the default downgrading method is gaussian'''
    if flag_resize_new == 1:
        
#        I_MS_LP = np.zeros((I_MS.shape[0],int(np.round(I_MS.shape[1]/ratio)+ratio),int(np.round(I_MS.shape[2]/ratio)+ratio)))
#            
#        for idim in range(I_MS.shape[0]):
#            imslp_pad=np.pad(I_MS[idim,:,:],int(2*ratio),'symmetric')
#            I_MS_LP[idim,:,:]=misc.imresize(imslp_pad,1/ratio,'bicubic',mode='F')
#            
#        I_MS_LR = I_MS_LP[:,2:-2,2:-2]
#       
#        I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
#        I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
#        I_PAN_LR=I_PAN_LR[2:-2,2:-2]
        
        sig = (1/(2*(2.772587)/ratio**2))**0.5
        kernel = np.multiply(cv2.getGaussianKernel(9, sig), cv2.getGaussianKernel(9,sig).T)
        
        t=[]
        for i in range(I_MS.shape[0]):
            temp = signal.convolve2d(I_MS[i, :, :], kernel, mode='same', boundary = 'wrap')
            temp = temp[0::int(ratio), 0::int(ratio)]
            temp = np.expand_dims(temp, 0)
            t.append(temp)
            
        I_MS_LR = np.concatenate(t, axis=0)
        
        I_PAN_LR = signal.convolve2d(I_PAN, kernel, mode='same', boundary = 'wrap')
        I_PAN_LR = I_PAN_LR[0::int(ratio), 0::int(ratio)]
        
    elif flag_resize_new==2:
        
        N=41
        I_MS_LP=np.zeros(I_MS.shape)
        fcut=1/ratio
        
        for j in range(I_MS.shape[0]):
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq[j])))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            w=kaiser2d(N,0.5)
            h=fir_filter_wind(Hd,w)
            I_MS_LP[j,:,:] = ndimage.filters.correlate(I_MS[j,:,:],np.real(h),mode='nearest')
        
        if flag_PAN_MTF==1:
            #fir filter with window method
            alpha = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyqPan)))
            H=gaussian2d(N,alpha)
            Hd=H/np.max(H)
            h=fir_filter_wind(Hd,w)
            I_PAN = ndimage.filters.correlate(I_PAN,np.real(h),mode='nearest')
            I_PAN_LR=I_PAN[int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]
            
        else:
            #bicubic resize
            I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
            I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
            I_PAN_LR=I_PAN_LR[2:-2,2:-2]
            
        I_MS_LR=I_MS_LP[:,int(ratio/2):-1:int(ratio),int(ratio/2):-1:int(ratio)]     
        
    I_MS_LR = np.transpose(I_MS_LR, (1, 2, 0))
    I_PAN_LR = np.expand_dims(I_PAN_LR, -1)
    
    return I_MS_LR,I_PAN_LR