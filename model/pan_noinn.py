# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine1
import torch.nn.init as init
import math
from functools import reduce
from model.base_net import *

# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
class Transformer_Fusion(nn.Module):
    def __init__(self,nc):
        super(Transformer_Fusion, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.Conv2d(2*nc,nc,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1))

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, ref_lv3):
        ######################   search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        res = self.conv_trans(torch.cat([T_lv3,lrsr_lv3],1))*S+lrsr_lv3

        return res
#
class PatchFusion(nn.Module):
    def __init__(self,nc):
        super(PatchFusion, self).__init__()
        self.fuse = Transformer_Fusion(nc)

    def forward(self,msf,panf):
        ori = msf
        b,c,h,w = ori.size()
        msf = F.unfold(msf,kernel_size=(24, 24), stride=8, padding=8)
        panf = F.unfold(panf, kernel_size=(24, 24), stride=8, padding=8)
        msf = msf.view(-1,c,24,24)
        panf = panf.view(-1,c,24,24)
        fusef = self.fuse(msf,panf)
        fusef = fusef.view(b,c*24*24,-1)
        fusef = F.fold(fusef, output_size=ori.size()[-2:], kernel_size=(24, 24), stride=8, padding=8)
        return fusef

#########################################################################################



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, channel_out)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1,self.conv2], 0.1)
        else:
            initialize_weights([self.conv1,self.conv2], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x2


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class FeatureExtract(nn.Module):
    def __init__(self, channel_in=3, channel_split_num=3, subnet_constructor=subnet('DBNet'), block_num=5):
        super(FeatureExtract, self).__init__()
        operations = []

        # current_channel = channel_in
        channel_num = channel_in

        for j in range(block_num):
            b = InvBlock(subnet_constructor, channel_num, channel_split_num)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.fuse = nn.Conv2d((block_num - 1) * channel_in, channel_in, 1, 1, 0)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x, rev=False):
        out = x  # x: [N,3,H,W]
        outfuse = out
        for i, op in enumerate(self.operations):
            out = op.forward(out, rev)
            if i > 1:
                outfuse = torch.cat([outfuse, out], 1)
        outfuse = self.fuse(outfuse)

        return outfuse


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class Conv_Fusion(nn.Module):
    def __init__(self,nc_in,nc_out):
        super(Conv_Fusion, self).__init__()
        self.conv = nn.Conv2d(nc_in*2,nc_out,3,1,1)
    def forward(self,pan,ms):
        return self.conv(torch.cat([ms,pan],1))

class Conv_Process(nn.Module):
    def __init__(self,ms_channels,pan_channels,nc):
        super(Conv_Process, self).__init__()
        self.convms = nn.Conv2d(ms_channels,nc,3,1,1)
        self.convpan = nn.Conv2d(pan_channels, nc, 3, 1, 1)

    def forward(self,pan,ms):
        return self.convpan(pan),self.convms(ms)



class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        ms_channels = 4
        pan_channels = 1
        n_feat = 16

        self.conv_process = Conv_Process(ms_channels,pan_channels,n_feat//2)
        self.conv_fusion = Conv_Fusion(n_feat//2,n_feat//2)

        self.transform_fusion = PatchFusion(n_feat//2)
        #self.extract = FeatureExtract(n_feat, n_feat//2,block_num=3)
        self.extract = nn.Sequential(DenseBlock(n_feat, n_feat), DenseBlock(n_feat, n_feat), DenseBlock(n_feat, n_feat))
        self.refine = Refine1(ms_channels + pan_channels, pan_channels, n_feat)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ms, b_ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)
        # finput = torch.cat([pan, mHR], dim=1)
        panf,mHRf = self.conv_process(pan,mHR)
        conv_f = self.conv_fusion(panf,mHRf)
        transform_f = self.transform_fusion(mHRf,panf)
        f_cat = torch.cat([conv_f,transform_f],1)
        # f_cat = conv_f
        fmid = self.extract(f_cat)
        HR = self.refine(fmid)+mHR

        return HR

