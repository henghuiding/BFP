###########################################################################
# Created by: NTU EEE
# Email: ding0093@e.ntu.edu.sg
# Copyright (c) 2019
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from torch.nn import Module, Conv1d, ReLU, Parameter
from ..models import BaseNet


__all__ = ['BFP', 'get_bfp']

class BFP(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BFP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = BFPHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)

        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])

        return tuple(outputs)
        
class BFPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(BFPHead, self).__init__()
        inter_channels = in_channels // 4
        self.no_class=out_channels
        self.adapt1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.adapt2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12 , padding=12, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.adapt3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, dilation=12 , padding=12, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.uag_rnn = UAG_RNN(inter_channels)
        self.seg1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels+1, 1))
        self.seg2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(2*torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1)/out_channels)

    def forward(self, x):
        # adapt from CNN
        feat1 = self.adapt1(x)
        feat2 = self.adapt2(feat1)
        # Boundary
        s1_output = self.seg1(feat2)
        s1_output_ = self.softmax(s1_output)
        score_ = torch.narrow(s1_output, 1, 0, self.no_class) 
        boundary_ = torch.narrow(s1_output_, 1, self.no_class, 1)

        ## boundary confidence to propagation confidence, method 1
        # boundary = 1 - self.sigmoid(20*boundary_-4)*self.gamma

        ## boundary confidence to propagation confidence, method 2
        boundary = torch.mean(torch.mean(boundary_, 2, True), 3, True)-boundary_+self.bias 
        boundary = (boundary - torch.min(torch.min(boundary, 3, True)[0], 2, True)[0])*self.gamma

        boundary = torch.clamp(boundary, max=1)
        boundary = torch.clamp(boundary, min=0)
        ## UAG-RNN
        feat3 = self.adapt3(feat1)
        uag_feat = self.uag_rnn(feat3, boundary)
        feat_sum = uag_feat + feat3 #residual
        s2_output = self.seg2(feat_sum)
        # sd_output = self.conv7(sd_conv)
        output1 = s2_output + score_

        output = [output1]
        output.append(s1_output)
        # import pdb
        # pdb.set_trace()
        return tuple(output)


def get_bfp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""BFP model from the paper `"Boundary-Aware Feature Propagation for Scene Segmentation"
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Boundary-Aware_Feature_Propagation_for_Scene_Segmentation_ICCV_2019_paper.pdf>`
    """
    acronyms = {
        'pascalcontext': 'pascalcontext',
        'ade20k': 'ade',
        'camvid': 'camvid',
    }
    # infer number of classes
    from ..datasets import datasets
    model = BFP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model



class DAG_RNN(Module):
    """ Directional Acyclic Graphs (DAGs)"""
    def __init__(self, in_dim):
        super(DAG_RNN, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()


        self.gamma1 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma2 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma3 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma4 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma5 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma6 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma7 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma8 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma9 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma10 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma11 = Parameter(torch.zeros(in_dim,in_dim))
        self.gamma12 = Parameter(torch.zeros(in_dim,in_dim))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        x_new = x.permute(1, 0, 2, 3)
        ## se plane
        hse = x_new*1
        for i in range(height):
            for j in range(width):
                if i>0:
                    hse[:,:,i,j] = hse[:,:,i,j].clone() + torch.mm(self.gamma1, hse[:,:,i-1,j].clone())
                if j>0:
                    hse[:,:,i,j] = hse[:,:,i,j].clone() + torch.mm(self.gamma2, hse[:,:,i,j-1].clone())
                if i>0 & j>0:
                    hse[:,:,i,j] = hse[:,:,i,j].clone() + torch.mm(self.gamma3, hse[:,:,i-1,j-1].clone())
                hse[:,:,i,j] = self.relu(hse[:,:,i,j].clone())

        ## sw plane
        hsw = x_new*1
        for i in reversed(range(height)):
            for j in range(width):
                if i<(height-1):
                    hsw[:,:,i,j] = hsw[:,:,i,j].clone() + torch.mm(self.gamma4, hsw[:,:,i+1,j].clone())
                if j>0:
                    hsw[:,:,i,j] = hsw[:,:,i,j].clone() + torch.mm(self.gamma5, hsw[:,:,i,j-1].clone())
                if i<(height-1) & j>0:
                    hsw[:,:,i,j] = hsw[:,:,i,j].clone() + torch.mm(self.gamma6, hsw[:,:,i+1,j-1].clone())
                hsw[:,:,i,j] = self.relu(hsw[:,:,i,j].clone())

        ## nw plane
        hnw = x_new*1
        for i in reversed(range(height)):
            for j in reversed(range(width)):
                if i<(height-1):
                    hnw[:,:,i,j] = hnw[:,:,i,j].clone() + torch.mm(self.gamma7, hnw[:,:,i+1,j].clone())
                if j<(width-1):
                    hnw[:,:,i,j] = hnw[:,:,i,j].clone() + torch.mm(self.gamma8, hnw[:,:,i,j+1].clone())
                if i<(height-1) & j<(width-1):
                    hnw[:,:,i,j] = hnw[:,:,i,j].clone() + torch.mm(self.gamma9, hnw[:,:,i+1,j+1].clone())
                hnw[:,:,i,j] = self.relu(hnw[:,:,i,j].clone())

        ## ne plane
        hne = x_new*1
        for i in range(height):
            for j in reversed(range(width)):
                if i>0:
                    hne[:,:,i,j] = hne[:,:,i,j].clone() + torch.mm(self.gamma10, hne[:,:,i-1,j].clone())
                if j<(width-1):
                    hne[:,:,i,j] = hne[:,:,i,j].clone() + torch.mm(self.gamma11, hne[:,:,i,j+1].clone())
                if j<(width-1) & i>0:
                    hne[:,:,i,j] = hne[:,:,i,j].clone() + torch.mm(self.gamma12, hne[:,:,i-1,j+1].clone())
                hne[:,:,i,j] = self.relu(hne[:,:,i,j].clone())

        out = hse + hsw + hnw + hne
        out = out.permute(1, 0, 2, 3)
        return out
class UAG_RNN(Module):
    """Unidirectional Acyclic Graphs (UCGs)"""
    def __init__(self, in_dim):
        super(UAG_RNN, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU()


        self.gamma1 = Parameter(0.5*torch.ones(1))
        self.gamma2 = Parameter(0.5*torch.ones(1))
        self.gamma3 = Parameter(0.5*torch.ones(1))
        self.gamma4 = Parameter(0.5*torch.ones(1))
        self.conv1 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv2 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv3 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv4 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv5 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv6 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv7 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv8 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv9 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv10 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv11 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv12 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv13 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv14 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv15 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv16 = Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
    def forward(self, x, y):
        m_batchsize, C, height, width = x.size()

        ## s plane
        hs = x*1
        for i in range(height):
            if i>0:
                hs[:,:,i,:] = self.conv1(hs[:,:,i,:].clone()) + self.conv2(hs[:,:,i-1,:].clone())*y[:,:,i-1,:]
                hs[:,:,i,:] = self.relu(hs[:,:,i,:].clone())

        ## s plane
        hse = hs*1
        for j in range(width):
            if j>0:
                tmp = self.conv3(hse[:,:,:,j-1].clone())*y[:,:,:,j-1]
                tmp = torch.cat((0*tmp[:,:,-1].view(m_batchsize, C, 1), tmp[:,:,0:-1]),2)##diagonal
                hse[:,:,:,j] = self.conv4(hse[:,:,:,j].clone()) + self.conv5(hse[:,:,:,j-1].clone())*y[:,:,:,j-1] + self.gamma1*tmp
                del tmp         
            hse[:,:,:,j] = self.relu(hse[:,:,:,j].clone())

        ## sw plane
        hsw = hs*1
        for j in reversed(range(width)):
            if j<(width-1):
                tmp = self.conv6(hsw[:,:,:,j+1].clone())*y[:,:,:,j+1]
                tmp = torch.cat((0*tmp[:,:,-1].view(m_batchsize, C, 1), tmp[:,:,0:-1]),2)##diagonal
                hsw[:,:,:,j] = self.conv7(hsw[:,:,:,j].clone()) + self.conv8(hsw[:,:,:,j+1].clone())*y[:,:,:,j+1] + self.gamma2*tmp
                del tmp
            hsw[:,:,:,j] = self.relu(hsw[:,:,:,j].clone())

        ## n plane
        hn = x*1
        for i in reversed(range(height)):
            if i<(height-1):
                hn[:,:,i,:] = self.conv9(hn[:,:,i,:].clone()) + self.conv10(hn[:,:,i+1,:].clone())*y[:,:,i+1,:]
            hn[:,:,i,:] = self.relu(hn[:,:,i,:].clone())

        ## ne plane
        hne = hn*1
        for j in range(width):
            if j>0:
                tmp = self.conv11(hne[:,:,:,j-1].clone())*y[:,:,:,j-1]
                tmp = torch.cat((tmp[:,:,1:], 0*tmp[:,:,0].view(m_batchsize, C, 1)),2)##diagonal
                hne[:,:,:,j] = self.conv12(hne[:,:,:,j].clone()) + self.conv13(hne[:,:,:,j-1].clone())*y[:,:,:,j-1] + self.gamma3*tmp
                del tmp
            hne[:,:,:,j] = self.relu(hne[:,:,:,j].clone())

        ## nw plane
        hnw = hn*1
        for j in reversed(range(width)):
            if j<(width-1):
                tmp = self.conv14(hnw[:,:,:,j+1].clone())*y[:,:,:,j+1]
                tmp = torch.cat((tmp[:,:,1:], 0*tmp[:,:,0].view(m_batchsize, C, 1)),2)##diagonal
                hnw[:,:,:,j] = self.conv15(hnw[:,:,:,j].clone()) + self.conv16(hnw[:,:,:,j+1].clone())*y[:,:,:,j+1] + self.gamma4*tmp
                del tmp
            hnw[:,:,:,j] = self.relu(hnw[:,:,:,j].clone())
            
        out = hse + hsw + hnw + hne

        return out

