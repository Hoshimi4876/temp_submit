"""
   crown.py
   COMP9444, CSE, UNSW
   Author: Kieran BAI 5397395
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full3Net(torch.nn.Module):
    def __init__(self, hid): # defaulthid = 10
        super(Full3Net, self).__init__()
        self.hid1 = nn.Tanh()
        self.hid2 = nn.Tanh()
        self.out = nn.Sigmoid()

    def forward(self, input):
        output = self.hid1(input[:,0])
        output = self.hid2(output[:,0])
        output = self.out(output[:,0])
        return output

class Full4Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full4Net, self).__init__()
        self.hid1 = nn.Tanh()
        self.hid2 = nn.Tanh()
        self.hid3 = nn.Tanh()
        self.out = nn.Sigmoid()

    def forward(self, input):
        output = self.hid1(input[:,0])
        output = self.hid2(output[:,0])
        output = self.hid3(output[:,0])
        output = self.out(output[:,0])
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.hid1 = nn.Tanh()
        self.hid2 = nn.Tanh()

    def forward(self, input):
        output = self.hid1(input[:,0])
        output = self.hid2(output[:,0])
        return 0*input[:,0]
