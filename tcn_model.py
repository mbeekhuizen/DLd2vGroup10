import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import torch
from wavelet import gen_waveletConCat


class TCN(nn.Module):


    def __init__(self, input_channels, wavelet, input_length, output_size, kernel_size, channel_lst, wavelet_output_size, device):
        super(TCN, self).__init__()
        self.input_size = input_channels
        self.wavelet = wavelet
        self.input_length = input_length
        self.device = device
        output_size = output_size
        kernel_size = kernel_size
        num_channels = channel_lst
        wavelet_output_size = wavelet_output_size
        num_channels = [int(x) for x in num_channels.split(',')]
        linear_size = num_channels[-1]

        if self.wavelet:
            self.linear_wavelet = nn.Linear(100, wavelet_output_size)
            self.linear_wavelet.double()
            linear_size += 2 * wavelet_output_size

        self.tcn = TemporalConvNet(
            self.input_size,
            num_channels,
            kernel_size=kernel_size)

        self.linear = nn.Linear(2 * input_channels, 30)

        self.input_bn = nn.BatchNorm1d(55)



    def forward(self, inputs, orinput):
        """Inputs have to have dimension (N, C_in, L_in)"""

        inputs = inputs.permute(0, 2, 1)
        y1 = self.tcn(inputs)
        last = y1[:, :, -1]



        if self.wavelet:
            #Generate transformed matrix from input
            hwt = gen_waveletConCat(orinput)
            #Put the result through a fully connected layer
            wt_out = self.linear(torch.Tensor(hwt).type(torch.DoubleTensor).to(self.device))
            last = torch.cat([last, wt_out], dim=1)

        normalized = self.input_bn(last)

        return normalized

