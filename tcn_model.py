import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import torch
from wavelet import gen_wavelet, gen_wavelet2


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
            # self.input_size = self.input_size // 2
            # wvlt_size = self.input_length * self.input_size // 2
            self.linear_wavelet = nn.Linear(100, wavelet_output_size)
            self.linear_wavelet.double()
            linear_size += 2 * wavelet_output_size

        self.tcn = TemporalConvNet(
            self.input_size,
            num_channels,
            kernel_size=kernel_size)

        self.linear2 = nn.Linear(2*input_channels, 30)

        self.input_bn = nn.BatchNorm1d(55)



    def forward(self, inputs, orinput):
        """Inputs have to have dimension (N, C_in, L_in)"""

        inputs = inputs.permute(0, 2, 1)
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        last = y1[:, :, -1]



        if self.wavelet:
            hwt = gen_wavelet2(orinput)
            # splits = torch.split(inputs, self.input_size, dim=2)
            # inputs = splits[0]
            # wvlt_inputs = splits[1]
            # wvlt_inputs_1 = torch.split(wvlt_inputs,
            #                             self.input_length // 2,
            #                             dim=0)[0]
            # wvlt_inputs_2 = torch.split(wvlt_inputs,
            #                             self.input_length // 2,
            #                             dim=0)[1]
            # bsize = inputs.size()[0]
            # t1 = torch.Tensor(hw1.T).type(torch.DoubleTensor).to(self.device)
            # wvlt_out1 = self.linear_wavelet(
            #     t1)
            # wvlt_out2 = self.linear_wavelet(
            #     torch.Tensor(hw2.T).type(torch.DoubleTensor).to(self.device))
            wt_out = self.linear2(torch.Tensor(hwt).type(torch.DoubleTensor).to(self.device))



        if self.wavelet:
            # temp1 = last[-1, :]
            # temp2 = wvlt_out1[-1, :]
            # temp3 = wvlt_out2[0, :]
            # last = torch.cat([temp1, temp2 , temp3])
            last = torch.cat([last, wt_out], dim=1)


        # last = torch.reshape(last, (1, 63))
        normalized = self.input_bn(last)
        # o = self.linear(normalized)
        # return o, {'orig': last, 'pos': None, 'neg': None}

        # o = self.linear3(normalized.T)
        return normalized

