import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet1DModel
from model_encoder import GaSNet3
import random


# loss function
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.beta = 1.0

    def forward(self, data, output):
        overlap_index = data['overlap_index']
        #input_spec = data['model']
        input_spec = data['fluctuate']
        ivar = data['ivar']
        chi_square_loss = []
        for i, index in zip(range(len(overlap_index)), overlap_index):
            one_input  = input_spec[i, 0:1, index[0]:index[1]]
            one_ivar = ivar[i, 0:1, index[0]:index[1]]
            one_output = output[i, 0:1, index[2]:index[3]]
            #temp_chi =  torch.mean(torch.abs(one_input-one_output), -1) #MAE loss
            temp_chi = F.smooth_l1_loss(one_output, one_input, reduction='none', beta=self.beta) #SmoothL1Loss
            temp_chi = temp_chi.mean(dim=-1)
            #temp_chi =  torch.mean((one_input-one_output)**2*one_ivar,-1) #chi-square loss
            chi_square_loss.append(temp_chi)
        chi_square_loss = torch.cat(chi_square_loss, dim=0)
        chi_square_loss = torch.mean(chi_square_loss, dim=-1)
        return chi_square_loss.mean()


# the network
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, egienV_num=10, dropout=0.0):
        super(Network, self).__init__()
        self.eigenvectors = nn.Parameter(1.0*torch.rand(egienV_num, output_dim), requires_grad=False)
        self.pad = 8**np.ceil(np.log(input_dim)/np.log(8)) - input_dim
        self.pad = int(self.pad)
        self.output_dim = output_dim
        self.egienV_num = egienV_num
        input_size = 4096
        self.out_channels = int(math.ceil(self.output_dim/input_size))
        #self.out_channels = 3
        self.UNet = UNet1DModel(sample_size=input_size,
                                in_channels=1,
                                out_channels=self.out_channels,
                                layers_per_block=2,
                                block_out_channels=(8, 16, 32, 64),
                                norm_num_groups = 8,
                                time_embedding_type = "positional",
                                use_timestep_embedding=False,
                                down_block_types=(
                                    "DownBlock1D", 
                                    "DownBlock1D",
                                    "DownBlock1D",
                                    "AttnDownBlock1D",),
                                up_block_types=(
                                    "AttnUpBlock1D",
                                    "UpBlock1D",
                                    "UpBlock1D",
                                    "UpBlock1D",))
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.flipped = False
        if dropout > 0:
            self.flipped = True
            
    '''
    def random_zero_out_segments(self, x):
        if self.dropout_rate <= 0.0:
            return x
        #Choose a random length between 0.0--0.5 x_len
        x_len = x.shape[-1]
        min_len = int(0.0*x_len)
        max_len = int(self.dropout_rate*x_len)
        for i in range(len(x)):
            # ensure the index does not excess the range of the tensor
            start_index = torch.randint(0, x_len-min_len, (1,)).item()
            end_index = torch.randint(start_index, min(x_len, start_index+max_len), (1,)).item()
            if start_index<end_index<x_len:
                # set it as ramdon values
                x[i, :, start_index:end_index] =  torch.rand(end_index-start_index)
        return x

    def noise_adding(self, x, device=torch.device('cuda:2')):
        if self.dropout_rate <= 0.0:
            return x
        noise = torch.randn_like(x)
        means = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        factors = 10*torch.rand_like(means)
        noise = factors * means * noise
        noise = noise.to(device)
        return x + noise
    
    def Parity(self, x, device=torch.device('cuda:2')):
        if self.dropout_rate <= 0.0:
            return x
        num_rows = x.shape[0] # row number of tensor
        rows_to_flip = random.sample(range(num_rows), k=num_rows // 2)  # Select half of the rows at random
        for row in rows_to_flip:
            x[row] = torch.flip(x[row], dims=[-1])  # Flip the selected row horizontally
        parity_tenser = torch.randint(0, 2, x.shape) * 2 - 1  # randint(0, 2) generates 0 or 1, which is then converted to -1 or 1
        parity_tenser = parity_tenser.to(device)
        x = x * parity_tenser
        return x
    '''

    def forward(self, data):
        x = data['fluctuate']
        if self.flipped:
            x = torch.flip(x, dims=[-1])
        x = F.pad(x, (0, self.pad))
        #if self.training:
        #    x = self.Parity(x)
        x = self.UNet(x, timestep=0)['sample']
        x = x.view(x.shape[0], 1, -1)
        x = x[:, :, 0:self.output_dim]
        coef_vec = torch.ones(x.shape[0], self.egienV_num)
        return x, coef_vec, self.eigenvectors
        

class GaSNet3_UNet(GaSNet3):
    def model_loader(self):
        egienV_num = self.cfg['egienV_num']
        dropout_rate = self.cfg['dropout']
        self.model = Network(self.input_dim, self.output_dim, egienV_num, dropout_rate)
        self.criterion = MyLoss()