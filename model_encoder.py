import torch, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils.data import DataLoader
from scipy.signal import find_peaks
import json
from pprint import pprint
from astropy.io import fits
from astropy.table import Table


if not os.path.exists('models'):
    os.mkdir('models')


# data loader
class MyDataset(Dataset):
    def __init__(self, data, input_grid, output_grid, device, poly_deg):
        self.poly_deg = poly_deg
        self.device = device
        self.input_grid, self.output_grid = input_grid, output_grid
        self.delta_loglamd = self.output_grid[1] - self.output_grid[0]
        print(f'delta loglamd = {self.delta_loglamd}')
        self.fluctuate, self.continuous, self.ivar = self.tranfomer(data['flux'], data['ivar']) # preprocess
        print('Spec shape=', self.fluctuate.shape, self.ivar.shape)
        print('Spec type=', self.fluctuate.dtype, self.ivar.dtype)

    def __len__(self):
        return len(self.fluctuate)
    
    def __getitem__(self, idx):
        if self.device == 'cpu':
            fluctuate, ivar = self.fluctuate[idx], self.ivar[idx]
        else:
            fluctuate, ivar = self.fluctuate[idx].clone().detach().float(), self.ivar[idx].clone().detach().float()
        return {'fluctuate':fluctuate, 'ivar':ivar, 'continuous':self.continuous[idx]}

    
    # transformation of data
    def tranfomer(self, flux, ivar):
        # Normalization
        dim = len(self.input_grid)
        input_grid = self.input_grid
        deg = self.poly_deg
        # poly fitting, only fitting the points of ivar != 0
        polyfit_1d = lambda flux_ivar: np.polyval(np.polyfit(input_grid[(flux_ivar[dim:]>0)], flux_ivar[:dim][(flux_ivar[dim:]>0)], deg=deg), input_grid)
        continuous = np.apply_along_axis(polyfit_1d, axis=-1, arr=np.hstack([flux,ivar]))
        continuous = continuous.astype(np.float32) #Note that we all work in the data form of float32!!
        fluctuate = flux-continuous
        #=== The normalization
        Norm = np.mean(fluctuate**2,axis=-1)**(0.5)
        Norm = Norm.reshape(-1,1)
        fluctuate = fluctuate/Norm
        ivar = ivar*Norm**2
        #===
        fluctuate = torch.from_numpy(fluctuate)
        fluctuate = fluctuate.view(len(fluctuate),1,len(fluctuate[0]))
        fluctuate = fluctuate.to(self.device) # to GPU
        # inver variance
        ivar = torch.from_numpy(ivar)  # transfer to tensor
        ivar = ivar.view(len(ivar),1,len(ivar[0]))
        ivar = ivar.to(self.device) # to GPU
        return fluctuate, continuous, ivar



# the network
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, egienV_num):
        super(Network, self).__init__()
        # learnable eigenvectors
        self.egienV_num = egienV_num
        self.eigenvectors = nn.Parameter(1.0*torch.rand(egienV_num, output_dim), requires_grad=True)
        # cnn layer encoder
        fun = lambda n : (((n - 9 + 2)//5 + 1) - 2)//2 + 1 # one layer
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*fun(fun(fun(input_dim))), egienV_num+1),
            )

    def forward(self, data):
        x = self.encoder(data['fluctuate'])
        amp = torch.abs(x[:, self.egienV_num:])
        amp = amp.view(x.size(0), 1, 1)
        coef_vec = x[:, 0:self.egienV_num]
        coef_vec = coef_vec**2/torch.sum(coef_vec**2, dim=-1).view(coef_vec.size(0),1)
        coef_vec = coef_vec.view(x.size(0), 1, self.egienV_num)
        x = amp*torch.matmul(coef_vec, self.eigenvectors)
        return x, coef_vec, self.eigenvectors


class GaSNet3:
    """
    Initialize, one shold set the wavelength_min, wavelength_max, output_label, and the name of model.
    """
    def __init__(self):
        
        self.device = torch.device('cuda:2')
        self.cfg = {'model_name': 'GaSNet3',
                    'batch_size': 128,
                    'start_learning_rate': 1e-3,
                    'end_learning_rate': 1e-4,
                    'epochs': 100,
                    'step_size':20,
                    'wavelength_min': 3600,
                    'wavelength_max': 9200,
                    'delta_loglamd': 1e-4,
                    'z_max': 2,
                    'z_min':0,
                    'poly_deg':5,
                    'egienV_num':10,
                    'SNR_thr':0.0,
                    'dropout':0.0,
                    }
        self.train_data = {}
        self.valid_data = {}
        self.test_data =  {}


    def Init(self, load_json=False, json_name=''):
        if not os.path.exists('models/'+self.cfg['model_name']):
            os.mkdir('models/'+self.cfg['model_name'])
        # load the model setting from json
        if load_json:
            with open(json_name, 'r') as json_file:
                config_dict = json.load(json_file)
            pprint(config_dict)
            self.cfg.update(config_dict)

        max_wave, min_wave = self.cfg['wavelength_max'], self.cfg['wavelength_min']
        delta = self.cfg['delta_loglamd']
        z_min, z_max = self.cfg['z_min'], self.cfg['z_max']
        print('z_min, z_max =', z_min, z_max)

        self.input_dim = round(np.log10(max_wave/min_wave)/delta + 1)
        self.input_grid = np.linspace(np.log10(min_wave), np.log10(max_wave), self.input_dim, endpoint=True)
        print(f'input spectrum wavelength {min_wave}-{max_wave} AA, dim={self.input_dim}')

        self.output_dim = round(self.input_dim + np.log10(1+z_max)/delta - np.log10(1+z_min)/delta)
        self.output_grid = np.linspace(-self.input_dim*delta-np.log10(1+z_max), -np.log10(1+z_min), num=self.output_dim, endpoint=True) + np.log10(max_wave)
        print('egienspectrum dim:', self.output_dim, self.output_grid)
        
        model_name = self.cfg['model_name']
        self.history = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'time', 'learning rate'])
        self.cfg['output_pth'] = f'models/{model_name}/{model_name}.pth'
        self.cfg['output_csv'] = self.cfg['output_pth'].replace('.pth','.csv')
        self.json_name = json_name

    def model_loader(self):
        egienV_num = self.cfg['egienV_num']
        self.model = Network(self.input_dim, self.output_dim, egienV_num)

    # correction of chi2_curev
    def corrected_square_curve(self, chi_square_curves):
        correction_factor = torch.ones_like(chi_square_curves)
        hlft_width = len(self.input_grid)//2
        correction_factor[:,:, 0:hlft_width] = 2 - torch.arange(0, hlft_width)/hlft_width
        correction_factor[:,:, -hlft_width:] = 1 + torch.arange(0, hlft_width)/hlft_width
        return chi_square_curves*correction_factor

    # the chi-square curves
    def chi_square_curve(self, input_spetrum, output_spetrum, ivar):
        chi_square_curves = []
        #=== the chi-square curve is divied into 3 terms
        for i in range(len(input_spetrum)):
            ivar_term  = F.conv1d(output_spetrum[i:i+1]**2, ivar[i:i+1], padding='same') #invert variance term
            constant = torch.sum(input_spetrum[i:i+1]**2 * ivar[i:i+1], dim=-1).unsqueeze(-1) #constant term
            constant = constant.repeat(1, 1, constant.shape[-1])
            corcorrelation = F.conv1d(output_spetrum[i:i+1], 2 * input_spetrum[i:i+1] * ivar[i:i+1], padding='same') # correlation term
            curve = ivar_term + constant - corcorrelation
            curve = self.corrected_square_curve(curve)
            chi_square_curves.append(curve)
        chi_square_curves = torch.cat(chi_square_curves)
        #== correction 
        chi_square_curves = self.corrected_square_curve(chi_square_curves)
        return chi_square_curves


    # get the minimum chi-square, best-fit redshift, and Degeneracy
    def get_parameters_from_chi_square_curve(self, chi_square_curve):
        min_indx = np.argmin(chi_square_curve)
        best_fit_z = 10**((self.input_grid[0]+self.input_grid[-1])/2 - self.output_grid[min_indx])-1
        min_chi_square = np.min(chi_square_curve)/len(self.input_grid)
        peak_indxs, _ = find_peaks(-chi_square_curve, distance=0.01/self.cfg['delta_loglamd'])
        peaks = chi_square_curve[peak_indxs]
        #===finding the top 2 min peaks
        top_indx = np.argsort(peaks)[0:2]
        peaks, peak_indxs = peaks[top_indx], peak_indxs[top_indx]
        if len(peaks)<2:
            Degeneracy = -1
        else:
            Degeneracy = (peaks[1]-peaks[0])/peaks[0]
        return min_chi_square, best_fit_z, Degeneracy

    # return the predict redshift and D,
    def prediction(self, fname):
        # load data
        self.test_dataset = MyDataset(self.test_data, self.input_grid, self.output_grid, self.device, self.cfg['poly_deg'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg['batch_size'])
        # load model
        self.model_loader()
        self.model.load_state_dict(torch.load(self.cfg['output_pth'], map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        # get reconstructed spectra, coefficients, and chi-square curves
        output_spectra, coefficients, chi_square_curves = [], [], []
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc='Testing'):
                batch_output, batch_coefficients, eigenvectors = self.model(data)
                batch_input_spectra, batch_ivar = data['fluctuate'], data['ivar']
                batch_chi_square_curves = self.chi_square_curve(batch_input_spectra, batch_output, batch_ivar)
                output_spectra.append(batch_output)
                coefficients.append(batch_coefficients)
                chi_square_curves.append(batch_chi_square_curves)
        output_spectra, coefficients, chi_square_curves = torch.cat(output_spectra), torch.cat(coefficients), torch.cat(chi_square_curves)
        output_spectra = output_spectra.cpu().detach().numpy()
        coefficients = coefficients.cpu().detach().numpy()
        chi_square_curves = chi_square_curves.cpu().detach().numpy()

        # get the minimum chi-square, best-fit redshift, and Degeneracy
        min_chi_squares, Best_fit_z, Degeneracies = [], [], []
        for curve in chi_square_curves:
            curve = curve.flatten()
            min_chi2, best_z, degeneracy = self.get_parameters_from_chi_square_curve(curve)
            min_chi_squares.append(min_chi2)
            Best_fit_z.append(best_z)
            Degeneracies.append(degeneracy)

        # save the parameters
        test_data = self.test_data
        #=== saving the results to fits
        test_data['Best_fit_z'] = np.array(Best_fit_z)
        test_data['min_chi_square'] = np.array(min_chi_squares)
        test_data['Degeneracy'] = np.array(Degeneracies)
        test_data['coefficients'] = coefficients
        
        results = test_data[:]
        del results['flux']
        del results['ivar']
        if 'reconstruction' in results.keys():
            del results['reconstruction']
            del results['chi_square_curve']
        fn = f'results/{fname}.fits'
        self.save_as_fits(fn, results)
        

    # save as fits file
    def save_as_fits(self, fname, t1):
        t2 = Table({'json_name':[self.json_name], 'input_grid': [self.input_grid], 'output_grid': [self.output_grid]})
        hdu1 = fits.BinTableHDU(t1)
        hdu2 = fits.BinTableHDU(t2)
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu1, hdu2])
        hdul.writeto(fname, overwrite=True)


    def get_eigenspec(self):
        self.model_loader()
        self.model.load_state_dict(torch.load(self.cfg['output_pth'], map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        eigenspec = self.model.eigenvectors
        return eigenspec.cpu().detach().numpy()
    

