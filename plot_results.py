import torch, os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from astropy.table import Table
import model_encoder as model
import seaborn as sns
sns.set_theme(style="ticks", palette="deep")
if not os.path.exists('figures'):
    os.mkdir('figures')

def calculate_continuous_and_Norm(flux, iavr, model, input_grid):

    input_spectrum = flux.flatten()
    continuous  = np.polyval(np.polyfit(input_grid[(iavr>0)], input_spectrum[(iavr>0)], deg=5), input_grid)
    input_spectrum = input_spectrum - continuous
    Norm = np.mean(input_spectrum**2)**0.5
    input_spectrum = input_spectrum/Norm
    models = (model - continuous)/Norm
    iavr = iavr*Norm**2
    return input_spectrum, iavr, models



#=== roughly display the prediction results
def plot_scatter(file_name, model_name, threshold=0.001, max_z=5):
    spec = Table.read(file_name)
    spec = spec[spec['Z']<max_z]
    best_fit_z, pipeline_z, delta_z = spec['Best_fit_z'], spec['Z'], spec['delta_z']
    min_chi_square = spec['min_chi_square']
    num = len(pipeline_z)
    fig, axs = plt.subplots(1, 3, figsize=(24, 6), dpi=160)
    axs[0].set_title(f'All-{model_name}-redshift, num={num}')
    axs[0].scatter(pipeline_z, best_fit_z,s=2,alpha=0.5, color='b',label='$\Delta z ={}$'.format(np.mean(delta_z)))
    axs[0].set_xlabel('real z')
    axs[0].set_ylabel('pred z')
    axs[0].legend()
    axs[1].set_title(f'GF={model_name}-redshift, num={num}')
    axs[1].axhline(y=threshold, color='r', linestyle='--')
    axs[1].scatter(pipeline_z, delta_z,s=2,alpha=0.5, color='b',label='GF ={}'.format(sum(delta_z<=threshold)/len(delta_z)))
    axs[1].set_xlabel('real z')
    axs[1].set_ylabel('delta z')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[2].set_title('$\chi^2_{min}$ dist,' + 'num={}'.format(len(min_chi_square)))
    axs[2].hist(min_chi_square, bins=100, range=(0, 20),density=False,label='mean={:2f}'.format(np.mean(min_chi_square)))
    axs[2].set_xlabel('$\chi^2_{min}$')
    axs[2].set_ylabel('Num')
    axs[2].set_yscale('log')
    plt.savefig(f'./figures/{model_name}.pdf', bbox_inches='tight', pad_inches=0.1)


def plot_reconstruction(file_name, show_all=True, indx=np.arange(10), model_name='Encoder', spec_type='x'):

    spec = Table.read(file_name, 1)
    if not show_all:
        spec = spec[indx]
    info = Table.read(file_name, 2)
    input_grid, output_grid = info['input_grid'][0], info['output_grid'][0]
    delta_loglamd = 1e-4

    if 'PLATE' in spec.keys():
        tem = np.array(['-',] * len(spec['PLATE']))
        id = np.char.add(spec['PLATE'].astype(str), tem)
        id = np.char.add(id, spec['MJD'].astype(str))
        id = np.char.add(id, tem)
        id = np.char.add(id, spec['FIBERID'].astype(str))
    else:
        id = spec['TARGETID']

    rows, cols = len(spec), 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*20, rows*6), dpi=160)
    if rows == 1:
        axs = [axs]
    f_size = 24
    for i in range(rows):
        input_spectrum, iavr, models = calculate_continuous_and_Norm(spec['flux'][i], spec['ivar'][i], spec['model'][i], input_grid)
        Norm_value = np.median(np.abs(input_spectrum))
        output_spectrum = spec['reconstruction'][i].flatten()
        chi_square_curve = spec['chi_square_curve'][i].flatten()
        pipeline_z, best_fit_z, min_chi_square, Degeneracy, subclass = spec['Z'][i],  spec['Best_fit_z'][i], spec['min_chi_square'][i], spec['Degeneracy'][i], spec['SUBCLASS'][i]
        Class, SNR = spec['CLASS'][i], spec['SN_MEDIAN_ALL'][i]
        pipeline_shift = np.log10(1+pipeline_z)
        best_fit_wave_shift = np.log10(1+best_fit_z)
        model_chi2 = np.sum((input_spectrum - models)**2*iavr)/len(input_spectrum)

        axs[i].plot(input_grid-pipeline_shift, models/Norm_value, linewidth=1.5, alpha=0.5, c='blue', label=f'SDSS model, $\chi^2={model_chi2:.2f}$, $z$={pipeline_z:.4f}')
        axs[i].plot(output_grid, chi_square_curve/np.max(chi_square_curve)*4+4, linewidth=1.5, alpha=1, c='red', label=f'{model_name} curve, $\chi_{spec_type}^2$={min_chi_square:.2f}, $R_{spec_type}$={Degeneracy:.2f}, $z_{spec_type}$={best_fit_z:.4f}')
        axs[i].vlines(x=output_grid[np.argmin(chi_square_curve)], ymin=-5, ymax=8, linewidth=1.0, alpha=1.0, colors='red', linestyles='--')

        axs[i].plot(output_grid, output_spectrum/Norm_value, linewidth=0.5, alpha=1, c='black',label='$\hat{F}_i$')
        axs[i].plot(input_grid-best_fit_wave_shift, input_spectrum/Norm_value, linewidth=0.2,alpha=0.5, c='black', label=f'$F_i$')

        #chi2_peaks, peaks_dic = find_peaks(-chi_square_curve, distance=0.01/delta_loglamd)
        #axs[i].plot(input_grid-best_fit_wave_shift, input_spectrum, linewidth=0.3, alpha=0.3, c='steelblue')
        #axs[i].plot(input_grid-pipeline_shift, input_spectrum, linewidth=0.3,alpha=0.3, c='black', label='$\widetilde{F}$, '+'$z_t$={:.4f}'.format(pipeline_z))
        subclass = subclass.replace(' ', 'nan')
        if show_all:
            axs[i].set_title(f'Id: {id[i]}, {Class}-{subclass}, z={pipeline_z:.4f}, SNR={SNR:.2f}, index={i}', fontsize=f_size)
        else:
            axs[i].set_title(f'Id: {id[i]}, {Class}-{subclass}, z={pipeline_z:.4f}, SNR={SNR:.2f}', fontsize=f_size)
        #axs[i].set_ylim(-10, 10)
        axs[i].set_ylim(-10, 20)
        legend = axs[i].legend(ncol=2, loc='upper right', fontsize=f_size, framealpha=0.1)
        axs[i].tick_params(axis='x', labelsize=f_size)
        axs[i].tick_params(axis='y', labelsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        for handle in legend.legendHandles:
            handle.set_linewidth(4.0)

    plt.ylabel('Norm flux', fontsize=f_size)
    plt.xlabel('Log$_{10}(\lambda)$', fontsize=f_size)
    plt.savefig(f'./figures/{model_name}_{str(show_all)}_reconstruction.pdf', bbox_inches='tight', pad_inches=0.1)


def plot_eigenspec(fname):
    info = Table.read(fname, 2)
    json_name, input_grid, output_grid = info['json_name'][0], info['input_grid'][0], info['output_grid'][0]
    Encoder = model.GaSNet3()
    Encoder.device = torch.device('cpu')
    Encoder.Init(load_json=True, json_name=json_name)
    Encoder_model = model.Network(Encoder.input_dim, Encoder.output_dim, Encoder.cfg['egienV_num'])
    Encoder_model.load_state_dict(torch.load(Encoder.cfg['output_pth'], map_location=Encoder.device))
    eigenspec = Encoder_model.eigenvectors
    num = len(eigenspec)
    rows,cols = num, 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*20, rows*4), dpi=160)
    for i, _ in enumerate(eigenspec):
        flux = eigenspec[i].cpu().detach().numpy()
        axs[i].plot(Encoder.output_grid, flux,label = f'{i}th eigenvector')
        axs[i].legend()
    plt.ylabel('Norm flux')
    plt.xlabel('rest-frame Log$_{10}(\lambda)$')
    plt.savefig('./figures/' + Encoder.cfg['model_name']+'_Eigenspectra.pdf', bbox_inches='tight', pad_inches=0.1)


def plot_all_chi_square_curve(model_names, indx=np.arange(10), svae_name='all_chi_square', shift_legends=False):
    spec_tabel = []
    input_grid_list, output_grids_list = [], []
    for fn in model_names:
        spec_tabel.append(Table.read(f'./results/test_classfy_{fn}_reconstruction_examples.fits', 1)[indx])
        info = Table.read(f'./results/test_classfy_{fn}_reconstruction_examples.fits', 2)
        input_grid_list.append(info['input_grid'][0])
        output_grids_list.append(info['output_grid'][0])
    delta_loglamd = 1e-4
    spec = spec_tabel[0]
    if 'PLATE' in spec.keys():
        tem = np.array(['-',] * len(spec['PLATE']))
        id = np.char.add(spec['PLATE'].astype(str), tem)
        id = np.char.add(id, spec['MJD'].astype(str))
        id = np.char.add(id, tem)
        id = np.char.add(id, spec['FIBERID'].astype(str))
    else:
        id = spec['TARGETID']
    
    rows,cols = len(spec_tabel[0]), 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*20, rows*6), dpi=160)
    f_size = 24
    colors = ['red', 'green', 'blue', ]
    labels = ['s', 'g', 'q']
    names = ['STAR', 'GALAXY', 'QSO']
    for i in range(rows):
        if rows == 1:
            ax = axs
        else:
            ax = axs[i]
        min_chi_square_list =  np.array([spectra[i]['min_chi_square'] for spectra in spec_tabel])
        Degeneracy_list =  np.array([spectra[i]['Degeneracy'] for spectra in spec_tabel])
        Best_fit_z_list =  np.array([spectra[i]['Best_fit_z'] for spectra in spec_tabel])
        chi_square_curve_list = np.array([spectra['chi_square_curve'][i].flatten() for spectra in spec_tabel])

        best_index = np.argmin(min_chi_square_list)
        spec = spec_tabel[best_index]
        input_grid, output_grid = input_grid_list[best_index], output_grids_list[best_index]
        input_spectrum, iavr, models = calculate_continuous_and_Norm(spec['flux'][i], spec['ivar'][i], spec['model'][i], input_grid)
        output_spectrum = spec['reconstruction'][i].flatten()
        pipeline_z, best_fit_z, Degeneracy, Class, subclass = spec['Z'][i], spec['Best_fit_z'][i], spec['Degeneracy'][i], spec['CLASS'][i], spec['SUBCLASS'][i]
        pipeline_shift = np.log10(1+pipeline_z)
        best_fit_wave_shift = np.log10(1+best_fit_z)
        model_chi2 = np.sum((input_spectrum - models)**2*iavr)/len(input_spectrum)

        lsty = ['solid','dashed', 'dashdot']
        for x, chi_square_curve, min_chi_square, zp, Den, color, label, name, ls in zip(output_grids_list, chi_square_curve_list, min_chi_square_list, Best_fit_z_list, Degeneracy_list, colors, labels, names, lsty):
            label = f'{name} curve, ($\chi_{label}^2$={min_chi_square:.2f}, $R_{label}$={Den:.2f}, $z_{label}$={zp:.4f})'
            shifted_curve = chi_square_curve/np.max(chi_square_curve)+1.2
            ax.plot(x, shifted_curve, linewidth=1.5, alpha=1.0, c=color, label=label, linestyle=ls)
            ax.vlines(x=x[np.argmin(chi_square_curve)], ymin=-1,ymax=np.min(shifted_curve), linewidth=1.5, alpha=1.0, colors=color, linestyles='--')

        ax.plot(output_grid, output_spectrum, linewidth=0.5, alpha=1,c='black',label='$\hat{F}_i$')
        ax.plot(input_grid-best_fit_wave_shift, input_spectrum, linewidth=0.1, alpha=0.3, c='black', label=f'$F_i$')
        
        subclass = subclass.replace(' ', 'nan')
        ax.set_title(f'Id: {id[i]}, {Class}-{subclass}, z={pipeline_z:.4f}', fontsize=f_size)
        if shift_legends:
            ax.set_ylim(-1, 5)
        else:
            ax.set_ylim(-1, 3)
        ax.set_xlim(output_grid[0], output_grid[-1])
        #legend = ax.legend(fontsize=f_size, loc='upper left', framealpha=0.1)
        legend = ax.legend(ncol=2, fontsize=f_size, loc=(0.05, 0.5), framealpha=0.1)
        ax.tick_params(axis='x', labelsize=f_size)
        ax.tick_params(axis='y', labelsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        for handle in legend.legendHandles:
            handle.set_linewidth(4.0)

    plt.ylabel('Norm flux', fontsize=f_size)
    plt.xlabel('Log$_{10}(\lambda)$', fontsize=f_size)
    plt.savefig(f'./figures/{svae_name}_reconstruction.pdf', bbox_inches='tight', pad_inches=0.1)



def plot_reconstruction_residual(file_name, model_name='Encoder', back_z=0):
    Emi_line = Table.read('lines/SDSS_EL.csv')
    spec = Table.read(file_name, 1)
    info = Table.read(file_name, 2)
    input_grid, output_grid = info['input_grid'][0], info['output_grid'][0]

    if 'PLATE' in spec.keys():
        tem = np.array(['-',] * len(spec['PLATE']))
        id = np.char.add(spec['PLATE'].astype(str), tem)
        id = np.char.add(id, spec['MJD'].astype(str))
        id = np.char.add(id, tem)
        id = np.char.add(id, spec['FIBERID'].astype(str))
    else:
        id = spec['TARGETID']

    rows, cols = len(spec), 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols*20, rows*4), dpi=160)
    if rows == 1:
        axs = [axs]
    f_size = 14
    for i in range(rows):
        input_spectrum, iavr, models = calculate_continuous_and_Norm(spec['flux'][i], spec['ivar'][i], spec['model'][i], input_grid)
        Norm_value = np.median(np.abs(input_spectrum))
        output_spectrum = spec['reconstruction'][i].flatten()
        pipeline_z, best_fit_z, min_chi_square, Degeneracy, subclass = spec['Z'][i],  spec['Best_fit_z'][i], spec['min_chi_square'][i], spec['Degeneracy'][i], spec['SUBCLASS'][i]
        Class, SNR = spec['CLASS'][i], spec['SN_MEDIAN_ALL'][i]
        best_fit_wave_shift = np.log10(1+best_fit_z)


        encoder_model = output_spectrum/Norm_value-5
        axs[i].plot(10**output_grid, encoder_model, linewidth=0.5, alpha=1, c='blue',label='$\hat{F}_i$')
        index0 = np.argmin(np.abs(output_grid-(input_grid[0]-best_fit_wave_shift)))
        wave = 10**(input_grid-best_fit_wave_shift)
        axs[i].set_xlim(wave[0], wave[-1])

        
        kenel = np.array([1,2,3,2,1])/9
        residual =  (input_spectrum-output_spectrum[index0:index0+len(input_grid)])/Norm_value+2
        residual = np.convolve(kenel, residual, mode='same')
        axs[i].plot(wave, residual, linewidth=0.5, alpha=1, c='black', label=f'residual')

        ymin, ymax = -10, 8
        line_name_size = 8
        back_z_shift = np.log10(1+back_z)
        for line, line_num in zip(Emi_line, range(len(Emi_line))):
            loglam = np.log10(line['lambda'])
            loglam = loglam + back_z_shift - best_fit_wave_shift
            lam = 10**loglam
            if lam<wave[0] or lam>wave[-1]:
                continue
            index = np.argmin(np.abs(wave - lam))
            axs[i].vlines(lam, ymin=residual[index]-1, ymax=ymax, color='red', linestyle=':',linewidth=1.0)
            axs[i].text(lam+5, ymax-4, line['name'], color='red', rotation=90, ha='left', va='bottom', alpha=1.0, fontsize=line_name_size)

        subclass = subclass.replace(' ', 'nan')
        axs[i].set_title(f'Id: {id[i]}, {Class}-{subclass}, z={pipeline_z:.4f}, SNR={SNR:.2f}', fontsize=f_size)
        axs[i].set_ylim(ymin, ymax)
        legend = axs[i].legend(loc='upper right', fontsize=f_size, framealpha=0.1)
        axs[i].tick_params(axis='x', labelsize=f_size)
        axs[i].tick_params(axis='y', labelsize=f_size)
        plt.xticks(fontsize=f_size)
        plt.yticks(fontsize=f_size)
        for handle in legend.legendHandles:
            handle.set_linewidth(2.0)

    plt.ylabel('Norm flux', fontsize=f_size)
    plt.xlabel('AA', fontsize=f_size)
    plt.savefig(f'./figures/{model_name}_residual.pdf', bbox_inches='tight', pad_inches=0.1)