import model_encoder
from astropy.table import Table
import torch
import numpy as np
import os, argparse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
wave_min, wave_max = 3600, 9200
delta_loglam = 1e-4
min_loglam, max_loglam = np.round(np.log10(wave_min),4), np.round(np.log10(wave_max),4)
pixel_num = round((max_loglam - min_loglam)/delta_loglam)+1
loglam_grid = np.linspace(min_loglam, max_loglam, pixel_num)


def classify(fname_list, lables):
    chi_square_list, Degeneracy_list, redsift_list, coefficients = [], [], [], []
    coeff_list = []
    for fname in fname_list:
        classify_result = Table.read(fname)
        Degeneracy_list.append(classify_result['Degeneracy'].reshape(-1,1))
        chi_square_list.append(classify_result['min_chi_square'].reshape(-1,1))
        redsift_list.append(classify_result['Best_fit_z'].reshape(-1,1))
        coeff_list.append(classify_result['coefficients'])
    chi_square_list = np.concatenate(chi_square_list, axis=-1)
    Degeneracy_list = np.concatenate(Degeneracy_list, axis=-1)
    redsift_list = np.concatenate(redsift_list, axis=-1)
    coeff_list = np.concatenate(coeff_list, axis=-2)
    pred_index = np.argmin(chi_square_list, axis=-1)
    degeneracy = Degeneracy_list[np.arange(Degeneracy_list.shape[0]), pred_index]
    redshift = redsift_list[np.arange(redsift_list.shape[0]), pred_index]
    min_chi_square = chi_square_list[np.arange(chi_square_list.shape[0]), pred_index]
    coefficients = coeff_list[np.arange(coeff_list.shape[0]), pred_index, :]
    pred_label = np.array(lables[pred_index])
    data = {'Best-Fit_CLASS':pred_label, 'Best-Fit_Z':redshift, 'min_chi_square':min_chi_square,  'Degeneracy':degeneracy, 'coefficients':coefficients}
    data = Table(data)
    data.write('./results/output.fits', format='fits', overwrite=True)
    return data


def run_model(test_specta):
    # the standard Encoder
    lables = np.array(['STAR','GALAXY', 'QSO'])
    model_names = ['Encoder_SDSS_STAR', 'Encoder_SDSS_GALAXY', 'Encoder_SDSS_QSO']
    for name in model_names:
        Encoder = model_encoder.GaSNet3()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Encoder.device = device
        json_file = f'./models/{name}/{name}.json'
        Encoder.Init(load_json=True, json_name=json_file)
        Encoder.test_data = test_specta
        Encoder.prediction(fname=f'{name}_output')
    fname_list = [f'results/{name}_output.fits' for name in model_names]
    data = classify(fname_list, lables)
    print(data)


def read_spec(file, survey='sdss'):
    """
    Read 1D spectrum from a FITS file for SDSS or LAMOST surveys.

    Parameters:
    ----------
    file : str
        Path to the FITS file containing the spectrum.
    survey : str, optional
        Survey name. Either 'sdss' or 'lamost'. Default is 'sdss'.

    Returns:
    -------
    loglam : np.ndarray
        Logarithm (base 10) of the wavelength array.
    flux : np.ndarray
        Flux array of the spectrum.
    ivar : np.ndarray
        Inverse variance (1 / sigma^2) of the flux array.
    """
    info_dic = {}
    # ========= 可以重写这部分
    if survey == 'sdss':
        hudl1 = Table.read(file,1)
        if 'LOGLAM' in hudl1.keys():
            loglam, flux, ivar = hudl1['LOGLAM'], hudl1['FLUX'], hudl1['IVAR']
        else:
            loglam, flux, ivar = hudl1['loglam'], hudl1['flux'], hudl1['ivar']
    elif survey == 'lamost':
        hudl1 = Table.read(file, 1)
        loglam, flux, ivar = np.log10(hudl1['WAVELENGTH'][0]), hudl1['FLUX'][0], hudl1['IVAR'][0]
    # =========
    info_dic['flux'], info_dic['ivar'], info_dic['loglam'] = flux, ivar, loglam
    print(f'flux shape: {flux.shape}, ivar shape: {ivar.shape}, loglam shape: {loglam.shape}')
    info_dic = Table([info_dic])
    # interp the spectra
    print('flux shape=', flux.shape, 'loglam shape=', loglam.shape)
    interption = lambda x: np.interp(loglam_grid, loglam, x)
    interp_flux = np.apply_along_axis(interption, axis=-1, arr=info_dic['flux'])
    interp_ivar = np.apply_along_axis(interption, axis=-1, arr=info_dic['ivar'])
    info_dic['flux'], info_dic['ivar'] = interp_flux, interp_ivar
    info_dic['ivar'][..., loglam_grid>loglam[-1]] = 0 #超出范围的ivar置为0
    info_dic['ivar'][..., loglam_grid<loglam[0]] = 0
    return info_dic


def main():
    # Get arguments from the command line
    parser = argparse.ArgumentParser(description="L1 coarse classifier")
    parser.add_argument("--fits", default="./spec/spec-0391-51782-0088.fits", help="fits file")
    parser.add_argument("--survey", default="sdss", help="survey, 'sdss' or 'lamost'. Default is 'sdss'.")
    args = parser.parse_args()
    spec = read_spec(args.fits, args.survey)
    result = run_model(spec)
    return result


if __name__ == "__main__":
    main()