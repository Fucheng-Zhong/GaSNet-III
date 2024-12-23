from astropy.table import Table,vstack,Column
import numpy as np

wave_min, wave_max = 3600, 9200
delta_loglam = 1e-4
min_loglam, max_loglam = np.round(np.log10(wave_min),4), np.round(np.log10(wave_max),4)
pixel_num = round((max_loglam - min_loglam)/delta_loglam)+1
loglam_grid = np.linspace(min_loglam, max_loglam, pixel_num)
print('loglam grid:', loglam_grid)
print('pixel num:', pixel_num, len(loglam_grid))


# Use padding extrapolation
def extrapolation_pad(x, y, mode='constant'):
    delta = 1e-2*delta_loglam
    # left side
    left_pad_num = sum(loglam_grid<(x[0]-delta))
    y = np.pad(y, (left_pad_num, 0), mode=mode) #===== padding 0
    # right side 
    right_pad_num = sum(loglam_grid>(x[-1:]+delta)) 
    y = np.pad(y, (0, right_pad_num), mode=mode) #===== padding 0
    return y

def Get_One_Spectrum(file):
    info_dic = {}
    hudl1 = Table.read(file,1)
    hudl2 = Table.read(file,2)

    info_dic['CLASS'] = hudl2['CLASS'][0]
    info_dic['SUBCLASS'] = hudl2['SUBCLASS'][0]
    info_dic['PLATE'] = hudl2['PLATE'][0]
    info_dic['MJD'] = hudl2['MJD'][0]
    info_dic['FIBERID'] = hudl2['FIBERID'][0]
    info_dic['Z'] = hudl2['Z'][0]
    info_dic['Z_ERR'] = hudl2['Z_ERR'][0]
    info_dic['SN_MEDIAN_ALL'] = hudl2['SN_MEDIAN_ALL'][0]
    info_dic['SPECTROFLUX'] = hudl2['SPECTROFLUX'][0][:] #MAGNITUDE

    if 'LOGLAM' in hudl1.keys():
        unmask_index = np.where(hudl1['AND_MASK'] == 0)[0]
        loglam, flux, ivar, model = hudl1['LOGLAM'], hudl1['FLUX'], hudl1['IVAR'], hudl1['MODEL']
    else:
        unmask_index = np.where(hudl1['and_mask'] == 0)[0]
        loglam, flux, ivar, model = hudl1['loglam'], hudl1['flux'], hudl1['ivar'], hudl1['model']

    info_dic['good_pixel_num'] = len(loglam[unmask_index])
    index = np.where((loglam >= min_loglam) & (loglam <= max_loglam))
    cut_flux, cut_ivar, cut_model, cut_loglam  = flux[index], ivar[index], model[index], loglam[index]

    #===== Use quadratic polynomial extrapolation  ! ! ! or # padding
    if len(cut_flux) < pixel_num:
        cut_flux = extrapolation_pad(cut_loglam, cut_flux)
        cut_model = extrapolation_pad(cut_loglam, cut_model)
        cut_ivar = extrapolation_pad(cut_loglam, cut_ivar) 
        print('###===extrapolation===###')
    info_dic['flux'], info_dic['ivar'], info_dic['model'] = cut_flux.astype(np.float32), cut_ivar.astype(np.float32), cut_model.astype(np.float32)
    print('shape:',info_dic['flux'].shape, info_dic['ivar'].shape, info_dic['model'].shape)
    if (len(info_dic['flux']) != pixel_num) or (len(info_dic['ivar']) != pixel_num) or (len(info_dic['model']) != pixel_num):
        print('#===dimension error')
        return 0

    info_dic = Table([info_dic])
    return info_dic, loglam, flux, ivar