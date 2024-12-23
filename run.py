import model_encoder, model_unet
from astropy.table import Table, vstack
import torch
import numpy as np

# one should change the number of spectra
def training_model(data_file, model_name, redshit, model_type, egien_num=10, epochs=100, num_train=20000, dropout=0.0):
    if model_type == 'encoder':
        Model = model_encoder.GaSNet3()
    elif model_type == 'unet':
        Model = model_unet.GaSNet3_UNet()
    Model.cfg['model_name'] = model_name
    Model.cfg['z_min'] = redshit[0]
    Model.cfg['z_max'] = redshit[1]
    Model.cfg['egienV_num'] = egien_num
    Model.cfg['epochs'] = epochs
    Model.cfg['dropout'] = dropout
    Model.device = torch.device('cuda:2')
    spec = Table.read(data_file)
    Model.train_data = spec[0:num_train]
    Model.valid_data = spec[20000:25000]
    Model.Init(load_json=False)
    Model.training_loop()
    Model.test_data = spec[25000:]
    Model.prediction()


train_Encoder = True
train_UNet = True
test_classification = True

model_names = ['Encoder_SDSS_STAR', 'Encoder_SDSS_GALAXY', 'Encoder_SDSS_QSO']
data_files = ['./train_data_3600_9200/STAR.fits', './train_data_3600_9200/GALAXY.fits', './train_data_3600_9200/QSO.fits']
redshift_range = [(-5e-3, 5e-3), (0, 1.5), (0, 5)]

if train_Encoder:
    for model_name, data_name, redshit in zip(model_names, data_files, redshift_range):
        training_model(data_name, model_name, redshit, model_type='encoder')

if train_UNet:
    model_names = ['UNetL1SmoFlip_SDSS_STAR', 'UNetL1SmoFlip_SDSS_GALAXY', 'UNetL1SmoFlip_SDSS_QSO']
    dropouts = [1.0, 0.0, 0.0]
    for model_name, data_name, redshit, dropout in zip(model_names, data_files, redshift_range, dropouts):
        training_model(data_name, model_name, redshit, model_type='unet', dropout=dropout, epochs=100)


if test_classification:
    test_classfy_specta = []
    for fname, zrang in zip(data_files, redshift_range):
        spec = Table.read(fname)
        spec[spec['Z']<=zrang[1]]
        spec = spec[30000:40000]
        test_classfy_specta.append(spec)
    test_classfy_specta = vstack(test_classfy_specta)
    idxs = list(range(len(test_classfy_specta)))
    np.random.seed(42)
    np.random.shuffle(idxs)
    test_classfy_specta = Table(np.array(test_classfy_specta)[idxs])
    # the standard Encoder
    model_names = ['Encoder_SDSS_STAR', 'Encoder_SDSS_GALAXY', 'Encoder_SDSS_QSO']
    for name in model_names:
        Encoder = model_encoder.GaSNet3()
        json_file = f'./models/{name}/{name}.json'
        Encoder.Init(load_json=True, json_name=json_file)
        Encoder.test_data = test_classfy_specta
        Encoder.prediction(fname=f'test_classfy_{name}')

    model_names = ['UNetL1SmoFlip_SDSS_STAR', 'UNetL1SmoFlip_SDSS_GALAXY', 'UNetL1SmoFlip_SDSS_QSO']
    for name in model_names:
        UNet = model_unet.GaSNet3_UNet()
        json_file = f'./models/{name}/{name}.json'
        UNet.Init(load_json=True, json_name=json_file)
        UNet.test_data = test_classfy_specta
        UNet.prediction(fname=f'test_classfy_{name}')