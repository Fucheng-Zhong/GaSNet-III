import model_encoder, model_unet
from astropy.table import Table, vstack
import torch
import numpy as np

# one should change the number of spectra
def training_model(data_file, model_name, redshit, poly_deg, egien_num, epochs=100, num_train=20000):
    Model = model_encoder.GaSNet3()
    Model.cfg['model_name'] = model_name
    Model.cfg['z_min'] = redshit[0]
    Model.cfg['z_max'] = redshit[1]
    Model.cfg['egienV_num'] = egien_num
    Model.cfg['poly_deg'] = poly_deg
    Model.cfg['epochs'] = epochs
    Model.device = torch.device('cuda:2')
    spec = Table.read(data_file)
    Model.train_data = spec[0:num_train]
    Model.valid_data = spec[20000:25000]
    Model.Init(load_json=False)
    Model.training_loop()
    Model.test_data = spec[25000:]
    Model.prediction()


model_names = ['Encoder_SDSS_STAR', 'Encoder_SDSS_GALAXY', 'Encoder_SDSS_QSO']
data_files = ['./train_data_3600_9200/STAR.fits', './train_data_3600_9200/GALAXY.fits', './train_data_3600_9200/QSO.fits']
redshift_range = [(-5e-3, 5e-3), (0, 1.5), (0, 5)]

EigenNum, poly_deg = 3, 2
for model_name, data_name, redshit in zip(model_names, data_files, redshift_range):
    model_name = model_name + f'_PolyDeg={poly_deg}_EigenNum={EigenNum}'
    training_model(data_name, model_name, redshit, poly_deg=poly_deg, egien_num=EigenNum)

EigenNum, poly_deg = 5, 2
for model_name, data_name, redshit in zip(model_names, data_files, redshift_range):
    model_name = model_name + f'_PolyDeg={poly_deg}_EigenNum={EigenNum}'
    training_model(data_name, model_name, redshit, poly_deg=poly_deg, egien_num=EigenNum)

EigenNum, poly_deg = 10, 2
for model_name, data_name, redshit in zip(model_names, data_files, redshift_range):
    model_name = model_name + f'_PolyDeg={poly_deg}_EigenNum={EigenNum}'
    training_model(data_name, model_name, redshit, poly_deg=poly_deg, egien_num=EigenNum)