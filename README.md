# GaSNet-III

Codebase for **GaSNet-III**, a deep learning model for spectral classification and redshift estimation.

---

## 🐳 Docker Installation

```bash
docker build -t gasnet3-app .
```

---

## 📥 Input

### 🧪 Run with Docker

Make sure Docker is running. The input should be an SDSS `.fits` file:

```bash
docker run --rm gasnet3-app python main.py --fits ./spec/spec-0436-51883-0633.fits
```

---

### 🛠️ Customize Input Parsing

Edit the following function in `main.py` to provide:

- Flux array  
- Logarithmic wavelength  
- Inverse variance  

The code will automatically crop the spectrum to the wavelength range $3600$–$9200\ \text{Å}$, and interpolate it to a fixed grid with step size $\delta \log_{10}(\lambda) = 10^{-4}$.

```python
def read_spec(file):
    info_dic = {}
    # ======== Modify this section as needed
    hudl1 = Table.read(file, 1)
    if 'LOGLAM' in hudl1.keys():
        loglam, flux, ivar = hudl1['LOGLAM'], hudl1['FLUX'], hudl1['IVAR']
    else:
        loglam, flux, ivar = hudl1['loglam'], hudl1['flux'], hudl1['ivar']
```

---

> 📌 Tip: The interpolation and normalization will be handled automatically after this function. You just need to ensure `loglam`, `flux`, and `ivar` are correctly returned.


## 📤 Output

Four FITS files will be created and saved in the **`results`** folder:

### 1. `Encoder_SDSS_GALAXY_output.fits`  
### 2. `Encoder_SDSS_STAR_output.fits`  
### 3. `Encoder_SDSS_QSO_output.fits`

These three files record the reconstruction information from each individual encoder model, trained on the corresponding SDSS spectral class (GALAXY, STAR, or QSO). The columns of these files are as follows:

- **Best_fit_z**: The best-fit redshift based on reconstruction using the GALAXY/STAR/QSO components.
- **min_chi_square**: The minimum $\chi^2$ value across different redshift values.
- **Degeneracy**: Defined as $(\chi^2_{\text{second-min}} / \chi^2_{\text{min}} - 1)$, it evaluates the significance of the best-fit prediction.
- **coefficients**: A 10-dimensional vector used to reconstruct the rest-frame spectrum from 10 rest-frame components learned by the deep learning model.

---

### 4. `output.fits`

This file contains the final results, including the best-fit class, redshift, and the minimal $\chi^2$ across all three encoder models.

- **Best-Fit_CLASS**: One of `STAR`, `GALAXY`, or `AGN`.
- **Best-Fit_Z**: The redshift corresponding to the lowest $\chi^2$ among the three encoder models.
- **min_chi_square**: The lowest $\chi^2$ value across the outputs of `Encoder_SDSS_STAR`, `Encoder_SDSS_GALAXY`, and `Encoder_SDSS_QSO`.
- **Degeneracy**: The degeneracy value corresponding to the best-fit model.
- **coefficients**: Same as described above; used for reconstructing the rest-frame spectrum.




---

## 📖 Citation

If you find this code helpful in your research, please citing the following paper:

> **GaSNet-III**  
> [https://arxiv.org/abs/2412.21130](https://arxiv.org/abs/2412.21130)

```
@ARTICLE{2024arXiv241221130Z,
       author = {{Zhong}, Fucheng et al.},
        title = "{Galaxy Spectra Networks (GaSNet). III. Generative pre-trained network for spectrum reconstruction, redshift estimate and anomaly detection}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = dec,
          eid = {arXiv:2412.21130},
        pages = {arXiv:2412.21130},
          doi = {10.48550/arXiv.2412.21130},
archivePrefix = {arXiv},
       eprint = {2412.21130},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241221130Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
