# GaSNet-III
Code of GaSNet-III

### docker 安装
docker build -t gasnet3-app .

### docker 运行， 光谱fits作为输入参数 一般是sdss的
docker run --rm gasnet3-app python main.py --fits ./spec/spec-0436-51883-0633.fits

### 或者重写main.py里的以下函数以适配别的文件
def read_spec(file):
    info_dic = {}
    # ========= 可以重写这部分
    hudl1 = Table.read(file,1)
    if 'LOGLAM' in hudl1.keys():
        loglam, flux, ivar = hudl1['LOGLAM'], hudl1['FLUX'], hudl1['IVAR']
    else:
        loglam, flux, ivar = hudl1['loglam'], hudl1['flux'], hudl1['ivar']

## If this code helps your research, please cite https://arxiv.org/abs/2412.21130