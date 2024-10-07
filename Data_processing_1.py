from satpy.scene import Scene
import warnings

warnings.filterwarnings('ignore')
import os, glob
import numpy as np
import sys
from osgeo import gdal, osr
from pyresample import create_area_def
import pandas as pd
from fypy.fy4 import FY4Scene
import tempfile
import configparser
import cv2 as cv


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def writeTiff(im_data, im_geotrans, out_filename, savepath, temp):
    im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(os.path.join(temp, out_filename), im_width, im_height, im_bands, gdal.GDT_Float32, options=["INTERLEAVE=BAND"])

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    # 创建投影对象
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # 设置为WGS84经纬度坐标系
    # 设置投影信息
    dataset.SetProjection(srs.ExportToWkt())

    for i in range(im_bands):
        if np.isnan(im_data[i]).any():
            df = pd.DataFrame(im_data[i])
            new_df = df.interpolate(method='polynomial', order=2)
            im_data[i] = new_df.values
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

    tif = gdal.Open(os.path.join(temp, out_filename))
    tif_array = tif.ReadAsArray()
    npy = np.array(tif_array)
    np.save(savepath, npy)


def FY4_AGRI(filepath, out_filename, savepath, area_extent, res):
    # readers = available_readers()
    # 创建scene对象
    if area_extent is None:
        area_extent = [72.82, 24.73, 104.82, 40.73]
    if 'FY4A' in os.path.basename(filepath):
        reader = 'agri_fy4a_l1'
    else:
        reader = 'agri_fy4b_l1'
    scn = Scene([filepath], reader=reader)
    # 查看可用的通道
    channels = scn.available_dataset_names()
    # 加载指定通道
    scn.load(channels)
    # 创建投影区域
    # area_extent = [66.5, 17.8, 138.5, 53.8] # 240的倍数    # 中国范围72, 2.9, 136, 54.1
    # area_extent = [72.2, 3, 136.2, 54.2]   # 1024*1280：256的倍数  72.2, 3, 136.2, 54.2
    areadef = create_area_def('china',
                              '+proj=eqc +pm=180',
                              area_extent=area_extent,
                              units='degrees',
                              resolution=res
                              )
    # 对投影区域重采样
    china_scene = scn.resample(areadef)

    # 获取仿射矩阵信息
    lonlat = china_scene[channels[-1]].attrs['area'].get_lonlats()
    lon_min = lonlat[0].min()
    lat_max = lonlat[1].max()
    im_geotrans = (lon_min, 0.05, 0.0, lat_max, 0.0, -0.05)

    im_data = []
    for i in range(len(channels)):
        data = china_scene[channels[i]].values
        im_data.append(data)
    im_data = np.array(im_data)

    if np.isnan(im_data).any():
        pass
    else:
        with tempfile.TemporaryDirectory(dir='./') as temp:
            writeTiff(im_data, im_geotrans, out_filename, savepath, temp)
            print(os.path.basename(filepath) + ' 数据处理完成！')
            sys.stdout.flush()


def FY4_AGRI_Product(filepath, savepath, prodid, area_extent, temp, resolution=0.05):
    scene = FY4Scene(filename=filepath)
    ds = scene.load(filepath, ProdID=prodid)
    ds = scene.clip(ds, extent=area_extent, resolution=resolution)
    scene.ds2tiff(os.path.join(temp, os.path.basename(filepath)[:-3] + '.tif'), srcDS=ds)

    a = cv.imread(os.path.join(temp, os.path.basename(filepath)[:-3] + '.tif'), cv.IMREAD_UNCHANGED)
    if a is None:
        print(os.path.join(temp, os.path.basename(filepath)[:-3] + '.tif'), '读取失败')
        return
    for j in range(a.shape[0]):
        for z in range(a.shape[1]):
            if a[j, z] == -1:
                a[j, z] = 255
        a = a.transpose(1, 2, 0)
        cv.imwrite(os.path.join(savepath, os.path.basename(filepath)[:-3] + '.png'), a)


def main(input_folder, output_folder, area_extent, res, nc_input_folder, nc_output_folder, prodid):
    os.makedirs(output_folder, exist_ok=True)
    input_hdfs = glob.glob(os.path.join(input_folder, "*FY4*_*.HDF"))
    input_hdfs.sort()
    for count, input_hdf in enumerate(input_hdfs):
        try:
            filename = os.path.basename(input_hdf)
            out_filename = filename.split('_')[0] + filename.split('_')[9] + '_' + filename.split('_')[10] + '.tif' # 风云校正后数据裁剪
            output_tif = os.path.join(output_folder, out_filename)
            FY4_AGRI(input_hdf, out_filename, output_tif[:-4] + '.npy', area_extent, res)
        except Exception as e:
            print(e)
            continue

    print('HDF数据处理完毕')
    sys.stdout.flush()
    flist = glob.glob(os.path.join(nc_input_folder, "*.NC"))
    with tempfile.TemporaryDirectory(dir='./') as temp:
        for fp in flist:
            FY4_AGRI_Product(fp, nc_output_folder, prodid, area_extent, temp, res)
            print(os.path.basename(fp) + '处理完成！')


def load():
    conf = configparser.ConfigParser()
    conf.read("config.ini", encoding='Utf-8')
    input_folder = conf.get('dir', 'hdf')
    output_folder = conf.get('dir', 'npy_out')
    area_extent = [float(x) for x in conf.get('load', 'area_extent').split(',')]  # 最小经度、最小纬度、最大经度、最大纬度
    res = float(conf.get('load', 'res'))
    nc_input_folder = conf.get('dir', 'nc')
    nc_output_folder = conf.get('dir', 'label_out')
    prodid = 'Convective_Initiation'
    main(input_folder, output_folder, area_extent, res, nc_input_folder, nc_output_folder, prodid)


if __name__ == '__main__':
    load()
    

