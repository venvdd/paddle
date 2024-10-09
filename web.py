import re
from datetime import datetime

from satpy.scene import Scene
import os
import shutil

import cv2
import gradio as gr
import pandas as pd
from osgeo import gdal, osr
from pyresample import create_area_def
from skimage import morphology
import webbrowser

from main import Model
import paddle
import numpy as np
import tempfile
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用于设置图的中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像时负号“-”显示为方块的问题


conf = configparser.ConfigParser()
conf.read("config.ini", encoding='Utf-8')
# area_extent = [70, 10, 145, 60]
area_extent = [float(x) for x in conf.get('load', 'area_extent').split(',')]  # 最小经度、最小纬度、最大经度、最大纬度
model = conf.get('dir', 'model')
# area_extent = [72.82, 24.73, 104.82, 40.73]
# area_extent = [73, 26, 97, 36]
# 70, 145, 10, 60.1
res = 0.05
height = int((area_extent[3] - area_extent[1]) / res)
width = int((area_extent[2] - area_extent[0]) / res)
dw = 100
dh = 100
name = ''
n = 0
rgb = False
nationwide = True

model_state_dict = paddle.load(os.path.join(model, 'final.pdparams'))
model = Model()
model.set_state_dict(model_state_dict)


def writeTiff(im_data, im_geotrans):
    global temp, name
    im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(os.path.join(temp, name + '.tif'), im_width, im_height, im_bands, gdal.GDT_Float32, options=["INTERLEAVE=BAND"])

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


def tif_npy():
    global temp, name
    tif = gdal.Open(os.path.join(temp, name + '.tif'))
    tif_array = tif.ReadAsArray()
    npy = np.array(tif_array)
    np.save(os.path.join(temp, name + '.npy'), npy)


def hdf_npy(inp):
    global temp, name, area_extent, res

    shutil.copy(inp.name, temp)
    if 'FY4A' in name:
        reader = 'agri_fy4a_l1'
    elif 'FY4B' in name:
        reader = 'agri_fy4b_l1'
    else:
        gr.Warning('Please upload the FY4A or FY4B imager full disk 4KML1 data file. It can be obtained from Fengyun Satellite remote sensing Data Service network.')
        return

    scn = Scene([inp.name], reader=reader)
    channels = scn.available_dataset_names()
    scn.load(channels)
    areadef = create_area_def('china',
                              '+proj=eqc +pm=180',
                              area_extent=area_extent,
                              units='degrees',
                              resolution=res
                              )
    china_scene = scn.resample(areadef)
    lonlat = china_scene[channels[-1]].attrs['area'].get_lonlats()
    lon_min = lonlat[0].min()
    lat_max = lonlat[1].max()
    im_geotrans = (lon_min, res, 0.0, lat_max, 0.0, -res)
    im_data = []
    for i in range(len(channels)):
        data = china_scene[channels[i]].values
        im_data.append(data)
    im_data = np.array(im_data)

    if np.isnan(im_data).any():
        gr.Warning('Data has missing values')
    else:
        writeTiff(im_data, im_geotrans)
        tif_npy()



def _split(data, i, j):
    global temp, name, n
    a = data[6:14, i:i+128, j:j+128]
    np.save(os.path.join(temp, name + f'_{n}.npy'), a)


def split(data):
    global dh, dw, n
    n = 0
    i = 0
    j = 0
    di = dh  # 步长
    dj = dw
    flag_i = False
    flag = True
    if 'FY4B' in name:
        data = np.delete(data, [10], axis=0)
    while flag:
        if i + 128 > data.shape[1]:
            i = data.shape[1] - 128
            flag_i = True

        if j + 128 > data.shape[2]:
            if flag_i:
                flag = False
            j = data.shape[2] - 128
            n += 1
            _split(data, i, j)
            i += di
            j = 0
        else:
            n += 1
            _split(data, i, j)
            j += dj


def save_pred():
    global temp, name, n, model
    for z in range(1, n+1):
        data = np.load(os.path.join(temp, name + f'_{z}.npy'))
        if 'FY4B' in name:
            data[:, :, 2] += 1.7
            data[:, :, 3] += 5.5
            data[:, :, 4] += 1.
            data[:, :, 5] += 1.2
            data[:, :, 6] += 1.5
            data[:, :, 7] -= 5.0
        data = (data - 105) / 255.0
        data = np.array(data, dtype='float32')
        data = paddle.to_tensor(data).unsqueeze(axis=0)
        pred = model(data)
        pred = paddle.transpose(pred, perm=[0, 2, 3, 1])
        pred = np.array(pred[0, :, :, :])
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] >= 0.5:
                    pred[i, j] = 1
                else:
                    pred[i, j] = 0
        cv2.imwrite(os.path.join(temp, name + f'_{z}.png'), pred * 255)


def _fy(label, pred, i, j):
    label[i:i + 128, j:j + 128] = pred / 255
    return label


def fy(data):
    global temp, name, n
    shape = data.shape
    h, w = shape[1], shape[2]
    label = np.zeros((h, w))
    i = 0
    j = 0
    di = dh  # 步长
    dj = dw
    for z in range(1, n+1):
        if i + 128 >= h:
            i = h - 128
        if j + 128 >= w:
            j = w - 128
            pred = cv2.imread(os.path.join(temp, name + f'_{z}.png'), 0)
            label = _fy(label, pred, i, j)
            i += di
            j = 0
        else:
            pred = cv2.imread(os.path.join(temp, name + f'_{z}.png'), 0)
            label = _fy(label, pred, i, j)
            j += dj

    label[label >= 1] = 1
    output = morphology.remove_small_objects(label > 0.5, min_size=10)
    return output * 255


def map_plt():
    global temp, rgb

    extent = [70, 145, 10, 60.1]
    proj = ccrs.PlateCarree()
    data1 = cv2.imread(os.path.join(temp, 'yun.png')) / 255.0
    data2 = cv2.imread(os.path.join(temp, 'label.png'), cv2.IMREAD_GRAYSCALE) / 255.0

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent, crs=proj)
    lon = np.linspace(extent[0], extent[1] + 0.01, data1.shape[1])
    lat = np.linspace(extent[2], extent[3] + 0.01, data1.shape[0])[::-1]
    xtick = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140]
    ytick = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    ax.set_xticks(xtick, crs=proj)
    ax.set_yticks(ytick, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ax.tick_params(labelsize=14)
    ax.gridlines(color='gray', linestyle='dotted', xlocs=xtick, ylocs=ytick)
    # 添加shp文件
    ## 添加国界线
    shp = Reader(os.path.join("/home/aistudio/dataset/shp/china.shp"))
    ## 设置国界线的颜色/粗细
    # self.ax.add_geometries(shp4.geometries(), crs=self.proj, edgecolor='red', linewidth=1.5, facecolor='None')
    ax.add_geometries(shp.geometries(), crs=proj, edgecolor='gray', linewidth=1, facecolor='None')

    cmap1 = 'gray'
    norm1 = colors.Normalize(-0.1, 1)
    ax.pcolorfast(lon, lat, data1, cmap=cmap1, norm=norm1, transform=proj)

    data2 = np.ma.masked_less_equal(data2, 0)
    cmap2 = colors.ListedColormap(['white', 'blue'])
    bounds2 = [0, 1]
    norm2 = colors.BoundaryNorm(bounds2, cmap2.N)
    ax.pcolorfast(lon, lat, data2, cmap=cmap2, norm=norm2, transform=proj)
    plt.savefig(os.path.join(temp, 'result.png'), transparent=True)


def get_rgb():
    global name, rgb
    match = re.search(r'(\d{8})(\d{6})_(\d{8})(\d{6})', name)
    if match:
        start_time_str = match.group(1) + match.group(2)  # 开始时间字符串
        end_time_str = match.group(3) + match.group(4)  # 结束时间字符串

        start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
        end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")


        # 判断开始时间和结束时间是否在2点到8点之间
        def is_between_2_and_8(time):
            return 2 <= time.hour < 8


        start_within_range = is_between_2_and_8(start_time)
        end_within_range = is_between_2_and_8(end_time)

        if start_within_range or end_within_range:
            rgb = True
        else:
            rgb = False


def fy4a_hdf_ci(inp):

    global name, temp, model, rgb, nationwide
    name = os.path.basename(inp.name).split('.')[0]
    hdf_npy(inp)

    data = np.load(os.path.join(temp, name + '.npy'))
    split(data)
    save_pred()
    output = fy(data)
    data = data.transpose(1, 2, 0)
    get_rgb()
    if rgb:
        yun = data[:, :, 0:3]
    else:
        yun = 255 - (data[:, :, 11:12] - 105)
    if nationwide:
        cv2.imwrite(os.path.join(temp, 'label.png'), output)
        cv2.imwrite(os.path.join(temp, 'yun.png'), yun)
        map_plt()
        result = cv2.imread(os.path.join(temp, 'result.png'))
        if rgb:
            return result, yun / 255, np.repeat(output[:, :, np.newaxis], 3, axis=2) / 255
        else:
            return result, np.repeat(yun, 3, axis=2) / 255, np.repeat(output[:, :, np.newaxis], 3, axis=2) / 255
    else:
        copy_yun = np.copy(yun)
        for i in range(yun.shape[0]):
            for j in range(yun.shape[1]):
                if output[i, j] == 255:
                    yun[i, j, 0] = 255
                    yun[i, j, 2] = 0
                    yun[i, j, 1] = 0
        if rgb:
            return yun / 255, copy_yun / 255, np.repeat(output[:, :, np.newaxis], 3, axis=2) / 255
        else:
            return yun / 255, np.repeat(copy_yun, 3, axis=2) / 255, np.repeat(output[:, :, np.newaxis], 3, axis=2) / 255



def update_disposition(*inp):
    global area_extent, res, height, width, nationwide
    for i in range(4):
        area_extent[i] = inp[i]
    if inp[-1] > 0:
        res = inp[-1]
    else:
        gr.Warning('The resolution must be greater than zero')
    if area_extent == [70, 10, 145, 56]:
        nationwide = True
    else:
        nationwide = False
    height = int((area_extent[3] - area_extent[1]) / res)
    width = int((area_extent[2] - area_extent[0]) / res)
    gr.Info('Successfully change configuration')

def open_url():
    webbrowser.open("https://satellite.nsmc.org.cn/portalsite/default.aspx")

def main():
    global temp, area_extent, res, height, widght

    with tempfile.TemporaryDirectory(dir='./') as temp:
        with gr.Blocks() as demo:
            gr.Markdown('预测对流初生')
            with gr.Row():
                with gr.Column():
                    input_hdf = gr.File(file_types=['.HDF'], label='Upload HDF File', height=25)
                    with gr.Accordion(open=False, label='配置信息'):
                        with gr.Row():
                            # 70, 145, 10, 60.1
                            input_min_longitude = gr.Number(value=area_extent[0], label='最小经度')
                            input_max_longitude = gr.Number(value=area_extent[2], label='最大经度')
                            input_min_latitude = gr.Number(value=area_extent[1], label='最小伟度')
                            input_max_latitude = gr.Number(value=area_extent[3], label='最大伟度')
                        input_res = gr.Number(value=0.05, label='分辨率')
                        button1 = gr.Button(value='保存配置', variant='primary')
                        button1.click(fn=update_disposition, inputs=[input_min_longitude, input_min_latitude,
                                                                     input_max_longitude, input_max_latitude,
                                                                     input_res])

                    gr.Examples([],
                                inputs=input_hdf)
                    gr.Text('注意！当选择国际标准时间2点到10点(即北京时间10点到18点)，输出的云图会替换成RGB可见光图\n'
                            '这里提供一个风云卫星遥感数据服务网的账号，仅供学习使用，可点击下方按钮跳转至风云卫星遥感数据服务网\n'
                            '账号：287141243@qq.com\n'
                            '密码：N210zhangyonghong', interactive=False, label='')
                    url = gr.Button(value='风云卫星数据遥感服务网')
                    url.click(fn=open_url)
                    button2 = gr.Button(value='获取CI预测结果', variant='primary')
                with gr.Column():
                    with gr.Accordion(open=True, label='CI_predict'):
                        output_label = gr.Image(value='numpy', format='png', image_mode='RGB', label='CI_predict')
                    with gr.Accordion(open=True, label='nephogram'):
                        output_label1 = gr.Image(value='numpy', format='png', image_mode='RGB', label='nephogram')
                    with gr.Accordion(open=True, label='CI'):
                        output_label2 = gr.Image(value='numpy', format='png', image_mode='L', label='CI')

            button2.click(fn=fy4a_hdf_ci, inputs=input_hdf, outputs=[output_label, output_label1, output_label2])

        demo.launch()


if __name__ == '__main__':
    main()





