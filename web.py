import os
import shutil

import cv2
import gradio as gr
import pandas as pd
from osgeo import gdal, osr
from pyresample import create_area_def
from satpy import Scene
from skimage import morphology
import webbrowser

from main import Model
import paddle
import numpy as np
import tempfile


area_extent = [72.82, 24.73, 104.82, 40.73]
res = 0.05
height = int((area_extent[3] - area_extent[1]) / res)
width = int((area_extent[2] - area_extent[0]) / res)
dw = 100
dh = 100
name = ''
n = 0

model_state_dict = paddle.load(r'F:\Pycharm_Projects\paddlepaddle\models\128_\final.pdparams')
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
    try:
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
    except Exception as e:
        gr.Error(f'error:{e}')


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


def fy4a_hdf_ci(inp):
    try:
        global name, temp, model
        name = os.path.basename(inp.name).split('.')[0]
        hdf_npy(inp)

        data = np.load(os.path.join(temp, name + '.npy'))
        split(data)
        save_pred()
        output = fy(data)
        data = data.transpose(1, 2, 0)


        yun = np.repeat((data[:, :, 11:12] - 105), repeats=3, axis=2)
        for i in range(yun.shape[0]):
            for j in range(yun.shape[1]):
                if output[i, j] == 255:
                    yun[i, j, 0] = 255
                    yun[i, j, 2] = 0
                    yun[i, j, 1] = 0
        # cv2.imwrite(r'F:\Pycharm_Projects\paddlepaddle\4a\yun\test.png', output)
        return yun / 255
    except Exception as e:
        gr.Error(f'error:{e}')


def update_disposition(*inp):
    global area_extent, res, height, width
    if 0 < inp[0] < inp[2] and 0 < inp[1] < inp[3]:
        for i in range(4):
            area_extent[i] = inp[i]
    else:
        gr.Warning('Please enter the correct latitude and longitude range.')
    if inp[-1] > 0:
        res = inp[-1]
    else:
        gr.Warning('The resolution must be greater than zero')

    height = int((area_extent[3] - area_extent[1]) / res)
    width = int((area_extent[2] - area_extent[0]) / res)

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
                            input_min_longitude = gr.Number(value=72.82, label='最小经度')
                            input_max_longitude = gr.Number(value=104.82, label='最大经度')
                            input_min_latitude = gr.Number(value=24.73, label='最小伟度')
                            input_max_latitude = gr.Number(value=40.73, label='最大伟度')
                        input_res = gr.Number(value=0.05, label='分辨率')
                        button1 = gr.Button(value='保存配置', variant='primary')
                        button1.click(fn=update_disposition, inputs=[input_min_longitude, input_min_latitude,
                                                                     input_max_longitude, input_max_latitude,
                                                                     input_res])

                    gr.Examples([r'F:\Pycharm_Projects\paddlepaddle\4a\FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20220624114500_20220624115959_4000M_V0001.HDF',
                                 r'F:\Pycharm_Projects\paddlepaddle\4a\FY4A-_AGRI--_N_DISK_1047E_L1-_FDI-_MULT_NOM_20220630210000_20220630211459_4000M_V0001.HDF',
                                 r'F:\Pycharm_Projects\paddlepaddle\4b\FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_20230215170000_20230215171459_4000M_V0001.HDF',
                                 r'F:\Pycharm_Projects\paddlepaddle\4b\FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_20230215180000_20230215181459_4000M_V0001.HDF'],
                                inputs=input_hdf)
                    gr.Text('这里提供一个风云卫星遥感数据服务网的账号，仅供学习使用，可点击下方按钮跳转至风云卫星遥感数据服务网\n'
                            '账号：287141243@qq.com\n'
                            '密码：N210zhangyonghong', interactive=False, label='')
                    url = gr.Button(value='风云卫星数据遥感服务网')
                    url.click(fn=open_url)

                with gr.Column():
                    output_label = gr.Image(value='numpy', format='png', height=height, width=width, image_mode='RGB', label='CI_predict')
            button2 = gr.Button(value='获取CI预测结果', variant='primary')
            button2.click(fn=fy4a_hdf_ci, inputs=input_hdf, outputs=output_label)

        demo.launch(share=True)


if __name__ == '__main__':
    main()

