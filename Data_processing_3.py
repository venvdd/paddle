import numpy as np
import cv2 as cv
import configparser
import os


conf = configparser.ConfigParser()
conf.read("config.ini", encoding='Utf-8')
npy_dir = conf.get('dir', 'hdf_out')
ci_dir = conf.get('dir', 'ci_gpm')
data_dir = conf.get('dir', 'data_dir')
label_dir = conf.get('dir', 'label_dir')

def get_percent(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    # 计算各种元素的占比
    proportions = counts / len(arr)**2
    # 输出结果
    if len(unique_elements) == 1:
        return 0
    else:
        for element, proportion in zip(unique_elements, proportions):
            if element == 255:
                return float(f'{proportion*100:.2f}')


def fg(arr):
    arr1 = arr[0:128, 0:128]
    arr2 = arr[128:, 0:128]
    arr3 = arr[0:128, 128:]
    arr4 = arr[128:, 128:]
    return arr1, arr2, arr3, arr4


def _fg(label, npy):
    flabel = [x for x in fg(label)]
    fdata = [x for x in fg(npy)]
    for j in range(len(flabel)):
        if get_percent(flabel[j]) >= 0.5:
            cv.imwrite(os.path.join(label_dir, cis[i][:-4] + f'_{j + 1}.png'), flabel[j])
            np.save(os.path.join(label_dir, npys[i][:-4] + f'_{j + 1}.npy'), fdata[j])
        else:
            pass

npys = os.listdir(npy_dir)
cis = os.listdir(ci_dir)

for i in range(len(npys)):
    npy = np.load(os.path.join(npy_dir, npys[i]))
    ci = cv.imread(os.path.join(ci_dir, cis[i]), 0)

    height, width = npy.shape[:2]

    start_x1 = (width - 256) // 2
    start_y1 = (height - 256) // 2
    start_x2 = start_x1 + 256
    start_y2 = start_y1 + 256

    # 切片图片
    npy1 = npy[start_y1:start_y2, start_x1:start_x2]
    npy2 = npy[start_y1:start_y2, start_x1+256:start_x2+256]
    label1 = ci[start_y1:start_y2, start_x1:start_x2]
    label2 = ci[start_y1:start_y2, start_x1 + 256:start_x2 + 256]
    _fg(label1, npy1)
    _fg(label2, npy2)
    
