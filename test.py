import os

import paddle
import numpy as np
import cv2
from main import Model
import configparser


conf = configparser.ConfigParser()
conf.read("config.ini", encoding='Utf-8')
model = conf.get('dir', 'model')
train_images_dir = os.path.join(conf.get('dir', 'test_dir'), '/data/')
label_path = os.path.join(conf.get('dir', 'test_dir'), '/label/')
pred_path = conf.get('dir', 'pred_dir')


model_state_dict = paddle.load(os.path.join(model, 'final.pdparams'))
model = Model()
model.set_state_dict(model_state_dict)


def get_iou_vector(label, pred):
    A = np.array(label > 0)
    B = np.array(pred > 0)
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def pred(path, pred_path, model):
    data = np.load(train_images_dir + '\\' + path)

    data = paddle.to_tensor(data).unsqueeze(axis=0)
    data = paddle.transpose(data, perm=[0, 3, 1, 2])
    pred = model(data)
    pred = paddle.transpose(pred, perm=[0, 2, 3, 1])
    pred = np.array(pred[0, :, :, :])
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i, j] >= 0.5:
                pred[i, j] = 1
            else:
                pred[i, j] = 0
    cv2.imwrite(pred_path + '\\' + f'{path[:-4]}.png', pred * 255)

    # return pred


datas = os.listdir(train_images_dir)

for i in range(len(datas)):
    pred(datas[i], pred_path, model)
    print(i)

ious = 0
n = 0

labels = os.listdir(label_path)
preds = os.listdir(pred_path)

for i in range(len(labels)):
    iou = get_iou_vector(cv2.imread(label_path + '\\' + labels[i], 0), cv2.imread(pred_path + '\\' + preds[i], 0))
    ious += iou
    n += 1
    print(f'n:{n} - iou:{iou:.4f}')

print(f'n:{n} - mean_iou:{ious/n:.4f}')
