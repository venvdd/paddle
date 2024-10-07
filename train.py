import os

import cv2
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from paddle.metric import Metric
from paddle.nn.functional import interpolate
from sklearn.model_selection import train_test_split
import configparser

import warnings
warnings.filterwarnings("ignore")


class BatchActivate(nn.Layer):
    def __init__(self, num):
        super().__init__()
        self.ba = nn.Sequential(
            nn.BatchNorm2D(num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ba(x)
        return x


class ConvolutionBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', activation=True):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate = BatchActivate(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activate(x)
        return x


class Basic(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cb = nn.Sequential(
            ConvolutionBlock(in_channels, out_channels, 3, 1),
            ConvolutionBlock(in_channels, out_channels, 3, 1, activation=False),
            nn.BatchNorm2D(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.cb(x)
        x = paddle.add(x1, x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Layer):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.x = nn.Sequential(
            ConvolutionBlock(in_channels, num_filters, 1),
            ConvolutionBlock(num_filters, 64, 3),
            ConvolutionBlock(64, 256, 1, activation=False),
        )
        self.y = nn.Sequential(
            ConvolutionBlock(in_channels, 256, 1),
            nn.BatchNorm2D(256)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.x(x)
        x2 = self.y(x)
        x = paddle.add(x1, x2)
        x = self.relu(x)
        return x


class Bottle(nn.Layer):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.x = nn.Sequential(
            ConvolutionBlock(in_channels, num_filters, 1),
            ConvolutionBlock(num_filters, num_filters, 3),
            ConvolutionBlock(num_filters, 256, 3, activation=False),
            nn.BatchNorm2D(256)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.x(x)
        x = paddle.add(x1, x)
        x = self.relu(x)
        return x


def add3(x1, x2, x3):
    y1 = paddle.add(x1, x2)
    y = paddle.add(y1, x3)
    return y


def add4(x1, x2, x3, x4):
    y1 = paddle.add(x1, x2)
    y2 = paddle.add(y1, x3)
    y = paddle.add(y2, x4)
    return y


class Resize(nn.Layer):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        x = interpolate(x, size=[self.size, self.size], mode='bilinear')
        return x


class Model(nn.Layer):
    def __init__(self, DropoutRatio=0.05):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv10 = nn.Sequential(
            # nn.BatchNorm2D(8),
            nn.Conv2D(8, 32, 3, 1, 'same'),
            BatchActivate(32),
            Basic(32, 32)
            # Basic(32, 32)
        )
        self.downconv10 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 2, 'same'),
            BatchActivate(64),
            # nn.Dropout2D(DropoutRatio / 2)
        )
        self.conv1 = nn.Sequential(
            ResidualBlock(64, 64),
            Bottle(256, 64),
            Bottle(256, 64)
        )
        self.downconv1 = nn.Sequential(
            nn.Conv2D(256, 32, 3, 1, 'same'),
            BatchActivate(32)
        )
        self.downconv1_1 = nn.Sequential(
            nn.Conv2D(256, 64, 3, 2, 'same'),
            BatchActivate(64),
            # nn.Dropout2D(DropoutRatio / 2)
        )
        self.conv2 = nn.Sequential(
            Basic(32, 32),
            Basic(32, 32)
        )
        self.conv2_1 = nn.Sequential(
            Basic(64, 64),
            Basic(64, 64)
        )
        self.downconv2 = nn.Sequential(
            nn.Conv2D(64, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv2_1 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 2, 'same'),
            nn.BatchNorm2D(64)
        )
        self.downconv2_2 = nn.Sequential(
            ConvolutionBlock(64, 128, 3, 2)
        )
        self.conv3 = nn.Sequential(
            Basic(32, 32),
            Basic(32, 32)
        )
        self.conv3_1 = nn.Sequential(
            Basic(64, 64),
            Basic(64, 64)
        )
        self.conv3_2 = nn.Sequential(
            Basic(128, 128),
            Basic(128, 128)
        )
        self.downconv31 = nn.Sequential(
            nn.Conv2D(64, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv32 = nn.Sequential(
            nn.Conv2D(128, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv3_11 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 2, 'same'),
            nn.BatchNorm2D(64)
        )
        self.downconv3_12 = nn.Sequential(
            nn.Conv2D(128, 64, 1, 1, 'same'),
            nn.BatchNorm2D(64),
            Resize(32)
        )
        self.downconv3_21 = nn.Sequential(
            ConvolutionBlock(32, 32, 3, 2),
            nn.Conv2D(32, 128, 3, 2, 'same'),
            nn.BatchNorm2D(128)
        )
        self.downconv3_22 = nn.Sequential(
            nn.Conv2D(64, 128, 3, 2, 'same'),
            nn.BatchNorm2D(128)
        )
        self.downconv3_3 = nn.Sequential(
            ConvolutionBlock(128, 256, 3, 2)
        )
        self.conv4 = nn.Sequential(
            Basic(32, 32),
            Basic(32, 32)
        )
        self.conv4_1 = nn.Sequential(
            Basic(64, 64),
            Basic(64, 64)
        )
        self.conv4_2 = nn.Sequential(
            Basic(128, 128),
            Basic(128, 128)
        )
        self.conv4_3 = nn.Sequential(
            Basic(256, 256),
            Basic(256, 256)
        )
        self.downconv41 = nn.Sequential(
            nn.Conv2D(64, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv42 = nn.Sequential(
            nn.Conv2D(128, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv43 = nn.Sequential(
            nn.Conv2D(256, 32, 1, 1, 'same'),
            nn.BatchNorm2D(32),
            Resize(64)
        )
        self.downconv4_11 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 2, 'same'),
            nn.BatchNorm2D(64)
        )
        self.downconv4_12 = nn.Sequential(
            nn.Conv2D(128, 64, 1, 1, 'same'),
            nn.BatchNorm2D(64),
            Resize(32)
        )
        self.downconv4_13 = nn.Sequential(
            nn.Conv2D(256, 64, 1, 1, 'same'),
            nn.BatchNorm2D(64),
            Resize(32)
        )
        self.downconv4_21 = nn.Sequential(
            ConvolutionBlock(32, 32, 3, 2),
            nn.Conv2D(32, 128, 3, 2, 'same'),
            nn.BatchNorm2D(128)
        )
        self.downconv4_22 = nn.Sequential(
            nn.Conv2D(64, 128, 3, 2, 'same'),
            nn.BatchNorm2D(128)
        )
        self.downconv4_23 = nn.Sequential(
            nn.Conv2D(256, 128, 1, 1, 'same'),
            nn.BatchNorm2D(128),
            Resize(16)
        )
        self.downconv4_31 = nn.Sequential(
            ConvolutionBlock(32, 32, 3, 2),
            ConvolutionBlock(32, 64, 3, 2),
            nn.Conv2D(64, 256, 3, 2, 'same'),
            nn.BatchNorm2D(256)
        )
        self.downconv4_32 = nn.Sequential(
            ConvolutionBlock(64, 64, 3, 2),
            nn.Conv2D(64, 256, 3, 2, 'same'),
            nn.BatchNorm2D(256)
        )
        self.downconv4_33 = nn.Sequential(
            nn.Conv2D(128, 256, 3, 2, 'same'),
            nn.BatchNorm2D(256)
        )
        self.downconv4_1 = ConvolutionBlock(64, 32, 1, 1)
        self.downconv4_2 = ConvolutionBlock(128, 32, 1, 1)
        self.downconv4_3 = ConvolutionBlock(256, 32, 1, 1)
        self.conv567 = Resize(64)
        self.conv9 = nn.Sequential(
            nn.Conv2D(128, 32, 3, 1, 'same', 1),
            BatchActivate(32)
        )
        self.conv9_1 = nn.Sequential(
            nn.Conv2D(128, 32, 3, 1, 'same', 2),
            BatchActivate(32)
        )
        self.conv9_2 = nn.Sequential(
            nn.Conv2D(128, 32, 3, 1, 'same', 4),
            BatchActivate(32)
        )
        self.conv9_3 = nn.Sequential(
            nn.Conv2D(128, 32, 3, 1, 'same', 8),
            BatchActivate(32)
        )
        self.up8 = nn.Sequential(
            ConvolutionBlock(128, 64, 3, 1),
            nn.Conv2DTranspose(64, 32, 3, 2, 'same'),
            BatchActivate(32)
        )
        self.output_layer = nn.Sequential(
            ConvolutionBlock(64, 32, 3, 1),
            nn.Conv2D(32, 1, 1, 1, 'same'),
            nn.Sigmoid()
        )




    def forward(self, x):
        conv10 = self.conv10(x)
        downconv10 = self.downconv10(conv10)
        conv1 = self.conv1(downconv10)
        downconv1 = self.downconv1(conv1)
        downconv1_1 = self.downconv1_1(conv1)
        conv2 = self.conv2(downconv1)
        conv2_1 = self.conv2_1(downconv1_1)
        downconv2 = self.relu(paddle.add(self.downconv2(conv2_1), conv2))
        downconv2_1 = self.relu(paddle.add(self.downconv2_1(conv2), conv2_1))
        downconv2_2 = self.downconv2_2(downconv2_1)
        conv3 = self.conv3(downconv2)
        conv3_1 = self.conv3_1(downconv2_1)
        conv3_2 = self.conv3_2(downconv2_2)
        downconv3 = self.relu(add3(conv3, self.downconv31(conv3_1), self.downconv32(conv3_2)))
        downconv3_1 = self.relu(add3(self.downconv3_11(conv3), conv3_1, self.downconv3_12(conv3_2)))
        downconv3_2 = self.relu(add3(self.downconv3_21(conv3), self.downconv3_22(conv3_1), conv3_2))
        downconv3_3 = self.downconv3_3(downconv3_2)
        conv4 = self.conv4(downconv3)
        conv4_1 = self.conv4_1(downconv3_1)
        conv4_2 = self.conv4_2(downconv3_2)
        conv4_3 = self.conv4_3(downconv3_3)
        downconv4 = self.relu(add4(conv4, self.downconv41(conv4_1), self.downconv42(conv4_2), self.downconv43(conv4_3)))
        downconv4_1 = self.relu(add4(self.downconv4_11(conv4), conv4_1, self.downconv4_12(conv4_2), self.downconv4_13(conv4_3)))
        downconv4_2 = self.relu(add4(self.downconv4_21(conv4), self.downconv4_22(conv4_1), conv4_2, self.downconv4_23(conv4_3)))
        downconv4_3 = self.relu(add4(self.downconv4_31(conv4), self.downconv4_32(conv4_1), self.downconv4_33(conv4_2), conv4_3))
        downconv4_1 = self.downconv4_1(downconv4_1)
        downconv4_2 = self.downconv4_2(downconv4_2)
        downconv4_3 = self.downconv4_3(downconv4_3)
        conv5 = self.conv567(downconv4_1)
        conv6 = self.conv567(downconv4_2)
        conv7 = self.conv567(downconv4_3)
        conv8 = paddle.concat([downconv4, conv5], axis=1)
        conv8 = paddle.concat([conv8, conv6], axis=1)
        conv8 = paddle.concat([conv8, conv7], axis=1)
        conv9 = self.conv9(conv8)
        conv9_1 = self.conv9_1(conv8)
        conv9_2 = self.conv9_2(conv8)
        conv9_3 = self.conv9_3(conv8)
        conv9_4 = paddle.concat([conv9, conv9_1], axis=1)
        conv9_4 = paddle.concat([conv9_4, conv9_2], axis=1)
        conv9_4 = paddle.concat([conv9_4, conv9_3], axis=1)
        up8 = self.up8(conv9_4)
        up9 = paddle.concat([up8, conv10], axis=1)
        output_layer = self.output_layer(up9)
        return output_layer


def get_train_data(train_images_dir, train_labels_dir, img_h=128, img_w=128):
    # scaler = MinMaxScaler()
    train_images=[]
    train_labels=[]
    files = os.listdir(train_images_dir)#get file names
    files2 = os.listdir(train_labels_dir)
    #total_images = np.zeros([len(files), img_h, img_w, N_channels])#for storing training imgs
    for idx in range(len(files)):
        img = np.load(os.path.join(train_images_dir,files[idx]))
        # img = np.delete(img, [0, 2,3,4, 7], axis=2)
        img = (img[:,:,:]-105)/255.0
        #img=np.reshape(img,(img_h, img_w,14))
        # img = scaler.fit_transform(img.reshape(-1, 1))
        img = np.reshape(img, (img_h, img_w, 8))
#        img = img[:,:,3:12]
        img = np.array(img, dtype="float32")
        train_images.append(img)

        label = cv2.imread(os.path.join(train_labels_dir,files2[idx]), cv2.IMREAD_GRAYSCALE)
        label = np.reshape(label, (img_h, img_w, 1))
        label = np.array(label, dtype="float32") / 255.0
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels=np.array(train_labels)
    return train_images, train_labels


class Mydata(Dataset):
    def __init__(self, data):
        super().__init__()
        self.X = data[0]
        self.y = data[1]

    def __getitem__(self, item):
        X = paddle.transpose(paddle.to_tensor(self.X[item, :, :, :]), perm=[2, 0, 1])
        y = paddle.transpose(paddle.to_tensor(self.y[item, :, :, :]), perm=[2, 0, 1])
        return X, y

    def __len__(self):
        return len(self.X)


class MyIoUMetric(Metric):
    def __init__(self, name='my_iou_metric'):
        super(MyIoUMetric, self).__init__()
        self._name = name
        self.reset()

    def reset(self):
        self.metric = []
        # self.i = 0

    def update(self, preds, labels):
        # self.i += 1
        preds = preds > 0.5
        batch_size = labels.shape[0]
        for batch in range(batch_size):
            t, p = labels[batch] > 0, preds[batch] > 0
            intersection = np.logical_and(t, p)
            union = np.logical_or(t, p)
            iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
            thresholds = np.arange(0.5, 1, 0.05)
            s = []
            for thresh in thresholds:
                s.append(iou > thresh)
            # print(np.mean(s), end=',')
            self.metric.append(np.mean(s))
        # print()

    def accumulate(self):
        # print(self.i)
        return np.mean(self.metric)

    def name(self):
        return self._name


if __name__ == '__main__':
    paddle.device.set_device("gpu")
    SEED = 7
    np.random.seed(SEED)
    EPOCHS = 3000
    BS = 40
    print('开始读取数据')
    conf = configparser.ConfigParser()
    conf.read("config.ini", encoding='Utf-8')
    train_images_dir = conf.get('dir', 'data_dir')
    train_labels_dir = conf.get('dir', 'label_dir')
    model_path = conf.get('dir', 'model')
    logs_dir = conf.get('dir', 'logs')
    test_dir = conf.get('dir', 'test_dir')
    all_x, all_y = get_train_data(train_images_dir, train_labels_dir)
    all_x, test_x, all_y, test_y = train_test_split(all_x, all_y, test_size=0.2, random_state=SEED)
    train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.2, random_state=SEED)
    # for i in range(len(test_x)):
    #     os.mkdir(os.path.join(test_dir, '/data/'))
    #     os.mkdir(os.path.join(test_dir, '/label/'))
    #     np.save(os.path.join(test_dir, f'/data/{i}.npy'), test_x[i])
    #     cv2.imwrite(os.path.join(test_dir, f'/label/{i}.png'), test_y[i] * 255)
    print('读取完成')
    train_dataset = Mydata([all_x, all_y])
    val_dataset = Mydata([valid_x, valid_y])
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
    model = Model()
    # for data, label in train_loader:
    #     output = model(data)
    #     print(output.shape)
    #     print(label.shape)
    #     break
    model = paddle.Model(model)
    # print(model)

    optimizer = paddle.optimizer.Adam(learning_rate=0.02, parameters=model.parameters())
    # loss = nn.functional.binary_cross_entropy()
    model.prepare(optimizer, nn.BCELoss(), metrics=MyIoUMetric())

    early_stopping = paddle.callbacks.EarlyStopping('my_iou_metric', mode='max', patience=60, 
                                                    verbose=1, min_delta=0, baseline=None, save_best_model=True)
    reduce_lr = paddle.callbacks.ReduceLROnPlateau(monitor='my_iou_metric', mode='max', factor=0.5, patience=12, min_lr=0.00001)
    modelcheck = paddle.callbacks.ModelCheckpoint(save_dir=model_path, save_freq=100)
    vision = paddle.callbacks.VisualDL(log_dir=logs_dir)
    model.fit(train_data=train_loader, eval_data=val_loader,  epochs=EPOCHS,
              callbacks=[modelcheck, early_stopping, reduce_lr, vision], verbose=1)

