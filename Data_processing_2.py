import numpy as np
import cv2 as cv
import os
import datetime
import configparser


conf = configparser.ConfigParser()
conf.read("config.ini", encoding='Utf-8')
rgb = conf.get('dir', 'gpm')
ci = conf.get('dir', 'nc_out')
ci_gpm = conf.get('dir', 'ci_gpm')

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0),
            Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]

# 计算两个点间的欧式距离
def get_dist(seed_location1, seed_location2):
    l1 = im[seed_location1.x, seed_location1.y]
    l2 = im[seed_location2.x, seed_location2.y]
    count = np.sqrt(np.sum(np.square(l1-l2)))
    return count

def se(class_k, x, y):
    T = 0         # 阈值
    seed_list = []
    seed_list.append(Point(x, y))
    while (len(seed_list) > 0):
        seed_tmp = seed_list[0]
        # 将以生长的点从一个类的种子点列表中删除
        seed_list.pop(0)
        img_mark[seed_tmp.x, seed_tmp.y] = class_k

        # 遍历8邻域
        for i in range(8):
            tmpX = seed_tmp.x + connects[i].x
            tmpY = seed_tmp.y + connects[i].y

            if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                continue
            dist = get_dist(seed_tmp, Point(tmpX, tmpY))
            # 在种子集合中满足条件的点进行生长
            if (dist <= T and img_mark[tmpX, tmpY] == 0):
                # for j in range(8):
                #     img_re[tmpX, tmpY][j] = im[tmpX, tmpY][j]
                img_mark[tmpX, tmpY] = class_k
                seed_list.append(Point(tmpX, tmpY))

cis = os.listdir(ci)
rgbs = os.listdir(rgb)

ccis = []
ci_p = []

for i in range(len(cis)):
    cis[i] = cis[i][:-4]
for i in range(len(rgbs)):
    rgbs[i] = rgbs[i][:-4]

for i in range(len(cis)):
    if cis[i][-4:] == '1459':
        l = list(cis[i])
        l[-4:] = '0000'
        ccis.append(''.join(l))

    elif cis[i][-4:] == '2959':
        l = list(cis[i])
        l[-4:] = '0000'
        ccis.append(''.join(l))
    elif cis[i][-4:] == '5959':
        l = list(cis[i])
        l[-4:] = '3000'
        ccis.append(''.join(l))
    else:
        print(cis[i])

n1 = []
for i in range(len(ccis)):
    year = ccis[i][:4]
    month = ccis[i][4:6]
    day = ccis[i][6:8]
    hour = ccis[i][8:10]
    minute = ccis[i][10:12]
    second = ccis[i][12:14]
    n1.append(datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)))

n2 = []
for i in range(len(n1)):
    date1 = n1[i] + datetime.timedelta(minutes=+30)
    n = []
    for j in n1[i], date1:
        b = str(j)
        for z in '- :':
            b = str(b).replace(z, '')
        n.append(b)
    n.append(cis[i])
    n2.append(n)
c = n2
cuo = []
for i in range(len(c)):
    rgb_1 = cv.imread(rgb + '\\' + c[i][0] + '.png', 0)
    rgb_2 = cv.imread(rgb + '\\' + c[i][1] + '.png', 0)
    ci_1 = cv.imread(ci + '\\' + c[i][2] + '.png', 0)
    if (rgb_1 is None) or (rgb_2 is None):
        cuo.append(c[i])
    else:
        rgb_1 = 255 - rgb_1
        rgb_2 = 255 - rgb_2
        x = 1
        rgb_1 = np.pad(rgb_1, ((x,x),(x,x)),mode='constant')
        rgb_2 = np.pad(rgb_2, ((x,x),(x,x)),mode='constant')
        im = ci_1
        im_shape = im.shape
        height = im_shape[0]
        width = im_shape[1]
        # 标记，判断种子是否已经生长
        img_mark = np.zeros([height, width])
        class_k = 0
        for a in range(height):
            for b in range(width):
                if ci_1[a, b] > 0 and img_mark[a, b] == 0:
                    class_k += 1
                    se(class_k, a, b)

        for z in range(1, class_k+1):
            flag = False
            for a in range(height):
                for b in range(width):
                    if img_mark[a, b] == z:
                        if np.sum(rgb_1[a:a+2*x+1, b:b+2*x+1]) or np.sum(rgb_2[a:a+2*x+1, b:b+2*x+1]):
                            flag = True
                            break
                if flag:
                    break
            if not flag:
                for a in range(height):
                    for b in range(width):
                        if img_mark[a, b] == z:
                            ci_1[a, b] = 0
        cv.imwrite(os.path.join(ci_gpm, c[i][2] + '.png'), ci_1)
