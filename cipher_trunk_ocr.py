# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author: wang121ye
# datetime: 2019/10/21 21:15
# software: PyCharm


import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Classifier:
    def __init__(self):
        self.number_of_digits = 5
        self.target_size = 28
        self.model_save_path = 'model_digit.mdl'
        if not os.path.exists(self.model_save_path):
            print('model file does not exist! (模型文件不存在, 请联系模型提供者)')
        self.model = torch.load(self.model_save_path).cpu()
        self.model.eval()

    def fetch_digits_from_image(self, img_path):
        img = Image.open(img_path)
        if img.size not in [(160, 120), (160, 32)]:
            print('figure size is not correct! (图片大小不符合规定, 请尝试160*120或160*32的图片大小)')
            return [0, 0, 0, 0, 0]

        # 二值化，切割，放缩
        if img.size[1] == 120:
            img = img.crop((0, 78, 160, 110))
        grey_img = img.convert('L')
        img_generator = lambda thd: grey_img.point([0 if i < thd else 1 for i in range(256)], '1')
        grey_imgs = [img_generator(th) for th in [60, 100, 150, 100, 60]]  # 切割得到清晰的数字

        # 切割
        digits = [np.array(grey_imgs[i].crop((
            grey_imgs[i].width * i // self.number_of_digits,
            0,
            grey_imgs[i].width * (i + 1) // self.number_of_digits,
            grey_imgs[i].height
        )).resize((self.target_size, self.target_size), Image.ANTIALIAS), dtype=np.int) for i in
                  range(self.number_of_digits)]

        ret = []
        with torch.no_grad():
            for digit in digits:
                data = digit[np.newaxis, np.newaxis, :, :]
                data = torch.FloatTensor(data)
                output = self.model(data)
                pred_label = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                ret.append(pred_label.item())
        return ret


if __name__ == '__main__':
    path = r'raw_digits/20190828/73683.jpg'
    # path = r'raw_digits/20190920/160x32/82916.jpg'
    classifier = Classifier()
    print(classifier.fetch_digits_from_image(path))
