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
        torch.save(self.model, 'model_digit_cpu.mdl')
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


def list_all_files(root: str, keys=[], outliers=[], full_path=False):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys: 所有关键字
    :param outliers: 所有排除关键字
    :param full_path: 是否返回全路径，True为全路径
    :return:
            所有根目录下包含关键字的文件全路径
    """
    _files = []
    _list = os.listdir(root)
    for i in range(len(_list)):
        path = os.path.join(root, _list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path, keys, outliers, full_path))
        if os.path.isfile(path) \
                and all([k in path for k in keys]) \
                and not any([o in path for o in outliers]):
            _files.append(os.path.abspath(path) if full_path else path)
    return _files


if __name__ == '__main__':
    files = list_all_files('raw_digits', ['20190828'])
    res = 0
    for f in files:
        classifier = Classifier()
        result = ''.join(map(str, classifier.fetch_digits_from_image(f)))
        label = f[f.rfind('\\') + 1:f.rfind('\\') + 6]
        print(f'result: {result}, label: {label}')
        if result == label:
            res += 1
        try:
            os.rename(f, f[:f.rfind('\\')+1] + result + '.jpg')
        except:
            pass
        finally:
            if os.path.exists(f):
                os.remove(f)

    print(f'accuracy: {res / len(files)}')
