import os
import re
import matplotlib.pyplot as plt
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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def list_all_files(root: str, keys=[], outliers=[], full_path=False):
    """
    列出某个文件下所有文件的全路径

    Author:   wangye
    Datetime: 2019/4/16 18:03

    :param root: 根目录
    :param keys: 所有关键字
    :param outliers: 所有排除关键字
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


def pre_processing():
    for fi, f in enumerate(list_all_files(raw_data_path)):
        img = Image.open(f)
        img_ = img
        labels = re.findall(r'\\(\d*)[._ (]', f)[0]
        print(fi, f, img.size, labels)
        # 二值化，切割，放缩
        img = img.crop((0, 80, 160, 110))
        grey_img = img.convert('L')
        img_generator = lambda thd: grey_img.point([0 if i < thd else 1 for i in range(256)], '1')
        # todo: Auto grey
        grey_imgs = [img_generator(th) for th in [60, 100, 150, 100, 60]]  # 切割得到清晰的数字
        # 切割
        digits = [grey_imgs[i].crop((
            grey_imgs[i].width * i // number_of_digits,
            0,
            grey_imgs[i].width * (i + 1) // number_of_digits,
            grey_imgs[i].height
        )).resize((target_size, target_size), Image.ANTIALIAS) for i in range(number_of_digits)]
        yield digits, labels, img_


if __name__ == '__main__':
    plt.rcParams['font.size'] = 20
    raw_data_path = '20190828'
    target_size = 28
    number_of_digits = 5
    model_save_path = 'model_digit.mdl'
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = torch.load(model_save_path)
    model.eval()
    for digit, target, raw_img in pre_processing():
        with torch.no_grad():
            data = np.array([np.array(d) for d in digit])
            data = data[:, np.newaxis, :, :]
            data = torch.FloatTensor(data)
            data = data.to(device)
            output = model(data)
            pred_label = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        plt.imshow(raw_img)
        plt.title(f'real number are {target}, predicted number is ' + ''.join([str(p.item()) for p in pred_label]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
