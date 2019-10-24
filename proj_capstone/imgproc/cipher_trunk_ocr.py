import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def histeq(im, nbr_bins=256):
    """ 对一幅灰度图像进行直方图均衡化"""

    # 计算图像的直方图
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # 归一化

    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ 使用A. Chambolle（2005）在公式（11）中的计算步骤实现Rudin-Osher-Fatemi（ROF）去噪模型
      输入：含有噪声的输入图像（灰度图像）、U 的初始值、TV 正则项权值、步长、停业条件
      输出：去噪和去除纹理后的图像、纹理残留"""

    m, n = im.shape  # 噪声图像的大小

    # 初始化
    U = U_init
    Px = im  # 对偶域的x 分量
    Py = im  # 对偶域的y 分量
    error = 1

    while error > tolerance:
        Uold = U

        # 原始变量的梯度
        GradUx = np.roll(U, -1, axis=1) - U  # 变量U 梯度的x 分量
        GradUy = np.roll(U, -1, axis=0) - U  # 变量U 梯度的y 分量

        # 更新对偶变量
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew  # 更新x 分量（对偶）
        Py = PyNew / NormNew  # 更新y 分量（对偶）

        # 更新原始变量
        RxPx = np.roll(Px, 1, axis=1)  # 对x 分量进行向右x 轴平移
        RyPy = np.roll(Py, 1, axis=0)  # 对y 分量进行向右y 轴平移

        DivP = (Px - RxPx) + (Py - RyPy)  # 对偶域的散度
        U = im + tv_weight * DivP  # 更新原始变量

        # 更新误差
        error = np.linalg.norm(U - Uold) / np.sqrt(n * m);

    return U, im - U  # 去噪后的图像和纹理残余


class Classifier:
    def __init__(self, model_path):
        self.number_of_digits = 5
        self.target_size = 28
        self.model_save_path = model_path
        print(os.getcwd())
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
        print(img.size)
        if img.size[1] == 120:
            img = img.crop((0, 78, 160, 110))
        # img = img.crop((0, 8, 160, 32))
        # img = np.array(img)
        # print(img)
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.imshow(img)
        # plt.show()
        # plt.show()
        grey_img = img.convert('L')
        grey_img = np.array(grey_img)
        plt.subplot(4, 1, 2)
        plt.imshow(grey_img)
        # grey_img = denoise(grey_img,grey_img,0.9)[0]
        plt.subplot(4, 1, 3)
        plt.imshow(grey_img)
        grey_img = histeq(grey_img)[0]
        plt.subplot(4, 1, 4)
        plt.imshow(grey_img)
        grey_img = np.uint8(grey_img)
        grey_img = Image.fromarray(grey_img)
        img_generator = lambda thd: grey_img.point([0 if i < thd else 1 for i in range(256)], '1')
        grey_imgs = [img_generator(th) for th in [60, 100, 150, 100, 60]]  # 切割得到清晰的数字
        grey_imgs = [img_generator(th) for th in [200, 200, 220, 210, 190]]  # 切割得到清晰的数字
        # grey_imgs = [img_generator(th) for th in [200, 200, 220, 220, 200]]  # 切割得到清晰的数字

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
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(img)
            for i, digit in enumerate(digits):
                plt.subplot(2, 5, i + 6)
                plt.imshow(digit)
                data = digit[np.newaxis, np.newaxis, :, :]
                data = torch.FloatTensor(data)
                output = self.model(data)
                pred_label = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                ret.append(pred_label.item())
        plt.suptitle(f'the prediction is: {ret}')
        plt.show()
        return ret


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'FangSong'
    plt.rcParams['font.size'] = 20
    path = r'D:\Code\Github\python\cipher_trunk_ocr\proj_capstone\img\aa.jpg'
    # path = r'D:\Code\Github\python\cipher_trunk_ocr\proj_capstone\img\aa.jpg'
    # path = r'raw_digits/20190920/160x32/82916.jpg'
    classifier = Classifier('model_digit_cpu.mdl')
    print(classifier.fetch_digits_from_image(path))
