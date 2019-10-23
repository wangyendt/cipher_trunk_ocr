
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output[0],target[0],output.size(),target.size())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    # torch.save(model, 'model.mdl')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def predict(model, device, pred, label):
    model.eval()
    test_acc = 0
    with torch.no_grad():
        for i, data in enumerate(pred):
            if i < 36:
                plt.subplot(6, 6, i + 1)
                plt.imshow(data)
                plt.xticks([])
                plt.yticks([])

            data = data[np.newaxis, np.newaxis, :, :]
            data = torch.FloatTensor(data)
            data = data.to(device)
            output = model(data)
            pred_label = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if pred_label.item() != int(label[i]):
                test_acc += 1
            if i < 36:
                plt.title(f'prediction: {str(pred_label.item())}')
    print(test_acc, len(pred))
    print(f'test accuracy: {1 - test_acc / len(pred)}')
    plt.show()


def get_test_dataset():
    labels = [re.findall(r'\\(\d*)[_.]', f)[0] for f in list_all_files('data')]
    if not os.path.exists(data_path):
        test_imgs = np.empty((0, target_size, target_size))
        for f in list_all_files('data'):
            print(f)
            data = np.loadtxt(f)[np.newaxis, :, :]
            test_imgs = np.vstack((test_imgs, data))
        with open(data_path, 'w') as outfile:
            for slice_2d in test_imgs:
                np.savetxt(outfile, slice_2d)
                outfile.write('# New slice\n')  # slice row
    else:
        test_imgs = np.loadtxt(data_path).reshape((-1, target_size, target_size))
    return test_imgs, labels


def mnist_train():
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=64, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    if need_to_retrain or not os.path.exists(model_save_path):
        for epoch in range(max_epochs):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
    else:
        model = torch.load(model_save_path)
    test(model, device, test_loader)  # for mnist test
    predict(model, device, test_imgs, labels)  # for chenyu test


def split_and_get_digit_figure():
    ind = 1
    for fi, f in enumerate(list_all_files('raw_digits', [], ['0920'])):
        img = Image.open(f)
        print(fi, f, img.size)
        # 二值化，切割，放缩
        img = img.crop((0, 80, 160, 110))
        grey_img = img.convert('L')
        img_generator = lambda thd: grey_img.point([0 if i < thd else 1 for i in range(256)], '1')
        grey_imgs = [img_generator(th) for th in [60, 100, 150, 100, 60]]  # 切割得到清晰的数字
        # 切割
        digits = [grey_imgs[i].crop((
            grey_imgs[i].width * i // number_of_digits,
            0,
            grey_imgs[i].width * (i + 1) // number_of_digits,
            grey_imgs[i].height
        )).resize((target_size, target_size), Image.ANTIALIAS) for i in range(number_of_digits)]
        for i in range(number_of_digits):
            # 放缩
            np.savetxt('data//' + str(ind) + '.txt', digits[i])
            ind += 1


if __name__ == '__main__':
    torch.manual_seed(1)
    need_to_retrain = True  # os.path.exists('model.mdl')
    max_epochs = 10
    use_cuda = True
    learning_rate = 0.01
    momentum = 0.5
    target_size = 28  # to fit mnist figures
    number_of_digits = 5
    model_save_path = 'model.mdl'
    data_path = 'data.txt'  # don't change

    test_imgs, labels = get_test_dataset()
    mnist_train()
    # split_and_get_digit_figure()
