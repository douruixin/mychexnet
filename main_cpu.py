import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import 
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator

import cv2
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageQt


class HeatmapGenerator():

    # 初始化热图生成器
    # 全连接神经网络 DenseNet121
    def __init__(self, pathModel, nnArchitecture, nnClassCount, transCrop):
        if nnArchitecture == 'DENSE-NET-121':
            if use_gpu:
                model = DenseNet121(nnClassCount, True).cuda()
            else:
                model = DenseNet121(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-169':
            if use_gpu:
                model = DenseNet169(nnClassCount, True).cuda()
            else:
                model = DenseNet169(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-201':
            if use_gpu:
                model = DenseNet201(nnClassCount, True).cuda()
            else:
                model = DenseNet201(nnClassCount, True)

        if use_gpu:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)

        if use_gpu:
            modelCheckpoint = torch.load(pathModel)
            state_dict = modelCheckpoint['state_dict']
            remove_data_parallel = False

            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                state_dict[new_key] = state_dict[key]
                if match or remove_data_parallel:
                    del state_dict[key]

            model.load_state_dict(modelCheckpoint['state_dict'])
        else:
            modelCheckpoint = torch.load(pathModel, map_location='cpu')
            state_dict = modelCheckpoint['state_dict']
            remove_data_parallel = False

            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(state_dict.keys()):
                match = pattern.match(key)
                new_key = match.group(1) + match.group(2) if match else key
                new_key = new_key[7:] if remove_data_parallel else new_key
                state_dict[new_key] = state_dict[key]
                if match or remove_data_parallel:
                    del state_dict[key]

            model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model.module.densenet121.features
        self.model.eval()

        # 初始化模型权重
        self.weights = list(self.model.parameters())[-2]

        # 图像变换，下采样
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)

        self.transformSequence = transforms.Compose(transformList)

    # --------------------------------------------------------------------------------

    def generate(self, pathImageFile, transCrop, probability):
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        input = torch.autograd.Variable(imageData)

        if use_gpu:
            self.model.cuda()
            output = self.model(input.cuda())
        else:
            output = self.model(input)

        heatmap = None
        for i in range(0, len(self.weights)):
            map = output[0, i, :, :]
            if i == 0:
                heatmap = self.weights[i] * map
            else:
                heatmap += self.weights[i] * map

        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))

        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        img = heatmap * 0.5 + imgOriginal

        img3 = img / np.max(img)
        img3 = 255 * img3
        img3 = img3.astype(np.uint8)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.subplot(121)
        plt.imshow(imgOriginal)
        plt.subplot(122)
        plt.title(probability)
        plt.imshow(img3)
        plt.show()


def predict_chest_xray(imagePath, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):

    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    if nnArchitecture == 'DENSE-NET-121':
        if use_gpu:
            model = DenseNet121(nnClassCount, True).cuda()
        else:
            model = DenseNet121(nnClassCount, True)
    elif nnArchitecture == 'DENSE-NET-169':
        if use_gpu:
            model = DenseNet169(nnClassCount, True).cuda()
        else:
            model = DenseNet169(nnClassCount, True)
    elif nnArchitecture == 'DENSE-NET-201':
        if use_gpu:
            model = DenseNet201(nnClassCount, True).cuda()
        else:
            model = DenseNet201(nnClassCount, True)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    if use_gpu:
        modelCheckpoint = torch.load(pathModel)
        state_dict = modelCheckpoint['state_dict']
        remove_data_parallel = False

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            if match or remove_data_parallel:
                del state_dict[key]

        model.load_state_dict(modelCheckpoint['state_dict'])
    else:
        modelCheckpoint = torch.load(pathModel, map_location='cpu')
        state_dict = modelCheckpoint['state_dict']
        remove_data_parallel = False

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            if match or remove_data_parallel:
                del state_dict[key]

        model.load_state_dict(modelCheckpoint['state_dict'])

    img = Image.open(imagePath).convert('RGB')  # 读取图像
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img2 = data_transforms(img)  # 归一化
    # 因为是一幅图，所以将维度更新为 [1,3,256,256]
    input = img2[None, :, :, :]

    model.eval()

    if use_gpu:
        input = Variable(input.cuda())
    else:
        input = Variable(input)

    output = model(input)
    output = output.cpu().data.numpy()
    maxIndex = output[0].argsort()[-3:][::-1]
    probability = str(CLASS_NAMES[maxIndex[0]] + ":" + format(output[0][maxIndex[0]], '.2%') + ", " +
        CLASS_NAMES[maxIndex[1]] + ":" + format(output[0][maxIndex[1]], '.2%') + ", " +
        CLASS_NAMES[maxIndex[2]] + ":" + format(output[0][maxIndex[2]], '.2%'))

    h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, imgtransCrop)
    h.generate(imagePath, imgtransCrop, probability)
    print(probability)
    return probability


if __name__ == '__main__':
    image_name = '00006821_002.png'
    imagePath = './test/' + image_name
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 2
    imgtransResize = 256
    imgtransCrop = 224
    pathModel = './m-25012018-123527.pth.tar'
    timestampLaunch = ''
    predict_chest_xray(imagePath, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)
    import pandas as pd
    import matplotlib.patches as patches
    labeldata = pd.read_csv('./test/BBox_List_2017.csv')
    image_labeldata = labeldata[labeldata["Image Index"] == image_name]
    im = np.array(Image.open(imagePath), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    for item in image_labeldata.values:
        Bbox_x = int(item[2])
        Bbox_y = int(item[3])
        Bbox_w = int(item[4])
        Bbox_h = int(item[5])
        Finding_Label = item[1]
        rect = patches.Rectangle((Bbox_x, Bbox_y), Bbox_w, Bbox_h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
