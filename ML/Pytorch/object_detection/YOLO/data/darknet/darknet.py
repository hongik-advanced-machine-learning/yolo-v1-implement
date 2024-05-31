from torch.nn.modules import activation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DarkNet(nn.Module):
    def __init__(self, cfg_path=None, init_weight=True):
        super(DarkNet, self).__init__()

        # Make layers
        if cfg_path == None:
            self.features = self._make_conv_layers_bn_padding()
        else:
            self.features = self.create_features(cfg_path)  ## for extraction.conv.cfg
        self.fc = self._make_fc_layers()

        # Initialize weights
        # if init_weight:
        #     self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def create_features(self, cfg_path):
        """
            get "extraction.conv.cfg" and covert to torch.nn
        """

        def parse_cfg(cfg_path):
            blocks = []
            fp = open(cfg_path, 'r')
            block = None
            line = fp.readline()
            while line != '':
                line = line.rstrip()
                if line == '' or line[0] == '#':
                    line = fp.readline()
                    continue
                elif line[0] == '[':
                    if block:
                        blocks.append(block)
                    block = dict()
                    block['type'] = line.lstrip('[').rstrip(']')
                    # set default value
                    if block['type'] == 'convolutional':
                        block['batch_normalize'] = 0
                else:
                    key, value = line.split('=')
                    key = key.strip()
                    if key == 'type':
                        key = '_type'
                    value = value.strip()
                    block[key] = value
                line = fp.readline()

            if block:
                blocks.append(block)
            fp.close()
            return blocks

        blocks = parse_cfg(cfg_path)

        models = nn.Sequential()
        conv_id = 0
        prev_filters = 0
        max_pool_id = 0

        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id += 1
                # is_bn = int(block['batch_normalize']) # extraction.conv.weight has no batch_normalize, but it needed.
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad_size = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                models.add_module(f"conv{conv_id}",
                                  nn.Conv2d(prev_filters, filters, kernel_size, stride, pad_size, bias=False))
                models.add_module(f"bn{conv_id}", nn.BatchNorm2d(filters))
                if activation == 'leaky':
                    models.add_module(f"leaky{conv_id}", nn.LeakyReLU(0.1, inplace=True))
                prev_filters = filters

            elif block['type'] == 'maxpool':
                max_pool_id += 1
                pool_size = int(block['size'])
                stride = int(block['stride'])
                models.add_module(f"maxpool{max_pool_id}", nn.MaxPool2d(pool_size, stride))

            # elif block['type'] == 'avgpool':
            #     models.add_module("avgpool", nn.AvgPool2d(7))

            # elif block['type'] == 'connected':
            #     filters = int(block['output'])
            #     models.add_module("fc", nn.Linear(prev_filters, filters))

            # elif block['type'] == 'softmax':
            #     models.add_module("softmax", nn.Softmax())

        # print(models)
        return models

    def _make_conv_layers_bn_padding(self):  ## 20 Convs, used for pretrained by IMAGE Net 1000 class
        """
            padding = 3, with BatchNormalization
        """
        conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            # padding=3 so, output is 224.
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 128, 1, bias=False),  ## kernel size = 1 이므로 padding = 0(defalut)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv

    def _make_conv_layers(self):
        """
            padding = 1 and No BatchNormalization like extraction.conv.cfg
        """
        conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1),
            # padding=3 so, output is 224.
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(192, 128, 1, padding=1),  ## kernel size = 1 이므로 padding = 0(defalut)
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(1024, 512, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv

    def _make_fc_layers(self):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 1000),
            nn.Softmax()
        )
        return fc

    def load_weights(self, weightfile):

        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=4)
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. Images seen by the network (during training)
        weight = np.fromfile(fp, dtype=np.float32)
        # print(f"{weightfile} has {weight.size} weight & bias")
        fp.close()

        ptr = 0
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.modules.conv.Conv2d):
                bn_layer = self.features[i + 1]
                num_w = layer.weight.numel()
                num_b = bn_layer.bias.numel()  # conv_layer doesn't need use bias.. because of bn. so l.bias in "darknet/src/parser.c..void save_convolutional_weights" can be a bn_layer.bias

                # print(f"loading bn weight :{bn_layer}, # of bias(beta) : {num_b}, # of weight(gamma,scale) : {bn_layer.weight.numel()}, # of running_mean : {bn_layer.running_mean.numel()}, # of running_var : {bn_layer.running_var.numel()}")
                bn_layer.bias.data.copy_(
                    torch.from_numpy(weight[ptr: ptr + num_b]).view_as(bn_layer.bias.data))  # l.bias
                ptr += num_b
                bn_layer.weight.data.copy_(
                    torch.from_numpy(weight[ptr: ptr + num_b]).view_as(bn_layer.weight.data))  # l.scale
                ptr += num_b
                bn_layer.running_mean.data.copy_(
                    torch.from_numpy(weight[ptr: ptr + num_b]).view_as(bn_layer.running_mean.data))  # l.scale
                ptr += num_b
                bn_layer.running_var.data.copy_(
                    torch.from_numpy(weight[ptr: ptr + num_b]).view_as(bn_layer.running_var.data))  # l.scale
                ptr += num_b

                # print(f"loading conv weight :{layer}, # of weights : {num_w}")
                layer.weight.data.copy_(torch.from_numpy(weight[ptr: ptr + num_w]).view_as(layer.weight.data))
                ptr += num_w
                # print(f"{weight.size - ptr} weight & bias remain")
                # for layer in self.fc:
        #     if isinstance(layer, nn.modules.Linear):
        #         num_w = layer.weight.numel()
        #         num_b = layer.bias.numel()
        #         print(f"loading fc weight :{layer}, # of weights : {num_w}, # of bias : {num_b}")
        #         layer.bias.data.copy_(torch.from_numpy(weight[ptr: ptr+num_b]).view_as(layer.bias.data))
        #         ptr += num_b
        #         layer.weight.data.copy_(torch.from_numpy(weight[ptr: ptr+num_w]).view_as(layer.weight.data))
        #         ptr += num_w
        #         print(f"{weight.size - ptr} weight & bias remain")

        return True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()
