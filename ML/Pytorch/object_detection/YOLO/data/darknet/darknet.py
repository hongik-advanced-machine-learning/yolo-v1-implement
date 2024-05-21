import torch
from torch import nn
from torch.functional import F
from torchsummary import summary as summary


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


#    print('\n\n'.join([repr(x) for x in blocks]))

def create_modules(blocks):
    inp_info = blocks[0]  # Captures the information about the input and pre-processing
    modules = blocks[1:-1]  # The layers of the neural network
    loss = blocks[-1]  # Loss function

    module_list = nn.ModuleList()

    index = 0  # indexing blocks helps with implementing route  layers (skip connections)

    prev_filters = 3

    output_filters = []

    for x in modules:
        module = nn.Sequential()

        # If it's a convolutional layer
        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif (x["type"] == "maxpool"):  # if it is a max pooling layer

            # Both YOLO f/ PASCAL and COCO don't use 2X2 pooling with stride 1
            # Tiny-YOLO does use it
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if stride > 1:
                pool = nn.MaxPool2d(kernel_size, stride)

            else:
                pool = MaxPoolStride1(kernel_size)

            module.add_module("pool_{0}".format(index), pool)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return (inp_info, module_list, loss)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.inp, self.module_list, self.loss = create_modules(self.blocks)

    def forward(self, x):
        outputs = {}  # We cache the outputs for the route layer

        for i in range(len(self.module_list)):
            module_type = (self.blocks[i + 1]["type"])
            if module_type == "convolutional" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
        return x


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = F.max_pool2d(padded_x, self.kernel_size, padding=self.pad)
        return pooled_x


if __name__ == '__main__':
    model = Darknet("yolov1.cfg")
    model.load_state_dict(torch.load("yolov1-pytorch.pth"))

    summary(model, (3, 448, 448))
