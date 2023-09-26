import torch

from torch import nn
from torch.nn import functional as F
import math


def get_model(model_name, train_set=None):

    if model_name == "net_cifar100_allcnnc":
        #model = net_cifar100_allcnnc(num_classes=100)
        model = net_cifar100_allcnnc()

    if model_name == "linear":
        model = LinearRegression(input_dim=train_set[0][0].shape[0], output_dim=1)

    if model_name == "mlp":
        model = Mlp(n_classes=10, dropout=False)
    
    if model_name == "mlp_dropout":
        model = Mlp(n_classes=10, dropout=True)

    elif model_name == "resnet34":
        model = ResNet([3, 4, 6, 3], num_classes=10)

    elif model_name == "resnet34_100":
        model = ResNet([3, 4, 6, 3], num_classes=100)

    elif model_name == "densenet121":
        model = DenseNet121(num_classes=10)

    elif model_name == "densenet121_100":
        model = DenseNet121(num_classes=100)

    elif model_name == "matrix_fac_1":
        model = LinearNetwork(6, [1], 10, bias=False)
    elif model_name == "matrix_fac_4":
        model = LinearNetwork(6, [4], 10, bias=False)
    
    elif model_name == "matrix_fac_10":
        model = LinearNetwork(6, [10], 10, bias=False)

    elif model_name == "linear_fac":
        model = LinearNetwork(6, [], 10, bias=False)

    return model

# =====================================================
# Linear Network
class LinearNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True):
        super().__init__()

        # iterate averaging:
        self._prediction_params = None

        self.input_size = input_size
        if output_size:
            self.output_size = output_size
            self.squeeze_output = False
        else :
            self.output_size = 1
            self.squeeze_output = True

        if len(hidden_sizes) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size, bias=bias)
        else:
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)

    def forward(self, x):
        '''
            x: The input patterns/features.
        '''
        x = x.view(-1, self.input_size)
        out = x

        for layer in self.hidden_layers:
            Z = layer(out)
            # no activation in linear network.
            out = Z

        logits = self.output_layer(out)
        if self.squeeze_output:
            logits = torch.squeeze(logits)

        return logits

# =====================================================
# Logistic
class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# =====================================================
# MLP
class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.dropout=dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits

# =================================
# ====================
# ResNet
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_DenseNet(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes):
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=32,
        num_classes=num_classes)

def DenseNet169():
    return DenseNet(Bottleneck_DenseNet, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck_DenseNet, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck_DenseNet, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

#########################################import deepobs ne_cifar100_allcnnc###################################################
#import needed modules
def tfconv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    tf_padding_type=None,
):
    modules = []
    if tf_padding_type == "same":
        padding = nn.ZeroPad2d(0)
        hook = hook_factory_tf_padding_same(kernel_size, stride)
        padding.register_forward_pre_hook(hook)
        modules.append(padding)

    modules.append(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    )
    return nn.Sequential(*modules)

#hook factory
def hook_factory_tf_padding_same(kernel_size, stride):
    """Generates the torch pre forward hook that needs to be registered on
    the padding layer to mimic tf's padding 'same'"""

    def hook(module, input):
        """The hook overwrites the padding attribute of the padding layer."""
        image_dimensions = input[0].size()[-2:]
        module.padding = _determine_padding_from_tf_same(
            image_dimensions, kernel_size, stride
        )

    return hook

# determine padding
def _determine_padding_from_tf_same(
    input_dimensions, kernel_dimensions, stride_dimensions
):
    """Implements tf's padding 'same' for kernel processes like convolution or pooling.
    Args:
        input_dimensions (int or tuple): dimension of the input image
        kernel_dimensions (int or tuple): dimensions of the convolution kernel
        stride_dimensions (int or tuple): the stride of the convolution
     Returns: A padding 4-tuple for padding layer creation that mimics tf's padding 'same'.
     """

    # get dimensions
    in_height, in_width = input_dimensions

    if isinstance(kernel_dimensions, int):
        kernel_height = kernel_dimensions
        kernel_width = kernel_dimensions
    else:
        kernel_height, kernel_width = kernel_dimensions

    if isinstance(stride_dimensions, int):
        stride_height = stride_dimensions
        stride_width = stride_dimensions
    else:
        stride_height, stride_width = stride_dimensions

    # determine the output size that is to achive by the padding
    out_height = math.ceil(in_height / stride_height)
    out_width = math.ceil(in_width / stride_width)

    # determine the pad size along each dimension
    pad_along_height = max(
        (out_height - 1) * stride_height + kernel_height - in_height, 0
    )
    pad_along_width = max((out_width - 1) * stride_width + kernel_width - in_width, 0)

    # determine padding 4-tuple (can be asymmetric)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom

def mean_allcnnc():
    """The all convolution layer implementation of torch.mean()."""
    # TODO implement pre forward hook to adapt to arbitary image size for other data sets than cifar100
    return nn.Sequential(nn.AvgPool2d(kernel_size=(6, 6)), nn.Flatten())

class net_cifar100_allcnnc(nn.Sequential):
    def __init__(self):
        super(net_cifar100_allcnnc, self).__init__()

        self.add_module("dropout1", nn.Dropout(p=0.2))

        self.add_module(
            "conv1",
            tfconv2d(
                in_channels=3, out_channels=96, kernel_size=3, tf_padding_type="same",
            ),
        )
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "conv2",
            tfconv2d(
                in_channels=96, out_channels=96, kernel_size=3, tf_padding_type="same",
            ),
        )
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96,
                out_channels=96,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        self.add_module("relu3", nn.ReLU())

        self.add_module("dropout2", nn.Dropout(p=0.5))

        self.add_module(
            "conv4",
            tfconv2d(
                in_channels=96, out_channels=192, kernel_size=3, tf_padding_type="same",
            ),
        )
        self.add_module("relu4", nn.ReLU())
        self.add_module(
            "conv5",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu5", nn.ReLU())
        self.add_module(
            "conv6",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                stride=(2, 2),
                tf_padding_type="same",
            ),
        )
        self.add_module("relu6", nn.ReLU())

        self.add_module("dropout3", nn.Dropout(p=0.5))

        self.add_module(
            "conv7", tfconv2d(in_channels=192, out_channels=192, kernel_size=3)
        )
        self.add_module("relu7", nn.ReLU())
        self.add_module(
            "conv8",
            tfconv2d(
                in_channels=192,
                out_channels=192,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu8", nn.ReLU())
        self.add_module(
            "conv9",
            tfconv2d(
                in_channels=192,
                out_channels=100,
                kernel_size=1,
                tf_padding_type="same",
            ),
        )
        self.add_module("relu9", nn.ReLU())

        self.add_module("mean", mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)