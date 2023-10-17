import torch.nn as nn
import torch
import numpy as np
import torch
from torch.distributions.uniform import Uniform
#################################
# Models for federated learning #
#################################
# McMahan et al., 2016; 199,210 parameters

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
class ConvB(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation2(out)
        return out

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        #..................................................................................
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, in_channels//2, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class UNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        # Question here
        n_channels = 32
        self.inc = ConvB(in_channels, n_channels)
        self.down1 = DownBlock(n_channels, n_channels*2, nb_Conv=2)
        self.down2 = DownBlock(n_channels*2, n_channels*4, nb_Conv=2)
        self.down3 = DownBlock(n_channels*4, n_channels*8, nb_Conv=2)
        self.down4 = DownBlock(n_channels*8, n_channels*16, nb_Conv=2)
        self.up4 = UpBlock(n_channels*16, n_channels*4, nb_Conv=2)
        self.up3 = UpBlock(n_channels*8, n_channels*2, nb_Conv=2)
        self.up2 = UpBlock(n_channels*4, n_channels, nb_Conv=2)
        self.up1 = UpBlock(n_channels*2, n_channels, nb_Conv=2)
        self.outc = nn.Conv2d(n_channels, num_classes, kernel_size=(1,1))
        #self.DynamicConv2d = DynamicConv2d(num_classes,num_classes)
        if num_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
        return logits


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(2,1,224,224)).cuda()
    model = UNet('unet',1,5).cuda()
    #param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    #print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    param1 = sum([param.nelement() for param in model.parameters()])
    # param = count_param(model)
    y = model(x)
    # print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    print(param1)
