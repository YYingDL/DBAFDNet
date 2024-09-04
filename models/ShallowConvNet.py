# this file contain a ShallowConvNet method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import torch.nn as nn
import torch

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm


    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNet(nn.Module):
    def __init__(self, num_channels=9,sampling_rate=250,F1=40,T1=25,F2=40,P1_T=75,P1_S=15,drop_out=0.5,pool_mode= 'mean'):
        super(ShallowConvNet, self).__init__()
        kernel_size = int(sampling_rate * 0.12)
        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))


        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, kernel_size), padding='same', max_norm=2.),
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, pooling_kernel_size), (1, pooling_stride_size)),
            ActLog(),
            nn.Dropout(drop_out),
            # nn.Flatten(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * 8, 40)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.net(x)
        output = self.flatten(x)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    input = torch.randn(64, 3, 9, 250)
    reference = torch.randn(1, 9, 40, 250)
    ### mobilevit_xxs
    model1 = ShallowConvNet()
    out = model1(input)
    print(out.shape)
