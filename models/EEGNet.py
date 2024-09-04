# this file contain a EEGNet method to decode SSVEP singles
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


class EEGNet(nn.Module):
    def __init__(self, sample_length=250, num_channels=9, F1=8, D=2, F2='auto', T1=65, T2=17, P1=4, P2=8, pool_mode='mean', drop_out=0.5):
        super(EEGNet, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1), padding=(0, T1 // 2), bias=False),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1), stride=4),
            nn.Dropout(drop_out)
        )
        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), padding=(0, T2 // 2), groups=F1 * D),
            nn.Conv2d(F2, F2, 1, stride=1, bias=False, padding=0),
            nn.BatchNorm2d(F2),
            # ActSquare(),
            nn.ELU(),
            pooling_layer((1, P2), stride=8),
            # ActLog(),
            nn.Dropout(drop_out)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * ((sample_length // 4) //8), 40)
        # self.dense = nn.Sequential(
        #     nn.Conv2d(16, 40, (1, 7)),
        #     nn.LogSoftmax(dim=1))


    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.spectral(x)
        x = self.spatial(x)
        x = self.temporal(x)
        output = self.flatten(x)
        output = self.fc(output)
        # x = self.dense(x)
        # x = torch.squeeze(x, 3)
        # output = torch.squeeze(x, 2)
        return output


if __name__ == '__main__':
    input = torch.randn(64, 3, 9, 125)
    reference = torch.randn(1, 9, 40, 250)
    ### mobilevit_xxs
    model1 = EEGNet(sample_length=125)
    out = model1(input)
    print(out.shape)
