# this file contain a CCNN method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import torch.nn as nn
import torch


class CCNN(nn.Module):
    def __init__(self, subban_no=3, n_channels=9, dropout_rate=0.5, sample_length=560):
        super(CCNN, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=2*n_channels, kernel_size=(n_channels,1)),
                        nn.BatchNorm2d(2*n_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate))

        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels=2 * n_channels, out_channels=2 * n_channels, kernel_size=(1, 10)),
                        nn.BatchNorm2d(2 * n_channels),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate))

        self.fc = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in_features=2 * n_channels * (sample_length - 9), out_features=40),
                        nn.Softmax(dim=1))

    def forward(self, x):
        # 前向传播
        x = torch.fft.rfft(x[:, 0].unsqueeze(1), n=1250, dim=-1)
        real = torch.real(x[:, :, :, 40:320])
        imag = torch.imag(x[:, :, :, 40:320])
        x = torch.cat((real, imag), dim=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    input = torch.randn(64, 3, 9, 250)
    model1 = CCNN()
    out = model1(input)
    print(out.shape)
