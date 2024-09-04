# this file contain a DNN method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import torch.nn as nn
import torch


class GuneyDNN(nn.Module):
    def __init__(self, subban_no=3, n_channels=9, dropout_second_stage=0.1, sample_length=250):
        super(GuneyDNN, self).__init__()
        # 定义卷积层和池化层
        self.harmonic_weight = nn.Conv2d(in_channels=subban_no, out_channels=1, kernel_size=(1,1))
        self.psf_weight = nn.Conv2d(in_channels=1, out_channels=120, kernel_size=(n_channels, 1))
        self.dropout0 = nn.Dropout(dropout_second_stage)
        self.down_sample = nn.Conv2d(in_channels=120, out_channels=120, kernel_size=(1, 2), stride=(1, 2))
        self.dropout1 = nn.Dropout(dropout_second_stage)
        self.relu = nn.ReLU()
        self.time_conv = nn.Conv2d(in_channels=120, out_channels=120, kernel_size=(1, 11), padding=(0, 5))
        self.dropout2 = nn.Dropout(0.95)
        self.fc = nn.Linear(in_features=120 * (sample_length // 2), out_features=40)

    def forward(self, x):
        # 前向传播
        x = x[:, :3]
        x = self.harmonic_weight(x)
        x = self.psf_weight(x)
        x = self.dropout0(x)
        x = self.down_sample(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.time_conv(x)
        x = self.dropout2(x)
        x = self.fc(x.view(-1, self.num_flat_features(x)))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features * s
        return num_features


if __name__ == '__main__':
    input = torch.randn(64, 5, 9, int(250*0.9))
    ### mobilevit_xxs
    model1 = GuneyDNN(sample_length=int(250*0.9))
    out = model1(input)
    print(out.shape)
