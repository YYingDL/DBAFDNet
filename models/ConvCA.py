# this file contain a ConvCA method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import torch.nn as nn
import torch

class CorrLayer(nn.Module):
    def __init__(self):
        super(CorrLayer, self).__init__()

    def forward(self, X, T):
        # X: n_batch, 1, 1, n_samples
        # T: n_batch, 1, n_classes, n_samples
        T = torch.swapaxes(T, -1, -2)
        corr_xt = torch.matmul(X, T)  # n_batch, 1, 1, n_classes
        corr_xx = torch.sum(torch.square(X), -1, keepdim=True)
        corr_tt = torch.sum(torch.square(T), -2, keepdim=True)
        corr = corr_xt / (torch.sqrt(corr_xx) * torch.sqrt(corr_tt))
        return corr


class ConvCA(nn.Module):
    def __init__(self, reference_signals, subban_no=1, n_channels=9, sample_length=250):
        super(ConvCA, self).__init__()
        # 定义卷积层和池化层
        self.signals_conv1 = nn.Conv2d(in_channels=subban_no, out_channels=16, kernel_size=(9,9), padding=(4,4))
        self.signals_conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(n_channels, 1),padding=(4,0))
        self.signals_conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(n_channels, 1))
        self.signals_dropout = nn.Dropout(0.75)

        self.reference_siganls = reference_signals
        self.reference_conv1 = nn.Conv2d(in_channels=n_channels, out_channels=40, kernel_size=(1, 9), padding=(0, 4))
        self.reference_conv2 = nn.Conv2d(in_channels=40, out_channels=1, kernel_size=(1, 9), padding=(0, 4))
        self.reference_dropout = nn.Dropout(0.15)

        self.corr = CorrLayer()
        self.dense = nn.Linear(in_features=40, out_features=40)

    def forward(self, x):
        # signals-cnn
        x = x[:, 0].unsqueeze(1)
        x = self.signals_conv1(x)
        x = self.signals_conv2(x)
        x = self.signals_conv3(x)
        x = self.signals_dropout(x)
        # reference-cnn
        reference_siganls = torch.tile(self.reference_siganls,[x.shape[0], 1, 1, 1])
        y = self.reference_conv1(reference_siganls.to(x.device))
        y = self.reference_conv2(y)
        y = self.reference_dropout(y)
        # cal correlation analysis
        r0 = self.corr(x, y)
        out = self.dense(r0.squeeze())
        return out




if __name__ == '__main__':
    input = torch.randn(64, 3, 9, 250)
    reference = torch.randn(1, 9, 40, 250)
    ### mobilevit_xxs
    model1 = ConvCA(reference_signals=reference)
    out = model1(input)
    print(out.shape)
