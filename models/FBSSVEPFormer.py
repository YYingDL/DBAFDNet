# this file contain a FBSSVEPFormer method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
import torch.nn as nn
import torch

class SSVEPFormer(nn.Module):
    def __init__(self, n_channels=9, dropout_rate=0.5, sample_length=280*2, class_num=40):
        super(SSVEPFormer, self).__init__()
        # 定义卷积层和池化层
        self.channel_combination = nn.Sequential(
                            nn.Conv1d(in_channels=n_channels, out_channels=2*n_channels, kernel_size=1),
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.GELU(),  
                            nn.Dropout(dropout_rate))
        
        self.subencoder_cnn1 = nn.Sequential(
                            nn.LayerNorm(sample_length),
                            nn.Conv1d(in_channels=2*n_channels, out_channels=2*n_channels, kernel_size=31, padding=15),
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.GELU(),
                            nn.Dropout(dropout_rate))

        self.subencoder_mlp1 = nn.Sequential(
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.Linear(in_features=sample_length, out_features=sample_length),
                            nn.GELU(),
                            nn.Dropout(dropout_rate))

        self.subencoder_cnn2 = nn.Sequential(
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.Conv1d(in_channels=2 * n_channels, out_channels=2 * n_channels, kernel_size=31, padding=15),
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.GELU(),
                            nn.Dropout(dropout_rate))

        self.subencoder_mlp2 = nn.Sequential(
                            nn.LayerNorm([2 * n_channels, sample_length]),
                            nn.Linear(in_features=sample_length, out_features=sample_length),
                            nn.GELU(),
                            nn.Dropout(dropout_rate))
        
        self.mlp = nn.Sequential(
                            nn.Flatten(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(in_features=2 * n_channels * sample_length, out_features=6 * class_num),
                            nn.LayerNorm(6 * class_num),
                            nn.GELU(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(in_features=6 * class_num, out_features=class_num))


    def forward(self, x):
        x = torch.fft.rfft(x, n=1250, dim=-1)
        real = torch.real(x[:, :, 40:320])
        imag = torch.imag(x[:, :, 40:320])
        x = torch.cat((real, imag), dim=2)
        # 前向传播
        x = self.channel_combination(x)
        x = x + self.subencoder_cnn1(x)
        x = x + self.subencoder_mlp1(x)
        x = x + self.subencoder_cnn2(x)
        x = x + self.subencoder_mlp2(x)
        x = self.mlp(x)
        return x


class FBSSVEPFormer(nn.Module):
    def __init__(self, subbands=3, n_channels=9, dropout_rate=0.5, sample_length=560, class_num=40):
        super(FBSSVEPFormer, self).__init__()
        # 定义卷积层和池化层
        self.layer = nn.ModuleList([SSVEPFormer(n_channels=n_channels, dropout_rate=dropout_rate, sample_length=sample_length, class_num=class_num) for _ in range(subbands)])
        self.convfuse = nn.Conv1d(in_channels=subbands, out_channels=1, kernel_size=1)

    def forward(self, x):
        # 前向传播
        output_layers = []
        for idx, layer in enumerate(self.layer):
            output_layers.append(layer(x[:, idx, :, :]).unsqueeze(1))
        x = torch.cat(output_layers, dim=1)
        x = self.convfuse(x)
        return x.squeeze(1)

if __name__ == '__main__':
    input = torch.randn(64, 3, 9, 250)
    ### mobilevit_xxs
    model1 = FBSSVEPFormer()
    out = model1(input)
    print(out.shape)
