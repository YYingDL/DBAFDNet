# this file contain a FBDBACFNet method to decode SSVEP singles
# created by Yang Yi, University of Macau, 2023.09.05
# email: yc27932@umac.mo
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, return_att=True):
        b, c, _ = x.size()
        residual = x

        avg_result = self.avgpool(x)
        avg_out = self.se(avg_result)
        channel_att = self.sigmoid(avg_out)

        out = x * channel_att
        if return_att:
            return out + residual, channel_att
        return out + residual

class PSA(nn.Module):

    def __init__(self, channel=280, reduction=4, S=7):
        super().__init__()
        self.S = S


        self.convs = nn.ModuleList()
        for i in range(S):
            psa_conv = nn.Conv1d(channel // S, channel // S, kernel_size=2 * (i) + 1, padding= i)
            self.convs.append(psa_conv)

        self.se_blocks = nn.ModuleList()
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        b, c, h= x.size()

        # Step1:SPC module
        x = x.view(b, self.S, c // self.S, h)  # bs,s,ci,h,w
        SPC_out = []
        for idx, conv in enumerate(self.convs):
            SPC_out.append(conv(x[:, idx, :,  :]))
        SPC_out = torch.stack(SPC_out, dim=1)


        # Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h)

        return PSA_out, softmax_out


class decoder(nn.Module):
    def __init__(self,  channel=9, hidden_channels=64, out_channels=16, input_dim=280, dropout_rate=0.5, class_num = 40):
        super(decoder, self).__init__()
        self.channel_attention = ChannelAttention(channel=channel)
        self.mlp_1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=hidden_channels, kernel_size=1),
                                    nn.LayerNorm([input_dim]),
                                    nn.GELU(),
                                    nn.Dropout(dropout_rate))

        self.psa = PSA(channel=hidden_channels, reduction=4, S=8)
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1),
                                nn.LayerNorm([input_dim]),
                                nn.GELU(),
                                nn.Dropout(dropout_rate))

    def forward(self, input):
        x, channel_att = self.channel_attention(input)
        x = self.mlp_1(x)
        x, psa_att = self.psa(x)
        x = self.mlp(x)
        return x.transpose(1, 2), channel_att, psa_att


class DBACFNet(nn.Module):
    def __init__(self,  channel=9, sample_length=280, dropout_rate=0.4, class_num=40):
        super(DBACFNet, self).__init__()
        self.decoder_amp = decoder()
        self.decoder_phase = decoder()
        self.share_decoder_amp = decoder()
        self.share_decoder_phase = decoder()
        self.fc_share = nn.Sequential(
                            nn.Flatten(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(in_features=4 * 16 * sample_length, out_features=6 * class_num),
                            nn.LayerNorm(6 * class_num),
                            nn.GELU(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(in_features=6 * class_num, out_features=class_num))


    def forward(self, fft_amp, fft_phase):
        amp, _, _ = self.decoder_amp(fft_amp)
        phase, _, _ = self.decoder_phase(fft_phase)

        shared_amp, channel_amp, psa_amp = self.share_decoder_amp(fft_amp)
        shared_phase, channel_phase,  psa_phase = self.share_decoder_phase(fft_phase)

        out = self.fc_share(torch.concat([shared_amp, shared_phase, amp, phase], dim=-1))
        # return out
        return out, [amp, shared_amp], [phase,shared_phase],  [channel_amp, channel_phase], [psa_amp, psa_phase]

class FBDBACFNet(nn.Module):
    def __init__(self,  channel=9, sample_length=280, dropout_rate=0.4, class_num=40):
        super(FBDBACFNet, self).__init__()
        self.subband_1 = DBACFNet()
        self.subband_2 = DBACFNet()
        self.subband_3 = DBACFNet()
        self.convfuse = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):
        fft_data = torch.fft.rfft(x, n=1250, axis=-1)

        fft_amp = torch.real(fft_data[:, :, :, 40:320])
        fft_phase = torch.imag(fft_data[:, :, :, 40:320])

        out_1, amp_feat_output_1, phase_feat_output_1, att1_1, att2_1 = self.subband_1(fft_amp[:, 0], fft_phase[:, 0])
        out_2, amp_feat_output_2, phase_feat_output_2, att1_2, att2_2 = self.subband_2(fft_amp[:, 1], fft_phase[:, 1])
        out_3, amp_feat_output_3, phase_feat_output_3, att1_3, att2_3= self.subband_3(fft_amp[:, 2], fft_phase[:, 2])
        out = torch.concat([out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)

        amp_feat = [amp_feat_output_1, amp_feat_output_2, amp_feat_output_3]
        phase_feat = [phase_feat_output_1, phase_feat_output_2, phase_feat_output_3]
        att1 = [att1_1, att1_2, att1_3]
        att2 = [att2_1, att2_2, att2_3]
        out = self.convfuse(out)
        # return out
        return out.squeeze(1), amp_feat, phase_feat, att1, att2

class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()

    # Should be orthogonal
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    # should be big
    @staticmethod
    def dot_product_normalize(shared_1, shared_2):
        diff = shared_1 - shared_2
        loss = torch.mean(torch.abs(diff ** 2))
        return loss

    def forward(self, output, target):
        classification_output, amp_feat, phase_feat, att1, att2= output

        # similarity Loss
        similarity_loss = 0
        for tmp in att1:
            similarity_loss += self.dot_product_normalize(tmp[0], tmp[1])
        for tmp in att2:
            similarity_loss += self.dot_product_normalize(tmp[0], tmp[1])

        # orthogonal_loss
        orthogonal_loss = 0
        for tmp in amp_feat:
            orthogonal_loss += self.orthogonal_loss(tmp[0], tmp[1])
        for tmp in phase_feat:
            orthogonal_loss += self.orthogonal_loss(tmp[0], tmp[1])

        # Classification Loss
        classification_loss_1 = F.cross_entropy(classification_output, target)

        loss = classification_loss_1 + 0.2 * similarity_loss + 0.2 * orthogonal_loss
        return loss


if __name__ == '__main__':
    input = torch.randn(50, 3, 9, 280)
    cbam = FBDBACFNet()
    output= cbam(input)
    print(output[0].shape)

# this is a test file to validate the model have different dropout rate with the same structure
# we have revised the dropout rate in the model: 0.5 to 0.2
# we have get a result to show, the model with dropout rate 0.2 is less than the model with dropout rate 0.5
# accodring to the loss cruve, we can see the model with dropout rate 0.2 is more difficult to get a stable model
# the benchamrk dataset sub.01 : 0.9458 (0.2) 0.950 (0.5) sub.02 : 0.9417 (0.2) 0.950 (0.5)
# 0.8 not good  sub.01 : 0.87 (0.8)