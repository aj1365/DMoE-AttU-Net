import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(64, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.softmax(self.fc2(x))


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_channels = max(in_channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc2(F.relu(self.fc1(y))).view(b, c, 1, 1)
        return x * torch.sigmoid(y)


class CNNExpert(nn.Module):
    def __init__(self, in_ch, dropout_p=0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_p),
        )

        self.attn = SEAttention(64)
        self.out = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.attn(x)
        return self.out(x)


class SARNoiseResilientEncoder(nn.Module):
    def __init__(self, in_ch, num_experts=3, dropout_p=0.2, input_size=(256, 256)):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout_p),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, input_size[0], input_size[1])
            dummy_out = self.shared(dummy)
            feature_dim = dummy_out.view(1, -1).shape[1]

        self.gating = GatingNetwork(feature_dim, num_experts, dropout_p=dropout_p)
        self.experts = nn.ModuleList(
            [CNNExpert(64, dropout_p=dropout_p) for _ in range(num_experts)]
        )

    def forward(self, x):
        x = self.shared(x)

        weights = self.gating(x.view(x.size(0), -1))

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=1
        )

        weights = weights.view(x.size(0), -1, 1, 1, 1)
        combined = (weights * expert_outputs).sum(dim=1)

        return combined, expert_outputs


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_p)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels_optical=3,
        in_channels_sar=1,
        out_channels=2,
        num_experts=3,
        dropout_p=0.3,
        input_size=(256, 256),
    ):
        super().__init__()

        self.optical_conv1 = ResidualBlock(in_channels_optical, 64, dropout_p)
        self.optical_pool1 = nn.MaxPool2d(kernel_size=2)

        self.optical_conv2 = ResidualBlock(64, 128, dropout_p)
        self.optical_pool2 = nn.MaxPool2d(kernel_size=2)

        self.sar_enc = SARNoiseResilientEncoder(
            in_ch=in_channels_sar,
            num_experts=num_experts,
            dropout_p=dropout_p,
            input_size=input_size,
        )

        self.bottleneck = ResidualBlock(128 + 64, 256, dropout_p)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att_gate1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder1 = ResidualBlock(128 + 128, 128, dropout_p)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att_gate2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.decoder2 = ResidualBlock(64 + 64, 64, dropout_p)

        self.final_out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, optical, sar):
        x1 = self.optical_conv1(optical)
        p1 = self.optical_pool1(x1)

        x2 = self.optical_conv2(p1)
        p2 = self.optical_pool2(x2)

        expert_fused, expert_outputs = self.sar_enc(sar)

        expert_fused = F.interpolate(
            expert_fused,
            size=p2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([p2, expert_fused], dim=1)
        x = self.bottleneck(x)

        x = self.up1(x)
        g1 = self.att_gate1(g=x, x=x2)
        x = torch.cat([x, g1], dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        g2 = self.att_gate2(g=x, x=x1)
        x = torch.cat([x, g2], dim=1)
        x = self.decoder2(x)

        seg_out = self.final_out(x)

        return seg_out, expert_outputs


def build_model(
    in_channels_optical=3,
    in_channels_sar=1,
    out_channels=2,
    num_experts=3,
    dropout_p=0.3,
    input_size=(256, 256),
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUNet(
        in_channels_optical=in_channels_optical,
        in_channels_sar=in_channels_sar,
        out_channels=out_channels,
        num_experts=num_experts,
        dropout_p=dropout_p,
        input_size=input_size,
    ).to(device)

    return model
