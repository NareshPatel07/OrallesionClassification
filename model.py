# model.py
# Transforms, UNet denoiser (for DDPM), DDPM wrapper, EfficientNet classifier

import torch
import torch.nn as nn
from torchvision import transforms, models

# ======================================================
# TRANSFORMS
# ======================================================

def get_ddpm_train_transform(image_size=64):
    """Transform for DDPM training: images in [-1,1]."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1,1]
    ])

def get_eval_transform_ddpm(image_size=64):
    """Transform for DDPM inference / denoising pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def get_eval_transform(image_size=224):
    """Transform for classifier evaluation / inference."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


def get_classifier_train_transform(image_size=224):
    """Transform for classifier training with augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


# ======================================================
# UNET DENOISER
# ======================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """
    Lightweight UNet for DDPM noise prediction.
    Input: x_t, (optionally timestep t, currently ignored in blocks)
    Output: predicted noise eps_hat.
    """
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)          # 3 -> 64
        self.enc2 = DoubleConv(base_ch, base_ch * 2)    # 64 -> 128
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)  # 128 -> 256

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)  # concat with x2

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)      # concat with x1

        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t=None):
        # t (timestep) is currently ignored but kept for DDPM interface
        x1 = self.enc1(x)              # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))  # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))  # [B, 256, H/4, W/4]

        x = self.up2(x3)               # [B, 128, H/2, W/2]
        x = torch.cat([x, x2], dim=1)  # [B, 256, H/2, W/2]
        x = self.dec2(x)               # [B, 128, H/2, W/2]

        x = self.up1(x)                # [B, 64, H, W]
        x = torch.cat([x, x1], dim=1)  # [B, 128, H, W]
        x = self.dec1(x)               # [B, 64, H, W]

        return self.out(x)             # predict noise: [B, 3, H, W]


# ======================================================
# DDPM WRAPPER
# ======================================================

class DDPM:
    """
    Minimal DDPM helper. Trains UNet to predict noise eps,
    uses p_sample_from() to denoise from x_t backwards.
    """
    def __init__(self, timesteps=200, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        # linear beta schedule
        self.beta = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_cum = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: q(x_t | x0).
        x0 in [-1,1], t in [0..T-1], noise ~ N(0,I)
        """
        ac = self.alpha_cum[t].view(-1, 1, 1, 1)  # [B,1,1,1]
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    @torch.no_grad()
    def p_sample_step(self, model, x, i):
        """
        Single reverse diffusion step:
        p(x_{t-1} | x_t).
        """
        beta = self.beta[i]
        alpha = self.alpha[i]
        alpha_cum = self.alpha_cum[i]

        B = x.size(0)
        t = torch.full((B,), i, device=self.device, dtype=torch.long)

        # Predict noise eps_theta(x_t, t)
        eps_hat = model(x, t)

        # Reverse update (DDPM)
        x = (1.0 / alpha.sqrt()) * (
            x - (beta / (1.0 - alpha_cum).sqrt()) * eps_hat
        )

        if i > 0:
            x = x + beta.sqrt() * torch.randn_like(x)

        return x

    @torch.no_grad()
    def p_sample_from(self, model, x_t, start_t, steps):
        """
        Run several reverse steps starting from x_t at time 'start_t'
        going backwards for 'steps' steps (or until t=0).
        """
        x = x_t
        t_cur = min(start_t, self.timesteps - 1)
        t_end = max(t_cur - steps, 0)
        for i in range(t_cur, t_end - 1, -1):
            x = self.p_sample_step(model, x, i)
        return x


# ======================================================
# EFFICIENTNET CLASSIFIER
# ======================================================

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)
