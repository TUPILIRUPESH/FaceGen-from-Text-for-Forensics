"""
StyleGAN2 Generator Architecture
Based on rosinality/stylegan2-pytorch implementation.
Implements the full StyleGAN2 generator for photorealistic face generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ─────────────────────────────────────────────────

class PixelNorm(nn.Module):
    """Pixelwise feature vector normalization."""
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale,
                       bias=self.bias * self.lr_mul if self.bias is not None else None)
        if self.activation == 'fused_lrelu':
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class EqualConv2d(nn.Module):
    """Conv2d with equalized learning rate."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, bias=self.bias,
                        stride=self.stride, padding=self.padding)


class ModulatedConv2d(nn.Module):
    """
    Modulated convolution — the core StyleGAN2 innovation.
    Applies style-based modulation to conv weights.
    """
    def __init__(self, in_ch, out_ch, kernel_size, style_dim, demodulate=True, upsample=False):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        self.demodulate = demodulate

        fan_in = in_ch * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1)

    def forward(self, x, style):
        batch, in_ch, h, w = x.shape

        # Modulate weights with style
        style = self.modulation(style).view(batch, 1, in_ch, 1, 1)
        weight = self.scale * self.weight * style

        # Demodulate
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_ch, 1, 1, 1)

        weight = weight.view(batch * self.out_ch, in_ch, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = x.view(1, batch * in_ch, h, w)
            # Reshape back to 5D to transpose channels safely
            weight = weight.view(batch, self.out_ch, in_ch, self.kernel_size, self.kernel_size)
            # Transpose out_ch and in_ch -> (batch, in_ch, out_ch, k, k)
            weight = weight.transpose(1, 2).reshape(
                batch * in_ch, self.out_ch, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, h2, w2 = out.shape
            out = out.view(batch, self.out_ch, h2, w2)
            # Blur after upsample
            out = self._blur(out)
        else:
            x = x.view(1, batch * in_ch, h, w)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch)
            _, _, h2, w2 = out.shape
            out = out.view(batch, self.out_ch, h2, w2)

        return out

    def _blur(self, x):
        """Simple blur filter after upsampling."""
        kernel = torch.tensor([[1, 3, 3, 1]], dtype=x.dtype, device=x.device)
        kernel = kernel.T @ kernel
        kernel = kernel / kernel.sum()
        kernel = kernel * 4  # scale by upsample_factor ** 2 to preserve energy
        kernel = kernel.view(1, 1, 4, 4).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, stride=1, padding=1, groups=x.shape[1])


class NoiseInjection(nn.Module):
    """Injects per-pixel noise scaled by a learned parameter."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, h, w = x.shape
            noise = x.new_empty(batch, 1, h, w).normal_()
        return x + self.weight * noise


class StyledConv(nn.Module):
    """Styled convolution block: ModulatedConv2d + Noise + Activation."""
    def __init__(self, in_ch, out_ch, kernel_size, style_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, style_dim, upsample=upsample)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = out + self.bias
        out = self.noise(out, noise=noise)
        out = F.leaky_relu(out, 0.2) * math.sqrt(2)
        return out


class ToRGB(nn.Module):
    """Converts feature maps to RGB image using modulated 1x1 conv."""
    def __init__(self, in_ch, style_dim, upsample=True):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsample = None
        self.conv = ModulatedConv2d(in_ch, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias

        if skip is not None and self.upsample is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConstantInput(nn.Module):
    """Learned constant input (4x4)."""
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch_size):
        return self.input.repeat(batch_size, 1, 1, 1)


# ─── Full Generator ──────────────────────────────────────────────────

class StyleGAN2GeneratorFull(nn.Module):
    """
    Full StyleGAN2 Generator.
    Mapping Network (z → w) + Synthesis Network (w → image).
    """

    def __init__(self, size=1024, style_dim=512, n_mlp=8, channel_multiplier=2):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        # Channel configuration per resolution
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # Mapping network: z → w
        mapping_layers = [PixelNorm()]
        for i in range(n_mlp):
            mapping_layers.append(
                EqualLinear(style_dim, style_dim, lr_mul=0.01, activation='fused_lrelu')
            )
        self.style = nn.Sequential(*mapping_layers)

        # Synthesis network
        self.input = ConstantInput(channels[4])

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        # Register noise buffers
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        # Initial conv at 4x4
        self.conv1 = StyledConv(channels[4], channels[4], 3, style_dim)
        self.to_rgb1 = ToRGB(channels[4], style_dim, upsample=False)

        in_ch = channels[4]

        # Progressive upsampling layers
        for i in range(3, self.log_size + 1):
            out_ch = channels[2 ** i]

            self.convs.append(StyledConv(in_ch, out_ch, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_ch, out_ch, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_ch, style_dim))

            in_ch = out_ch

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        """Generate random noise tensors for all layers."""
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_latent(self, n_latent):
        """Compute the mean W latent vector."""
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in)
        return latent.mean(0, keepdim=True)

    def forward(self, styles, input_is_latent=False, noise=None,
                randomize_noise=True, truncation=1, truncation_latent=None):
        """
        Generate images from latent codes.

        Args:
            styles: list of latent tensors or single tensor
            input_is_latent: if True, skip mapping network (styles are W vectors)
            noise: optional list of noise tensors
            randomize_noise: if True, generate random noise
            truncation: truncation ratio (1 = no truncation)
            truncation_latent: mean W vector for truncation

        Returns:
            Generated image tensor (B, 3, H, W) in [-1, 1] range
        """
        if not isinstance(styles, list):
            styles = [styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        # Map z → w (or use w directly)
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        # Truncation trick
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t

        # Broadcast to all layers
        if len(styles) == 1:
            latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)
        elif len(styles) == 2:
            # Style mixing
            inject_index = self.n_latent // 2
            latent = torch.cat([
                styles[0].unsqueeze(1).repeat(1, inject_index, 1),
                styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            ], dim=1)
        else:
            latent = torch.stack(styles, dim=1)

        # Synthesis
        out = self.input(latent.shape[0])
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[i])
            out = conv2(out, latent[:, i + 1], noise=noise[i + 1])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        return image
