"""
Latent Mapping Network
Maps 512-dim CLIP text embeddings to StyleGAN2 W latent space (512-dim).
Small MLP with residual-like structure.
"""

import torch
import torch.nn as nn
import os


class LatentMapper(nn.Module):
    """MLP that maps CLIP text embeddings to StyleGAN2 W latent vectors."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Map text embedding to StyleGAN2 latent vector.

        Args:
            text_embedding: Tensor of shape (batch, 512) or (512,)

        Returns:
            Latent vector of shape (batch, 512) or (512,)
        """
        squeeze = False
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
            squeeze = True

        latent = self.network(text_embedding)

        if squeeze:
            latent = latent.squeeze(0)

        return latent

    def save_weights(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[LatentMapper] Weights saved to {path}")

    def load_weights(self, path: str):
        """Load pretrained weights if available."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"[LatentMapper] Weights loaded from {path}")
            return True
        print(f"[LatentMapper] No weights found at {path} — using random init (demo mode)")
        return False
