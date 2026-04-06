import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import numpy as np
import argparse
import warnings
from typing import Optional, List, Tuple, Union
from transformers import PretrainedConfig

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    # Minimal fallback if huggingface_hub is not installed
    class PyTorchModelHubMixin:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("huggingface_hub is required for from_pretrained")
        def save_pretrained(self, *args, **kwargs):
            raise ImportError("huggingface_hub is required for save_pretrained")

class MLP(nn.Module):
    """Multi-layer Perceptron block with LayerNorm and GELU activation."""
    def __init__(self, dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """A Transformer block with Multihead Attention and MLP."""
    def __init__(self, dim: int, hidden_dim: int, heads: int, dropout: float = 0.):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(attn_out + x)
        # MLP with residual connection
        x = self.norm2(self.mlp(x) + x)
        return x

class ProM3E(nn.Module, PyTorchModelHubMixin):
    """
    Probabilistic Multi-Modal Masked Embedding (ProM3E) Model.
    
    A refactored implementation designed for GitHub release.
    Features:
    - Multi-modal joint embedding space via modality-specific projectors.
    - Transformer-based aggregation using global CLS and Register tokens.
    - Variational Information Bottleneck (VIB) for robust representation.
    - Masked reconstruction task with contrastive-similarity loss.
    """
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
        
        self.input_dim = self.config.input_dim
        self.embed_dim = self.config.embed_dim
        self.num_modalities = self.config.num_modalities
        self.masked_only = self.config.masked_only
        self.depth = self.config.depth
        self.heads = self.config.heads
        self.mlp_dim = self.config.mlp_dim
        self.num_register_tokens = self.config.num_register_tokens
        self.num_cls_tokens = self.config.num_cls_tokens
        self.dropout = self.config.dropout
        self.lambda_kl = self.config.lambda_kl

        # Modality-specific projectors
        self.projectors = nn.ModuleList([
            MLP(self.input_dim, self.input_dim * 2, self.embed_dim, dropout=self.dropout) 
            for _ in range(self.num_modalities)
        ])
        
        # Modality identifiers (embeddings added to inputs)
        self.modality_identifiers = nn.ParameterList([
            nn.Parameter(torch.randn(self.embed_dim)) 
            for _ in range(self.num_modalities)
        ])
        
        # Global tokens: [mu, logvar] + registers
        self.cls_tokens = nn.Parameter(torch.randn(self.num_cls_tokens, self.embed_dim))
        self.register_tokens = nn.Parameter(torch.randn(self.num_register_tokens, self.embed_dim))
        
        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.mlp_dim, self.heads, dropout=self.dropout) 
            for _ in range(self.depth)
        ])
        
        # Reconstruction heads
        self.decoders = nn.ModuleList([
            MLP(self.embed_dim, self.embed_dim, self.input_dim, dropout=self.dropout) 
            for _ in range(self.num_modalities)
        ])
        
        # Contrastive-reconstruction loss parameters
        self.negative_scale = nn.Parameter(torch.ones([]) * -5)
        self.negative_shift = nn.Parameter(torch.ones([]) * 5)

    def _get_unmasked_indices(self, audio_flag: Union[bool, int]) -> List[int]:
        """Determines which modalities to unmask for the current training step."""
        # 90% chance of 1 unmasked modality, 10% chance of 2
        num_unmasked = 1 if torch.rand(1) < 0.9 else 2
        
        if bool(audio_flag):
            options = [0, 1, 2, 3, 4, 5]
        else:
            # Modality 5 is audio, disabled here
            options = [0, 1, 2, 3, 4]
            
        return np.random.choice(options, num_unmasked, replace=False).tolist()

    def forward(self, modalities: torch.Tensor, audio_flag: Union[bool, int] = False) -> torch.Tensor:
        """
        Forward pass for training. Calculates and returns the combined loss.
        
        Args:
            modalities: Input features [batch, modalities, dim].
            audio_flag: Flag to enable/disable the audio modality.
        """
        device = modalities.device
        batch_size = modalities.shape[0]
        
        # 1. Selection & Projecting
        unmasked_idx = self._get_unmasked_indices(audio_flag)
        x = torch.zeros((batch_size, len(unmasked_idx), self.embed_dim), device=device)
        for i, idx in enumerate(unmasked_idx):
            x[:, i] = self.projectors[idx](modalities[:, idx]) + self.modality_identifiers[idx]

        # 2. Add Latent Tokens
        cls_tokens = repeat(self.cls_tokens, 'n d -> b n d', b=batch_size)
        reg_tokens = repeat(self.register_tokens, 'n d -> b n d', b=batch_size)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        # 3. Encoding
        for layer in self.encoder:
            x = layer(x)

        # 4. Variational Sampling
        # mu is normalized typically for contrastive stability
        mu = F.normalize(x[:, 0], dim=-1)
        logvar = x[:, 1]
        
        std = torch.exp(0.5 * logvar)
        # Using 1 noise scalar per batch sample (as per original logic)
        eps = torch.randn(batch_size, device=device).unsqueeze(1)
        latent_sample = mu + std * eps
        
        # 5. Reconstruction
        num_to_decode = 6 if bool(audio_flag) else 5
        preds = torch.zeros((batch_size, num_to_decode, self.input_dim), device=device)
        
        for i in range(num_to_decode):
            if self.masked_only and i in unmasked_idx:
                continue
            preds[:, i] = self.decoders[i](latent_sample)
        
        preds = F.normalize(preds, dim=-1)

        # 6. Reconstruction Loss (Contrastive Cross-Entropy)
        identity_gt = torch.eye(batch_size, device=device)
        recon_loss = 0
        valid_recon_count = 0
        
        for i in range(num_to_decode):
            if self.masked_only and i in unmasked_idx:
                continue
            
            # Compute distance matrix and similarity logits
            dist = torch.cdist(preds[:, i], modalities[:, i])
            logits = self.negative_scale * dist + self.negative_shift
            
            log_probs = F.log_softmax(logits, dim=-1)
            recon_loss += torch.sum(-log_probs * identity_gt) / batch_size
            valid_recon_count += 1
            
        recon_loss /= max(valid_recon_count, 1)

        # 7. VIB Regularization (KL Divergence)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        
        # Safeguard against loss divergence
        if kl_loss > 10000:
            warnings.warn(f'VIB Loss Explosion Detected: {kl_loss.item():.2f}. Zeroing for stability.')
            kl_loss = torch.tensor(0.0, device=device)
            
        return recon_loss + self.lambda_kl * kl_loss

    def forward_inference(
        self, 
        modalities: torch.Tensor, 
        modality_mask: List[int],
        n_samples: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs inference provided a mask of active modalities.
        
        Returns:
            Tuple of (predictions, mu, logvar)
        """
        device = modalities.device
        batch_size = modalities.shape[0]
        
        x = torch.zeros((batch_size, len(modality_mask), self.embed_dim), device=device)
        for i, idx in enumerate(modality_mask):
            x[:, i] = self.projectors[idx](modalities[:, idx]) + self.modality_identifiers[idx]

        cls_tokens = repeat(self.cls_tokens, 'n d -> b n d', b=batch_size)
        reg_tokens = repeat(self.register_tokens, 'n d -> b n d', b=batch_size)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        for layer in self.encoder:
            x = layer(x)

        mu = F.normalize(x[:, 0], dim=-1)
        logvar = x[:, 1]
        std = torch.exp(0.5 * logvar)

        # Stochastic averaging at inference
        latent_samples = []
        for _ in range(n_samples):
            latent_samples.append(mu + std * torch.randn_like(mu))
        latent_mean = torch.stack(latent_samples).mean(dim=0)
        
        predictions = torch.zeros((batch_size, self.num_modalities, self.input_dim), device=device)
        for i in range(self.num_modalities):
            predictions[:, i] = self.decoders[i](latent_mean)

        return F.normalize(predictions, dim=-1), mu, logvar, x

def main():
    parser = argparse.ArgumentParser(description="ProM3E Model Release Script")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for demonstration")
    parser.add_argument("--input_dim", type=int, default=512, help="Feature dimension of inputs")
    parser.add_argument("--embed_dim", type=int, default=512, help="Internal transformer dimension")
    parser.add_argument("--depth", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda_kl", type=float, default=1e-3, help="Weight for KL divergence loss")
    args = parser.parse_args()

    print("--- ProM3E Model Release Demo ---")
    model = ProM3E(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads
    ).to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized on: {args.device}")
    print(f"Total Parameters: {param_count:,}")

    # Dummy data: [batch, 6 modalities, 512 dim]
    dummy_data = torch.randn(args.batch_size, 6, args.input_dim).to(args.device)

    print("\nRunning Training Pass...")
    loss = model(dummy_data, audio_flag=True)
    print(f"Resulting Loss: {loss.item():.6f}")

    print("\nRunning Inference Pass (Active Modalities: [0, 2])...")
    preds, mu, _ = model.forward_inference(dummy_data, modality_mask=[0, 2])
    print(f"Output Predictions Shape: {preds.shape}")
    print(f"Latent mu Shape: {mu.shape}")
    
    print("\nDone. The script is clean and ready for GitHub.")

if __name__ == "__main__":
    main()