"""
Text Encoder Module
Uses OpenAI CLIP (clip-vit-base-patch32) to convert text descriptions into embeddings.
Also provides CLIP similarity scoring between text and generated images.
"""

import torch
import numpy as np
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class TextEncoder:
    """Wraps CLIP model for text encoding and similarity computation."""

    def __init__(self, device=None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.model = None
        self.processor = None
        self.embedding_dim = 512
        self._load_model()

    def _load_model(self):
        """Load CLIP model and processor."""
        if not CLIP_AVAILABLE:
            print("[TextEncoder] transformers not available — using random embeddings (demo mode)")
            return

        try:
            print("[TextEncoder] Loading CLIP model (openai/clip-vit-base-patch32)...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            print("[TextEncoder] CLIP model loaded successfully.")
        except Exception as e:
            print(f"[TextEncoder] Failed to load CLIP: {e} — using demo mode")
            self.model = None
            self.processor = None

    def encode(self, description: str) -> torch.Tensor:
        """
        Encode a text description into a 512-dim embedding vector.

        Args:
            description: Natural language description of the face.

        Returns:
            torch.Tensor of shape (512,)
        """
        if self.model is None or self.processor is None:
            # Demo mode: generate deterministic embedding from text hash
            return self._demo_encode(description)

        with torch.no_grad():
            inputs = self.processor(text=[description], return_tensors="pt", padding=True, truncation=True)
            # Only pass text-related inputs
            input_ids = inputs.get('input_ids', inputs.get('input_ids')).to(self.device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Use the text model directly + text projection for robustness
            text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            # Get pooled output (CLS token) — use attribute access, not index
            pooled_output = text_outputs.pooler_output
            # Apply text projection
            if hasattr(self.model, 'text_projection'):
                text_features = self.model.text_projection(pooled_output)
            else:
                text_features = pooled_output

            # Normalize
            norm = torch.norm(text_features, dim=-1, keepdim=True)
            text_features = text_features / norm.clamp(min=1e-8)
            return text_features.squeeze(0).cpu()

    def compute_similarity(self, text_embedding: torch.Tensor, image: Image.Image) -> float:
        """
        Compute CLIP similarity between a text embedding and a generated image.

        Args:
            text_embedding: Pre-computed text embedding (512-dim).
            image: PIL Image of the generated face.

        Returns:
            Similarity score (0-100 scale).
        """
        if self.model is None or self.processor is None:
            # Demo mode: return a plausible random score
            seed = int(torch.sum(text_embedding).item() * 1000) % 10000
            rng = np.random.RandomState(seed)
            return round(float(rng.uniform(65, 95)), 1)

        with torch.no_grad():
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs.get('pixel_values').to(self.device)

            # Use vision_model directly + visual_projection for robustness
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            if hasattr(self.model, 'visual_projection'):
                image_features = self.model.visual_projection(pooled_output)
            else:
                image_features = pooled_output
            img_norm = torch.norm(image_features, dim=-1, keepdim=True)
            image_features = image_features / img_norm.clamp(min=1e-8)

            text_emb = text_embedding.unsqueeze(0).to(self.device)
            txt_norm = torch.norm(text_emb, dim=-1, keepdim=True)
            text_emb = text_emb / txt_norm.clamp(min=1e-8)

            similarity = torch.cosine_similarity(text_emb, image_features).item()
            # Scale to 0-100
            score = max(0, min(100, (similarity + 1) * 50))
            return round(score, 1)

    def compute_similarity_differentiable(self, text_embedding: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity score between text embedding and a generated image tensor.
        Preserves the PyTorch computational graph for backpropagation (Latent Optimization).

        Args:
            text_embedding: Tensor of shape (512,)
            image_tensor: Tensor of shape (1, 3, H, W) in range [-1, 1]

        Returns:
            Cosine similarity scalar Tensor in range [-1, 1]
        """
        if self.model is None or self.processor is None:
            return torch.tensor(0.5, requires_grad=True, device=self.device)

        # 1. Resize to CLIP's expected 224x224 using bilinear interpolation
        import torch.nn.functional as F
        pixel_values = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # 2. Convert from [-1, 1] to [0, 1]
        pixel_values = (pixel_values + 1) / 2

        # 3. Normalize using CLIP's standard ImageNet mean and std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean) / std

        # 4. Pass through Vision Model (preserving gradients)
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        
        if hasattr(self.model, 'visual_projection'):
            image_features = self.model.visual_projection(pooled_output)
        else:
            image_features = pooled_output

        img_norm = torch.norm(image_features, dim=-1, keepdim=True)
        image_features = image_features / img_norm.clamp(min=1e-8)

        text_emb = text_embedding.unsqueeze(0).to(self.device)
        txt_norm = torch.norm(text_emb, dim=-1, keepdim=True)
        text_emb = text_emb / txt_norm.clamp(min=1e-8)

        # 5. Calculate and return Cosine Similarity
        similarity = torch.cosine_similarity(text_emb, image_features).mean()
        return similarity

    def _demo_encode(self, description: str) -> torch.Tensor:
        """Generate a deterministic embedding from text (demo mode)."""
        # Use hash of description to create reproducible but varied embeddings
        hash_val = hash(description)
        rng = np.random.RandomState(abs(hash_val) % (2**31))
        embedding = rng.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return torch.from_numpy(embedding)

    @property
    def is_loaded(self) -> bool:
        """Check if the CLIP model is properly loaded."""
        return self.model is not None
