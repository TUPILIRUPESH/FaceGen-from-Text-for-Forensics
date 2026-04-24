"""
StyleGAN2 Generator Module
Generates face images from latent vectors.
Supports full pretrained StyleGAN2 model OR procedural demo mode.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math


class StyleGAN2Generator:
    """
    StyleGAN2 face generator.
    - With pretrained weights: generates photorealistic faces
    - Without weights: procedural face composites (demo mode)
    """

    def __init__(self, weights_path=None, device=None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.resolution = 256  # Output resolution (even if model generates 1024)
        self.demo_mode = True
        self.model = None
        self.mean_latent = None

        # Try to find weights
        if weights_path is None:
            # Look for weights in the default location (root/weights)
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(root_dir, 'weights', 'stylegan2-ffhq-config-f.pt')
            if os.path.exists(default_path):
                weights_path = default_path

        if weights_path and os.path.exists(weights_path):
            self._load_full_model(weights_path)

    def _load_full_model(self, weights_path):
        """Load the full StyleGAN2 generator with pretrained weights."""
        try:
            print(f"[StyleGAN2] Loading pretrained weights from {weights_path}...")

            # Import the full architecture
            from models.stylegan2_arch import StyleGAN2GeneratorFull

            # Load checkpoint
            ckpt = torch.load(weights_path, map_location='cpu')

            # Determine model config from checkpoint
            if isinstance(ckpt, dict) and 'g_ema' in ckpt:
                # rosinality format: {'g': ..., 'd': ..., 'g_ema': ...}
                state_dict = ckpt['g_ema']
                size = self._detect_resolution(state_dict)
            elif isinstance(ckpt, dict) and any('style' in k for k in ckpt.keys()):
                # Direct state dict
                state_dict = ckpt
                size = self._detect_resolution(state_dict)
            else:
                print(f"[StyleGAN2] Unknown checkpoint format. Keys: {list(ckpt.keys())[:10]}")
                return

            print(f"[StyleGAN2] Detected model resolution: {size}x{size}")

            # Map keys to match our custom architecture
            mapped_state_dict = {}
            for k, v in state_dict.items():
                if 'activate.bias' in k:
                    k = k.replace('.activate.bias', '.bias')
                    v = v.view(1, -1, 1, 1)  # Reshape bias for broadcasting
                elif 'conv.blur.kernel' in k:
                    continue
                # Reshape other biases from 1D to 4D where appropriate
                if k.endswith('.bias') and 'style' not in k and 'modulation' not in k and 'to_rgb' not in k:
                    if len(v.shape) == 1:
                        v = v.view(1, -1, 1, 1)
                mapped_state_dict[k] = v

            # Create model
            self.model = StyleGAN2GeneratorFull(
                size=size, style_dim=512, n_mlp=8, channel_multiplier=2
            )

            # Load weights
            missing, unexpected = self.model.load_state_dict(mapped_state_dict, strict=False)
            if missing:
                print(f"[StyleGAN2] Warning: missing keys {missing[:5]}")
            self.model.to(self.device)
            self.model.eval()

            # Compute mean latent for truncation trick
            with torch.no_grad():
                self.mean_latent = self.model.mean_latent(4096)

            self.demo_mode = False
            self.model_resolution = size
            print(f"[StyleGAN2] Full model loaded! Generating {size}x{size} -> {self.resolution}x{self.resolution} faces")

        except Exception as e:
            print(f"[StyleGAN2] Failed to load full model: {e}")
            print(f"[StyleGAN2] Falling back to demo mode.")
            import traceback
            traceback.print_exc()
            self.model = None
            self.demo_mode = True

    def _detect_resolution(self, state_dict):
        """Detect model resolution from state dict keys."""
        max_res = 4
        for key in state_dict.keys():
            if 'convs' in key and 'weight' in key:
                # Count conv layers to determine resolution
                try:
                    idx = int(key.split('.')[1])
                    res = 2 ** (idx // 2 + 3)
                    max_res = max(max_res, res)
                except (ValueError, IndexError):
                    pass
        return max(max_res, 256)

    def generate(self, latent_vector: torch.Tensor, variation_index: int = 0) -> Image.Image:
        """
        Generate a face image from a latent vector.

        Args:
            latent_vector: Tensor of shape (512,)
            variation_index: Index for generating variations

        Returns:
            PIL.Image of the generated face
        """
        if not self.demo_mode and self.model is not None:
            return self._full_generate(latent_vector, variation_index)
        return self._demo_generate(latent_vector, variation_index)

    def generate_variations(self, latent_vector: torch.Tensor, num_variations: int = 3) -> list:
        """
        Generate multiple face variations from a single latent vector.

        Args:
            latent_vector: Base latent vector (512,)
            num_variations: Number of variations

        Returns:
            List of PIL.Image objects
        """
        images = []
        for i in range(num_variations):
            img = self.generate(latent_vector, variation_index=i)
            images.append(img)
        return images

    def _full_generate(self, latent_vector: torch.Tensor, variation_index: int = 0) -> Image.Image:
        """Generate using the full StyleGAN2 model."""
        with torch.no_grad():
            # Prepare latent
            latent = latent_vector.clone().detach().float()

            # Add variation noise
            if variation_index > 0:
                noise_seed = int(abs(latent.sum().item()) * 1000 + variation_index) % (2**31)
                rng = torch.Generator(device=self.device if self.device != 'mps' else 'cpu')
                # Wait! 'mps' generator might not be fully supported in some PyTorch versions for randn_like.
                # Safe approach: generate noise on CPU, then move to latent.device!
                rng.manual_seed(noise_seed)
                noise = torch.randn(latent.shape, generator=rng, device='cpu', dtype=latent.dtype).to(latent.device)
                latent = latent + noise * 0.12

            # Ensure latent has a batch dimension
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            
            latent = latent.to(self.device)

            # Generate with truncation trick for better quality
            img_tensor = self.model(
                [latent],
                input_is_latent=True,  # Map Z to W is skipped because optimize_latent already gives W
                truncation=0.7,
                truncation_latent=self.mean_latent,
            )

            # Convert tensor to PIL Image
            img = self._tensor_to_image(img_tensor)

            # Resize to target resolution
            if img.size != (self.resolution, self.resolution):
                img = img.resize((self.resolution, self.resolution), Image.LANCZOS)

            return img

    def optimize_latent(self, text_embedding, text_encoder, steps=40, lr=0.08):
        """
        Zero-Shot Latent Optimization.
        Iteratively updates a generic face latent vector to semantically match the text description.

        Args:
            text_embedding: CLIP embedding of the description
            text_encoder: TextEncoder instance with differentiable similarity scoring
            steps: Number of optimization iterations
            lr: Learning rate for Adam optimizer

        Returns:
            Optimized W latent vector, output image tensor
        """
        if self.demo_mode or self.model is None:
            return None, None

        # Start from the mean face (W space)
        latent_w = self.mean_latent.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([latent_w], lr=lr)

        print(f"[StyleGAN2] Starting latent optimization for {steps} steps...")
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Generate face tensor from W latent (input_is_latent=True)
            img_tensor = self.model([latent_w], input_is_latent=True)

            # Calculate cosine similarity loss with CLIP
            similarity = text_encoder.compute_similarity_differentiable(text_embedding, img_tensor)
            
            # We want to MAXIMIZE similarity, so MINIMIZE negative similarity
            loss = -similarity
            
            # Backpropagate and step
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"[StyleGAN2] Step {step}/{steps} - Loss: {loss.item():.4f}")

        print(f"[StyleGAN2] Optimization complete. Final similarity: {-loss.item():.4f}")
        return latent_w.detach()

    def _tensor_to_image(self, tensor):
        """Convert model output tensor [-1, 1] to PIL Image."""
        img = tensor.squeeze(0).clamp(-1, 1).cpu()
        img = (img + 1) / 2  # [-1,1] → [0,1]
        img = (img * 255).byte()
        img = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img, 'RGB')

    # ─── Demo Mode (procedural generation) ────────────────────────────

    def _demo_generate(self, latent_vector: torch.Tensor, variation_index: int = 0) -> Image.Image:
        """Generate a procedural face composite from the latent vector."""
        latent = latent_vector.detach().cpu().numpy()
        if variation_index > 0:
            var_seed = int(abs(latent.sum()) * 1000 + variation_index) % (2**31)
            var_rng = np.random.RandomState(var_seed)
            noise = var_rng.randn(*latent.shape).astype(np.float32) * 0.15
            latent = latent + noise

        size = self.resolution
        img = Image.new('RGB', (size, size), '#1a1a2e')
        draw = ImageDraw.Draw(img)
        params = self._extract_params(latent)
        self._draw_background(draw, size, params)
        self._draw_face_shape(draw, size, params)
        self._draw_eyes(draw, size, params)
        self._draw_nose(draw, size, params)
        self._draw_mouth(draw, size, params)
        self._draw_eyebrows(draw, size, params)
        self._draw_hair(draw, size, params)
        self._draw_ears(draw, size, params)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        return img

    def _extract_params(self, latent):
        v = latent
        return {
            'skin_r': int(np.clip(160 + v[0] * 60, 100, 255)),
            'skin_g': int(np.clip(130 + v[1] * 50, 80, 220)),
            'skin_b': int(np.clip(100 + v[2] * 40, 60, 180)),
            'face_width': np.clip(0.32 + v[3] * 0.08, 0.25, 0.42),
            'face_height': np.clip(0.40 + v[4] * 0.06, 0.32, 0.48),
            'jaw_width': np.clip(0.28 + v[5] * 0.06, 0.20, 0.38),
            'forehead_height': np.clip(0.12 + v[6] * 0.04, 0.08, 0.18),
            'eye_size': np.clip(0.04 + v[7] * 0.015, 0.025, 0.06),
            'eye_spacing': np.clip(0.12 + v[8] * 0.03, 0.08, 0.18),
            'eye_height': np.clip(0.42 + v[9] * 0.04, 0.36, 0.48),
            'eye_r': int(np.clip(60 + v[10] * 80, 20, 200)),
            'eye_g': int(np.clip(80 + v[11] * 80, 20, 200)),
            'eye_b': int(np.clip(120 + v[12] * 80, 20, 220)),
            'nose_width': np.clip(0.04 + v[13] * 0.02, 0.02, 0.07),
            'nose_length': np.clip(0.08 + v[14] * 0.03, 0.05, 0.12),
            'mouth_width': np.clip(0.10 + v[15] * 0.04, 0.06, 0.16),
            'mouth_height': np.clip(0.62 + v[16] * 0.04, 0.56, 0.68),
            'lip_r': int(np.clip(180 + v[17] * 40, 140, 230)),
            'lip_g': int(np.clip(90 + v[18] * 30, 60, 150)),
            'lip_b': int(np.clip(90 + v[19] * 30, 60, 140)),
            'hair_r': int(np.clip(60 + v[20] * 80, 10, 200)),
            'hair_g': int(np.clip(40 + v[21] * 60, 10, 160)),
            'hair_b': int(np.clip(30 + v[22] * 50, 10, 140)),
            'hair_volume': np.clip(0.8 + v[23] * 0.3, 0.4, 1.2),
            'hair_style_v': v[24],
            'brow_thickness': np.clip(3 + v[25] * 2, 1, 6),
            'brow_arch': np.clip(0.02 + v[26] * 0.02, 0.0, 0.05),
            'chin_shape': v[27],
            'cheekbone': np.clip(0.5 + v[28] * 0.3, 0.2, 0.9),
        }

    def _draw_background(self, draw, size, p):
        for y in range(size):
            ratio = y / size
            r = int(20 + ratio * 15)
            g = int(20 + ratio * 10)
            b = int(35 + ratio * 20)
            draw.line([(0, y), (size, y)], fill=(r, g, b))

    def _draw_face_shape(self, draw, size, p):
        cx, cy = size // 2, int(size * 0.48)
        fw = int(p['face_width'] * size)
        fh = int(p['face_height'] * size)
        skin = (p['skin_r'], p['skin_g'], p['skin_b'])
        neck_w = int(fw * 0.4)
        neck_top = cy + fh - int(fh * 0.15)
        draw.rectangle([cx - neck_w, neck_top, cx + neck_w, size], fill=skin)
        draw.ellipse([cx - fw, cy - fh, cx + fw, cy + fh], fill=skin)
        jw = int(p['jaw_width'] * size)
        jaw_points = [(cx - fw, cy + int(fh * 0.1)), (cx - jw, cy + fh), (cx, cy + fh + int(fh * 0.1)), (cx + jw, cy + fh), (cx + fw, cy + int(fh * 0.1))]
        draw.polygon(jaw_points, fill=skin)
        shadow = (max(0, p['skin_r'] - 20), max(0, p['skin_g'] - 15), max(0, p['skin_b'] - 10))
        cs = int(p['cheekbone'] * fw * 0.4)
        draw.ellipse([cx - fw + 5, cy - 5, cx - fw + 5 + cs, cy + cs], fill=shadow)
        draw.ellipse([cx + fw - 5 - cs, cy - 5, cx + fw - 5, cy + cs], fill=shadow)

    def _draw_eyes(self, draw, size, p):
        cx = size // 2
        ey = int(p['eye_height'] * size)
        sp = int(p['eye_spacing'] * size)
        es = int(p['eye_size'] * size)
        for side in [-1, 1]:
            ex = cx + side * sp
            draw.ellipse([ex - es * 2, ey - es, ex + es * 2, ey + es], fill=(240, 240, 245))
            ir = int(es * 0.75)
            draw.ellipse([ex - ir, ey - ir, ex + ir, ey + ir], fill=(p['eye_r'], p['eye_g'], p['eye_b']))
            pr = int(es * 0.35)
            draw.ellipse([ex - pr, ey - pr, ex + pr, ey + pr], fill=(10, 10, 15))
            rr = max(1, int(es * 0.15))
            draw.ellipse([ex - pr + 2, ey - pr + 1, ex - pr + 2 + rr, ey - pr + 1 + rr], fill=(255, 255, 255))

    def _draw_nose(self, draw, size, p):
        cx = size // 2
        nt = int(p['eye_height'] * size) + int(p['eye_size'] * size) + 5
        nb = nt + int(p['nose_length'] * size)
        nw = int(p['nose_width'] * size)
        ns = (max(0, p['skin_r'] - 15), max(0, p['skin_g'] - 12), max(0, p['skin_b'] - 8))
        draw.line([(cx, nt), (cx, nb)], fill=ns, width=2)
        draw.ellipse([cx - nw, nb - 3, cx - 2, nb + 5], fill=ns)
        draw.ellipse([cx + 2, nb - 3, cx + nw, nb + 5], fill=ns)

    def _draw_mouth(self, draw, size, p):
        cx = size // 2
        my = int(p['mouth_height'] * size)
        mw = int(p['mouth_width'] * size)
        lc = (p['lip_r'], p['lip_g'], p['lip_b'])
        dl = (max(0, p['lip_r'] - 30), max(0, p['lip_g'] - 20), max(0, p['lip_b'] - 15))
        draw.polygon([(cx - mw, my), (cx - mw // 2, my - 4), (cx, my - 2), (cx + mw // 2, my - 4), (cx + mw, my)], fill=lc)
        draw.polygon([(cx - mw, my), (cx - mw // 2, my + 6), (cx, my + 8), (cx + mw // 2, my + 6), (cx + mw, my)], fill=lc)
        draw.line([(cx - mw + 2, my), (cx + mw - 2, my)], fill=dl, width=1)

    def _draw_eyebrows(self, draw, size, p):
        cx = size // 2
        ey = int(p['eye_height'] * size)
        sp = int(p['eye_spacing'] * size)
        es = int(p['eye_size'] * size)
        arch = p['brow_arch'] * size
        th = int(p['brow_thickness'])
        bc = (max(0, p['hair_r'] - 10), max(0, p['hair_g'] - 10), max(0, p['hair_b'] - 5))
        for side in [-1, 1]:
            ex = cx + side * sp
            by = ey - es - 8
            bw = int(es * 2.5)
            pts = []
            for i in range(10):
                t = i / 9
                x = ex + side * (t - 0.5) * bw
                y = by + (-arch * math.sin(t * math.pi))
                pts.append((x, y))
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=bc, width=th)

    def _draw_hair(self, draw, size, p):
        cx = size // 2
        fw = int(p['face_width'] * size)
        fh = int(p['face_height'] * size)
        cy = int(size * 0.48)
        hc = (p['hair_r'], p['hair_g'], p['hair_b'])
        vol = p['hair_volume']
        ht = cy - fh - int(vol * 20)
        hl = cx - fw - int(vol * 15)
        hr = cx + fw + int(vol * 15)
        draw.ellipse([hl, ht, hr, cy - int(fh * 0.2)], fill=hc)
        if p['hair_style_v'] > 0:
            draw.rectangle([hl, cy - int(fh * 0.3), hl + 15, cy + int(fh * 0.5)], fill=hc)
            draw.rectangle([hr - 15, cy - int(fh * 0.3), hr, cy + int(fh * 0.5)], fill=hc)

    def _draw_ears(self, draw, size, p):
        cx = size // 2
        cy = int(size * 0.48)
        fw = int(p['face_width'] * size)
        eh = int(size * 0.06)
        skin = (p['skin_r'], p['skin_g'], p['skin_b'])
        es = (max(0, p['skin_r'] - 10), max(0, p['skin_g'] - 8), max(0, p['skin_b'] - 5))
        for side in [-1, 1]:
            ex = cx + side * (fw - 2)
            draw.ellipse([ex - 6, cy - eh, ex + 6, cy + eh], fill=skin, outline=es)

    @property
    def is_loaded(self) -> bool:
        return not self.demo_mode
