import torch
from models.stylegan2_arch import StyleGAN2GeneratorFull

ckpt = torch.load('e:/rupeshmajor/weights/stylegan2-ffhq-config-f.pt', map_location='cpu')
state_dict = ckpt['g_ema']

mapped = {}
for k, v in state_dict.items():
    if 'activate.bias' in k:
        k = k.replace('.activate.bias', '.bias')
        v = v.view(1, -1, 1, 1)  # Reshape bias to (1, C, 1, 1)
    elif 'conv.blur.kernel' in k:
        continue
    # Check for other biases
    if k.endswith('.bias') and not 'style' in k and not 'modulation' in k and not 'to_rgb' in k:
        if len(v.shape) == 1:
            v = v.view(1, -1, 1, 1)
            
    mapped[k] = v

model = StyleGAN2GeneratorFull(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2)
missing, unexpected = model.load_state_dict(mapped, strict=False)

print("Missing keys:")
for m in missing:
    print(m)

print("\nUnexpected keys:")
for u in unexpected:
    print(u)

