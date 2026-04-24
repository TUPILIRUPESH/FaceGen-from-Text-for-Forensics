import torch
from models.stylegan2 import StyleGAN2Generator

g = StyleGAN2Generator()
print("Loaded:", g.is_loaded)

# Hook into the layers to print stats
def print_stats(name):
    def hook(module, input, output):
        print(f"{name:15s} Output shape: {list(output.shape)}, min: {output.min().item():.3f}, max: {output.max().item():.3f}, mean: {output.mean().item():.3f}")
    return hook

g.model.conv1.register_forward_hook(print_stats("conv1"))
for i, conv in enumerate(g.model.convs):
    conv.register_forward_hook(print_stats(f"convs[{i}]"))
for i, to_rgb in enumerate(g.model.to_rgbs):
    to_rgb.register_forward_hook(print_stats(f"to_rgbs[{i}]"))

lat = torch.randn(512)
with torch.no_grad():
    img = g.generate(lat)
