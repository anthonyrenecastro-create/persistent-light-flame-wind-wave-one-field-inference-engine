"""
Small training and visualization helper.
- Trains the net briefly with L1 sparse coupling objective
- Records field images and spectrum during rollout
- Saves PNGs to disk
"""
import os
import matplotlib.pyplot as plt
import torch
from atlantean.atlantean_engine import AtlanteanFieldNetAdvanced, train_with_l1, dodecagonal_mask, spectral_target_loss

def save_field_image(field, path):
    plt.figure(figsize=(4,4))
    plt.imshow(field, cmap="RdBu", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_spectrum_image(fields, path):
    # fields: (nf,H,W)
    pattern = fields.sum(dim=0)
    P = torch.fft.fftshift(torch.fft.fft2(pattern.cpu(), norm="ortho"))
    power = (P.abs()**2)
    plt.figure(figsize=(4,4))
    plt.imshow(torch.log1p(power).numpy(), cmap='magma')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def quick_train_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AtlanteanFieldNetAdvanced(H=48, W=48, nf=10, rank=2, spec_rank=3, use_spectral=True, device=device)
    outdir = "python/visuals"
    os.makedirs(outdir, exist_ok=True)
    print("Training short run with L1...")
    net = train_with_l1(net, steps=80, lr=5e-4, dt=0.04, pde_steps=2, device=device, l1_w=5e-3, num_transformers=5)
    # run a rollout and save visuals
    for t in range(4):
        net(input_spikes=None, pde_steps=2, dt=0.04, integrator="etdrk4")
        fields = net.phi.detach().cpu()
        save_field_image(fields[0].numpy(), os.path.join(outdir, f"field0_t{t}.png"))
        save_spectrum_image(fields, os.path.join(outdir, f"spectrum_t{t}.png"))
    print("Saved visuals to", outdir)

if __name__ == "__main__":
    quick_train_and_visualize()