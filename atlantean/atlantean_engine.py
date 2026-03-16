#!/usr/bin/env python3
"""
Atlantean engine process.

- Implements AtlanteanFieldNetAdvanced (ETD2 / ETDRK4 integrators)
- ETDRK4 coefficient cache
- Low-rank separable spectral filter
- Transformer-layer -> field-group mapping and L1-sparse coupling support
- Mixed-precision inference + TorchScript / ONNX export helpers
- CLI loop: reads prompts from stdin, runs lightweight inference, returns JSON line

This is intended to be spawned by the Electron service (use python -u for unbuffered I/O).
"""
import sys
import json
import time
import math
import hashlib
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------
def sha1_of_tensors(*tensors) -> str:
    h = hashlib.sha1()
    for t in tensors:
        if t is None:
            continue
        arr = t.detach().cpu().numpy().tobytes()
        h.update(arr)
    return h.hexdigest()

# -------------------------
# Model: Advanced Atlantean field net
# (Condensed but contains ETD2 and ETDRK4, low-rank spectral filters)
# -------------------------
class ETDRK4Cache:
    """
    Cache ETDRK4 coefficients for a specific linear-symbol tensor L and dt.
    Keyed by sha1(gamma,beta,dt) to avoid recomputing expensive contour integrals.
    """
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def get(self, gamma: torch.Tensor, beta: torch.Tensor, k2: torch.Tensor, k4: torch.Tensor, dt: float):
        key = sha1_of_tensors(gamma, beta, k2, k4, torch.tensor(dt, dtype=torch.float64))
        if key in self._cache:
            return self._cache[key]
        # compute L
        gamma = gamma.view(-1,1,1)
        beta = beta.view(-1,1,1)
        L = (gamma * k2 + beta * k4).detach().cpu().numpy().astype('complex64')
        # compute coefficients using Kassam & Trefethen approach
        M = 32
        j = np.arange(1, M+1)
        r = np.exp(1j * math.pi * (j - 0.5) / M).astype('complex64')
        L_flat = L.reshape(-1,1)  # (N,1)
        LR = dt * L_flat + r[None,:]  # (N,M)
        expLR = np.exp(LR)
        expLR2 = np.exp(LR/2.0)
        Q_flat = dt * np.mean((expLR2 - 1.0) / LR, axis=1)
        f1_flat = dt * np.mean((-4.0 - LR + expLR * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=1)
        f2_flat = dt * np.mean((2.0 + LR + expLR * (-2.0 + LR)) / (LR**3), axis=1)
        f3_flat = dt * np.mean((-4.0 - 3.0 * LR - LR**2 + expLR * (4.0 - LR)) / (LR**3), axis=1)
        # reshape back to (nf,H,W)
        nf, H, W = gamma.shape[0], k2.shape[0], k2.shape[1]
        cache_entry = {
            "Q": torch.from_numpy(Q_flat.reshape(nf,H,W)).to(dtype=torch.complex64),
            "f1": torch.from_numpy(f1_flat.reshape(nf,H,W)).to(dtype=torch.complex64),
            "f2": torch.from_numpy(f2_flat.reshape(nf,H,W)).to(dtype=torch.complex64),
            "f3": torch.from_numpy(f3_flat.reshape(nf,H,W)).to(dtype=torch.complex64),
            "E": None, "E2": None, "dt": dt
        }
        self._cache[key] = cache_entry
        return cache_entry

# we need numpy for ETDRK4 precomputation
import numpy as np

class AtlanteanFieldNetAdvanced(nn.Module):
    def __init__(self, H=64, W=64, nf=10, rank=3, spec_rank=3,
                 use_spectral=True, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.H, self.W = H, W
        self.nf = nf
        self.rank = rank
        self.spec_rank = spec_rank
        self.use_spectral = use_spectral

        # fields
        self.phi = nn.Parameter(torch.randn(self.nf, H, W, device=self.device, dtype=self.dtype) * 0.02)

        # diffusion-like
        self.gamma = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.3)
        self.beta  = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.05)
        self.alpha = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.2)

        # reaction constants
        self.extra = nn.ParameterDict({
            f"c_{i}": nn.Parameter(torch.tensor(0.1 * (1.0 + 0.1*i), device=self.device, dtype=self.dtype))
            for i in range(self.nf)
        })

        # spatial resonant coupling low-rank
        self.U = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.V = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.S = nn.Parameter(torch.ones(self.rank, device=self.device, dtype=self.dtype) * 0.02)

        # low-rank separable spectral filter parameters
        if self.use_spectral:
            self.Fy = nn.Parameter(torch.randn(self.nf, self.spec_rank, self.H, device=self.device, dtype=self.dtype) * 0.01)
            self.Fx = nn.Parameter(torch.randn(self.nf, self.spec_rank, self.W, device=self.device, dtype=self.dtype) * 0.01)
            self.Fa = nn.Parameter(torch.ones(self.nf, self.spec_rank, device=self.device, dtype=self.dtype) * 0.05)
        else:
            self.register_parameter("Fy", None)
            self.register_parameter("Fx", None)
            self.register_parameter("Fa", None)

        # spectral grids
        kx = torch.fft.fftfreq(self.W, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        ky = torch.fft.fftfreq(self.H, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        kxg, kyg = torch.meshgrid(kx, ky, indexing="xy")
        kxg = kxg.T; kyg = kyg.T
        self.register_buffer("kx", kxg)
        self.register_buffer("ky", kyg)
        self.register_buffer("k2", (kxg**2 + kyg**2))
        self.register_buffer("k4", (kxg**2 + kyg**2)**2)

        # laplacian conv
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=self.device, dtype=self.dtype)
        lap_k = lap.view(1,1,3,3).repeat(self.nf,1,1,1)
        self.register_buffer("lap_kernel", lap_k)

        # SNN
        self.register_buffer("mem", torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype))
        self.threshold_base = 1.0
        self.leak_base = 0.9

        # ETDRK4 cache
        self.etdrk_cache = ETDRK4Cache()

    # ---- helpers ----
    def fft2c(self, x): return torch.fft.fft2(x, norm="ortho")
    def ifft2c(self, X): return torch.fft.ifft2(X, norm="ortho").real

    def laplacian(self, fields):
        x = fields.unsqueeze(1)
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=self.nf)
        return lap.squeeze(1)

    def resonant_coupling(self, fields, group_mask: Optional[torch.Tensor]=None):
        US = self.U * self.S.unsqueeze(0)
        w = US @ self.V.t()   # (nf,nf)
        if group_mask is not None:
            w = w * group_mask.to(w.dtype)
        coupling_factor = torch.einsum("ij, jxy -> ixy", w, fields)
        return 1e-3 * fields * coupling_factor, w

    def build_spectral_filter(self):
        if not self.use_spectral:
            return None
        Fy_e = self.Fy.unsqueeze(-1)   # (nf,R,H,1)
        Fx_e = self.Fx.unsqueeze(-2)   # (nf,R,1,W)
        prod = (Fy_e * Fx_e) * self.Fa.unsqueeze(-1).unsqueeze(-1)
        filt = prod.sum(dim=1)  # (nf,H,W)
        return filt

    def field_reaction(self, fields):
        r = torch.zeros_like(fields)
        abs_phi = fields.abs()
        c = lambda i: self.extra[f"c_{i}"]
        r[0] = -self.gamma[0] * fields[0]**3 - self.alpha[0] * (abs_phi[0]**2 * fields[0])
        r[1] = -self.gamma[1] * fields[1]**3 + c(1) * torch.sin(fields[1]) + 0.01 * (fields[1]**5)
        r[2] = -self.gamma[2] * fields[2]**3 + c(2) * fields[2] * (1.0 - abs_phi[2]**2)
        r[3] = -c(3) * fields[3]**5
        safe_log = torch.log(abs_phi[4] + 1e-6)
        r[4] = -self.gamma[4] * fields[4]**3 + c(4) * fields[4] * safe_log
        r[5] = -self.gamma[5] * fields[5]**3 + c(5) * fields[5]**5
        r[6] = -self.gamma[6] * fields[6]**3 + c(6) * fields[6]**5
        r[7] = -self.gamma[7] * fields[7]**3 + c(7) * torch.sinh(fields[7])
        r[8] = -self.gamma[8] * fields[8]**3 + c(8) * fields[8]**3
        r[9] = -self.gamma[9] * fields[9]**3 + c(9) * fields[9]**7 * 1e-3
        return r

    def spectral_coupling(self, fields):
        filt = self.build_spectral_filter()
        if filt is None:
            return torch.zeros_like(fields)
        F_hat = self.fft2c(filt)
        phi_hat = self.fft2c(fields)
        coupled_hat = phi_hat * F_hat
        return self.ifft2c(coupled_hat) * 1e-2

    # ---- Integrators ----
    @staticmethod
    def phi1(z):
        small = z.abs() < 1e-12
        out = (torch.exp(z) - 1.0) / (z + 1e-20)
        out = out * (~small).to(out.dtype) + 1.0 * small.to(out.dtype)
        return out

    def spectral_etd2_step(self, dt, fields, group_mask: Optional[torch.Tensor]=None):
        gamma = self.gamma.view(-1,1,1)
        beta  = self.beta.view(-1,1,1)
        L = gamma * self.k2 + beta * self.k4

        u = fields
        uhat = self.fft2c(u)
        N_u = self.field_reaction(u)
        spec = self.spectral_coupling(u) if self.use_spectral else 0.0
        resonant, _ = self.resonant_coupling(u, group_mask)
        N_u = N_u + spec + resonant
        Nhat_u = self.fft2c(N_u)

        E2 = torch.exp(0.5 * dt * L)
        E = torch.exp(dt * L)
        phi1_E2 = self.phi1(0.5 * dt * L)
        phi1_E  = self.phi1(dt * L)

        uhat_half = E2 * uhat + (dt*0.5) * phi1_E2 * Nhat_u
        u_half = self.ifft2c(uhat_half)
        N_half = self.field_reaction(u_half)
        N_half = N_half + (self.spectral_coupling(u_half) if self.use_spectral else 0.0) + self.resonant_coupling(u_half, group_mask)[0]
        Nhat_half = self.fft2c(N_half)
        uhat_new = E * uhat + dt * phi1_E * Nhat_half
        u_new = self.ifft2c(uhat_new)
        return u_new

    def spectral_etdrk4_step(self, dt, fields, group_mask: Optional[torch.Tensor]=None):
        # use cache for coefficients
        cache = self.etdrk_cache.get(self.gamma, self.beta, self.k2, self.k4, dt)
        # compute E and E2
        gamma = self.gamma.view(-1,1,1)
        beta  = self.beta.view(-1,1,1)
        L = gamma * self.k2 + beta * self.k4
        E = torch.exp(dt * L).to(torch.complex64)
        E2 = torch.exp(dt * L / 2.0).to(torch.complex64)

        # if cache entries don't have E/E2, set them
        # (cache stores Q,f1..f3 as complex tensors on CPU; move to device)
        Q = cache["Q"].to(device=self.device)
        f1 = cache["f1"].to(device=self.device)
        f2 = cache["f2"].to(device=self.device)
        f3 = cache["f3"].to(device=self.device)

        u = fields
        uhat = torch.fft.fft2(u, norm="ortho")
        N0 = self.field_reaction(u) + (self.spectral_coupling(u) if self.use_spectral else 0.0) + self.resonant_coupling(u, group_mask)[0]
        N0_hat = torch.fft.fft2(N0, norm="ortho")

        uhat_a = E2 * uhat + Q * N0_hat
        ua = torch.fft.ifft2(uhat_a, norm="ortho").real
        Na = self.field_reaction(ua) + (self.spectral_coupling(ua) if self.use_spectral else 0.0) + self.resonant_coupling(ua, group_mask)[0]
        Na_hat = torch.fft.fft2(Na, norm="ortho")

        uhat_b = E2 * uhat + Q * Na_hat
        ub = torch.fft.ifft2(uhat_b, norm="ortho").real
        Nb = self.field_reaction(ub) + (self.spectral_coupling(ub) if self.use_spectral else 0.0) + self.resonant_coupling(ub, group_mask)[0]
        Nb_hat = torch.fft.fft2(Nb, norm="ortho")

        uhat_c = E * uhat + Q * (2.0 * Nb_hat - N0_hat)
        uc = torch.fft.ifft2(uhat_c, norm="ortho").real
        Nc = self.field_reaction(uc) + (self.spectral_coupling(uc) if self.use_spectral else 0.0) + self.resonant_coupling(uc, group_mask)[0]
        Nc_hat = torch.fft.fft2(Nc, norm="ortho")

        uhat_new = E * uhat + f1 * N0_hat + 2.0 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        u_new = torch.fft.ifft2(uhat_new, norm="ortho").real
        return u_new

    def pde_step(self, dt=0.02, integrator="etdrk4", group_mask: Optional[torch.Tensor]=None):
        if integrator == "etdrk4":
            new_phi = self.spectral_etdrk4_step(dt, self.phi, group_mask)
        elif integrator == "etd2":
            new_phi = self.spectral_etd2_step(dt, self.phi, group_mask)
        else:
            # explicit Euler fallback
            fields = self.phi
            lap = self.laplacian(fields)
            bilap = self.laplacian(lap)
            diffusion = - self.gamma.view(-1,1,1) * lap + self.beta.view(-1,1,1) * bilap
            reaction = self.field_reaction(fields)
            resonant, _ = self.resonant_coupling(fields, group_mask)
            new_phi = fields + dt * (diffusion + reaction + resonant)
        with torch.no_grad():
            self.phi.copy_(new_phi)
            # stabilize the field values: clamp and remove NaNs/Infs
            self.phi.data = torch.nan_to_num(self.phi.data, nan=0.0, posinf=5.0, neginf=-5.0)
            self.phi.data.clamp_(-5.0, 5.0)

    def snn_step(self, input_spikes):
        excitability = self.phi[0]
        plasticity = self.phi[4]
        threshold = self.threshold_base - 0.4 * excitability.clamp(-0.9, 0.9)
        leak = self.leak_base / (1.0 + 0.2 * plasticity.abs())
        self.mem = leak * self.mem + input_spikes.to(self.device, self.dtype)
        spikes = (self.mem > threshold).to(self.dtype)
        self.mem = self.mem * (1.0 - spikes)
        return spikes

    def forward(self, input_spikes=None, pde_steps=1, dt=0.02, integrator="etdrk4", group_mask: Optional[torch.Tensor]=None):
        for _ in range(max(1, pde_steps)):
            self.pde_step(dt=dt, integrator=integrator, group_mask=group_mask)
            with torch.no_grad():
                resonant, _ = self.resonant_coupling(self.phi, group_mask)
                self.phi.add_(resonant * 0.5)
        if input_spikes is None:
            input_spikes = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)
        spikes = self.snn_step(input_spikes)
        with torch.no_grad():
            self.phi[2].add_(0.005 * spikes)
        return {"fields": self.phi, "spikes": spikes}

    # ----- helpers for export and mixed precision -----
    def to_mixed_precision(self):
        # convert parameters and buffers to float16 where safe
        self.half()
        # keep k-grid and buffers in float32 for stability
        self.kx = self.kx.float()
        self.ky = self.ky.float()
        self.k2 = self.k2.float()
        self.k4 = self.k4.float()
        return self

    def export_torchscript(self, path: str, example_input=None):
        self.eval()
        if example_input is None:
            example_input = torch.zeros(self.H, self.W, device=self.device)
        try:
            scripted = torch.jit.trace(self, (example_input, ), strict=False)
            scripted.save(path)
            return True, None
        except Exception as e:
            return False, str(e)

    def export_onnx(self, path: str, example_input=None):
        self.eval()
        if example_input is None:
            example_input = torch.zeros(1, self.H, self.W, device=self.device)
        # create wrapper for ONNX (batch dim)
        class Wrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, x):
                # expect x shape (1,H,W)
                x = x.squeeze(0)
                out = self.net(input_spikes=x, pde_steps=1, dt=0.02, integrator="etdrk4")
                # return flattened fields and spike sum
                return out["fields"].view(-1), out["spikes"].view(-1)
        wrapper = Wrapper(self)
        try:
            dummy_in = torch.zeros(1, self.H, self.W, device=self.device)
            torch.onnx.export(wrapper, dummy_in, path, opset_version=13, dynamic_axes=None)
            return True, None
        except Exception as e:
            return False, str(e)

# -------------------------
# Transformer-layer -> field-group mapping & sparse mask helper
# -------------------------
def map_transformer_layers_to_groups(num_transformers: int, nf: int) -> Tuple[Dict[int, range], torch.Tensor]:
    """
    Partition nf fields into num_transformers groups (as even as possible).
    Return mapping dict and a soft mask (nf,nf) that encourages intra-group coupling.
    The mask values: 1.0 inside group, 0.2 outside group (soft).
    """
    base = nf // num_transformers
    remainder = nf % num_transformers
    groups = {}
    cur = 0
    mask = torch.full((nf,nf), 0.2, dtype=torch.float32)
    for t in range(num_transformers):
        size = base + (1 if t < remainder else 0)
        start = cur
        end = cur + size
        groups[t] = range(start, end)
        # set intra-group to 1.0
        for i in range(start, end):
            for j in range(start, end):
                mask[i,j] = 1.0
        cur = end
    return groups, mask

# -------------------------
# Training routine with L1 sparsity on coupling
# -------------------------
def train_with_l1(net: AtlanteanFieldNetAdvanced,
                  steps=200, lr=1e-3, dt=0.03, pde_steps=2, device=None,
                  l1_w=1e-3, num_transformers=5, optimize_spectral=True):
    device = device or net.device
    net.to(device)
    groups, group_mask = map_transformer_layers_to_groups(num_transformers, net.nf)
    group_mask = group_mask.to(device)

    # select parameters to optimize
    params = []
    if optimize_spectral and net.use_spectral:
        params += [net.Fa, net.Fy, net.Fx]
    params += [net.U, net.V, net.S] + list(net.extra.parameters()) + [net.gamma, net.beta]
    params = [p for p in params if p is not None and p.requires_grad]

    opt = torch.optim.Adam(params, lr=lr)
    for s in range(steps):
        opt.zero_grad()
        inp = (torch.rand(net.H, net.W, device=device) < 0.005).float()
        out = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, integrator="etdrk4", group_mask=group_mask)
        fields = out["fields"]
        # simple spectral objective: maximize fraction energy near radius k0 via dodecagonal mask
        mask = dodecagonal_mask(net.H, net.W, k0=7.5, device=device, dtype=net.dtype)
        loss_spec, _ = spectral_target_loss(fields, mask)
        # L1 on coupling matrix w (computed from U,S,V)
        US = net.U * net.S.unsqueeze(0)
        w = US @ net.V.t()
        l1 = w.abs().sum()
        loss = loss_spec + l1_w * l1 + 1e-4 * (fields.abs().mean())
        loss.backward()
        opt.step()
        if s % max(1, steps//10) == 0:
            print(f"train step {s} loss {loss.item():.6f} l1 {l1.item():.6f}")
    return net

# -------------------------
# Spectral helpers (mask and loss)
# -------------------------
def dodecagonal_mask(H, W, k0=8.0, bandwidth=1.5, angle_tolerance=6.0, device=None, dtype=torch.float32):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    kx = torch.fft.fftfreq(W, 1.0).to(device, dtype=dtype) * (2*math.pi)
    ky = torch.fft.fftfreq(H, 1.0).to(device, dtype=dtype) * (2*math.pi)
    kxg, kyg = torch.meshgrid(kx, ky, indexing="xy")
    kxg = kxg.T; kyg = kyg.T
    kr = torch.sqrt(kxg**2 + kyg**2)
    theta = torch.atan2(kyg, kxg) * 180.0 / math.pi
    theta = (theta + 360.0) % 360.0
    mask = torch.zeros_like(kr)
    angles = torch.arange(0, 360, 30.0, device=device)
    for a in angles:
        diff = ((theta - a + 180.0) % 360.0) - 180.0
        ang_mask = (diff.abs() <= angle_tolerance).float()
        rad_mask = torch.exp(-((kr - k0)**2) / (2.0 * (bandwidth**2)))
        mask += ang_mask * rad_mask
    mask = mask / (mask.max().clamp(min=1.0))
    return mask

def spectral_target_loss(fields, mask):
    pattern = fields.sum(dim=0)
    P_hat = torch.fft.fft2(pattern, norm="ortho")
    power = (P_hat.abs()**2)
    E_mask = (power * mask).sum()
    E_tot = power.sum() + 1e-12
    frac = E_mask / E_tot
    loss = -frac
    return loss, frac.item()

# -------------------------
# Simple encode and inference helpers for CLI usage
# -------------------------
def encode_prompt_into_phi(prompt: str, phi: torch.Tensor, H: int):
    # simple index-based injection (deterministic)
    idx = abs(hash(prompt)) % H
    with torch.no_grad():
        phi[0, idx, :] += 2.0
    return phi

def lightweight_infer(net: AtlanteanFieldNetAdvanced, prompt: str, steps=200, dt=0.04, pde_steps=1, integrator="etdrk4"):
    # encode prompt
    encode_prompt_into_phi(prompt, net.phi, net.H)
    # small poisson noise to excite dynamics
    for _ in range(steps):
        inp = (torch.rand(net.H, net.W, device=net.device) < 0.005).float()
        net(input_spikes=inp, pde_steps=pde_steps, dt=dt, integrator=integrator)
    # compute simple readout: mean of field 3 and spectral fraction
    fields = net.phi.detach().cpu()
    val = fields[3].mean().item()
    mask = dodecagonal_mask(net.H, net.W, k0=7.5, device=net.device, dtype=net.dtype)
    _, frac = spectral_target_loss(net.phi, mask)
    # simple text response mapping
    response = {
        "value_field3_mean": float(val),
        "dodec_frac": float(frac),
        "result": "TRUE" if val > 0 else "FALSE",
        "fields": fields.numpy().tolist()
    }
    return response

# -------------------------
# CLI process loop
# -------------------------
def main_loop():
    # initialize model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AtlanteanFieldNetAdvanced(H=48, W=48, nf=10, rank=2, spec_rank=3, use_spectral=True, device=device)
    print(json.dumps({"status":"ready"}))
    sys.stdout.flush()
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            print(json.dumps({"error":"empty prompt"}))
            sys.stdout.flush()
            continue
        t0 = time.time()
        try:
            res = lightweight_infer(net, prompt, steps=120, dt=0.04, pde_steps=1, integrator="etdrk4")
            res["time_s"] = time.time() - t0
            print(json.dumps(res))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

if __name__ == "__main__":
    main_loop()