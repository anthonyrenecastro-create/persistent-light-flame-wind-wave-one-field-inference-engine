# AtlanteanFieldNetExtended -- FFT-based coupling, ETD spectral integrator, training loop, profiler
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtlanteanFieldNetExtended(nn.Module):
    """
    Extended AtlanteanFieldNet with:
      - optional spectral (FFT-based) nonlocal coupling (learnable spectral filters)
      - semi-implicit spectral integrator (ETD1 / integrating factor) for stability at larger dt
      - retained low-rank resonant coupling (U, V, S)
      - same SNN-style lightweight membrane model
    """

    def __init__(self, H=64, W=64, nf=10, rank=3, use_spectral=True, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.H, self.W = H, W
        self.nf = nf
        self.rank = rank
        self.use_spectral = use_spectral

        # fields (Parameters)
        init = (torch.randn(self.nf, H, W, device=self.device, dtype=self.dtype) * 0.02)
        self.phi = nn.Parameter(init)

        # per-field linear diffusion coeffs (learnable)
        self.gamma = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.3)
        self.beta  = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.05)
        self.alpha = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.2)

        # small reaction coefficients per field
        self.extra = nn.ParameterDict({
            f"c_{i}": nn.Parameter(torch.tensor(0.1 * (1.0 + 0.1*i), device=self.device, dtype=self.dtype))
            for i in range(self.nf)
        })

        # low-rank resonant coupling (cheap nf x nf)
        self.U = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.V = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.S = nn.Parameter(torch.ones(self.rank, device=self.device, dtype=self.dtype) * 0.02)

        # spectral filters (optional nonlocal coupling): small magnitude init
        if self.use_spectral:
            # storing full spectral filter per-field (nf, H, W) -- for small H,W this is efficient and flexible
            self.spectral_filter = nn.Parameter(torch.randn(self.nf, H, W, device=self.device, dtype=self.dtype) * 1e-3)
        else:
            self.register_parameter("spectral_filter", None)

        # precompute k-grid for spectral integrator and kernels (as buffers)
        kx = torch.fft.fftfreq(W, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        ky = torch.fft.fftfreq(H, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="xy")
        # Note: use shapes (H, W)
        kx_grid = kx_grid.T  # align to (H,W)
        ky_grid = ky_grid.T
        self.register_buffer("kx", kx_grid)
        self.register_buffer("ky", ky_grid)
        self.register_buffer("k2", (kx_grid**2 + ky_grid**2))
        self.register_buffer("k4", (kx_grid**2 + ky_grid**2)**2)

        # SNN state
        self.register_buffer("mem", torch.zeros(H, W, device=self.device, dtype=self.dtype))
        self.threshold_base = 1.0
        self.leak_base = 0.9

        # small laplacian kernel for optional local conv ops
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=self.device, dtype=self.dtype)
        lap_k = lap.view(1,1,3,3).repeat(self.nf,1,1,1)
        self.register_buffer("lap_kernel", lap_k)

    # ---- helpers: FFT transforms ----
    def _rfft2(self, x):
        # x: (nf, H, W) real spatial -> complex spectral (nf, H, W)
        return torch.fft.rfft2(x, norm="ortho")

    def _irfft2(self, x_hat, H=None, W=None):
        # x_hat: complex rfft output shape (nf, H, W//2+1) -> real (nf, H, W)
        return torch.fft.irfft2(x_hat, s=(H or self.H, W or self.W), norm="ortho")

    # ---- local spectral symbols + conv ----
    def laplacian(self, fields):
        # group conv laplacian (nf, H, W)
        x = fields.unsqueeze(1)       # (nf,1,H,W)
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=self.nf)
        return lap.squeeze(1)

    def biharmonic(self, fields):
        return self.laplacian(self.laplacian(fields))

    # ---- low-rank resonant coupling (same as before) ----
    def resonant_coupling(self, fields):
        US = self.U * self.S.unsqueeze(0)    # (nf,rank)
        w = US @ self.V.t()                  # (nf, nf)
        coupling_factor = torch.einsum("ij, jxy -> ixy", w, fields)
        coupling = fields * coupling_factor
        return 1e-3 * coupling

    # ---- spectral nonlocal coupling (learnable filters) ----
    def spectral_coupling(self, fields):
        # fields: (nf,H,W) -> rfft -> multiply by spectral_filter in spectral domain -> ifft
        # spectral_filter stored in real spatial shape (nf,H,W) but we use rfft2 over same grid;
        # to be consistent we convert spectral_filter to rfft2 and reuse same truncated frequencies.
        # For simplicity compute rfft2(filter) on the fly (cheap) and multiply component-wise.
        # NOTE: for performance at scale you may store precomputed rfft of filters or parameterize in spectral domain directly.
        fh_hat = self._rfft2(fields)           # complex (nf, H, Wfreq)
        filter_hat = self._rfft2(self.spectral_filter)  # treat spectral_filter as real spatial
        coupled_hat = fh_hat * filter_hat
        coupled = self._irfft2(coupled_hat, self.H, self.W)
        return 1e-2 * coupled   # weak scale

    # ---- field-specific reaction terms ----
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

    # ---- spectral semi-implicit integrator (ETD1 / integrating-factor) ----
    def spectral_etd_step(self, dt, fields):
        """
        Integrate fields using an exponential integrating factor for the linear part
        and explicit-first-order for the nonlinear part (ETD1):
           phi_hat_{n+1} = E * phi_hat_n + (E - 1)/L * dt * N_hat_n
        where L(k) is the linear symbol per-field: L = gamma*k^2 + beta*k^4
        N is nonlinear (reactions + nonlocal coupling treated explicitly)
        """
        # compute linear symbol L per field: shape (nf,H,W)
        k2 = self.k2
        k4 = self.k4
        # expand gamma/beta to (nf,1,1)
        gamma = self.gamma.view(-1,1,1)
        beta  = self.beta.view(-1,1,1)
        L = gamma * k2 + beta * k4  # (nf,H,W)

        # spectral transforms
        phi_hat = self._rfft2(fields)  # complex shape (nf, H, Wfreq)
        # compute nonlinear term in spatial domain
        reaction = self.field_reaction(fields)
        nonlocal = self.spectral_coupling(fields) if (self.use_spectral and self.spectral_filter is not None) else 0.0
        resonant = self.resonant_coupling(fields)
        N = reaction + nonlocal + resonant  # (nf,H,W)

        N_hat = self._rfft2(N)

        # For rfft shapes, need L in full (H,W) but N_hat is truncated in freq dim.
        # We'll compute E = exp(L*dt) in full grid and then slice for rfft multiplication.
        E_full = torch.exp(L * dt)  # (nf,H,W)
        # compute phi_hat_new_full = E_full * phi_full + (E_full - 1)/L * dt * N_full in spectral domain
        # but we only have rfft representations. Implement by computing phi_full and N_full via ifft?
        # Simpler: compute everything in full spatial domain using pointwise integrating-factor trick:
        # phi_new_spatial = irfft( E * fft(phi) ) + irfft( ((E -1)/L * dt) * fft(N) )
        # To avoid costly full complex-slicing mismatch, perform multiplication using full-frequency arrays via rfft-compatible slicing:
        # Build L_rfft shaped like rfft output by truncating last axis
        freq_shape = phi_hat.shape  # (nf, H, Wfreq)
        # build L_hat_rfft by taking rfft of a real array representing L (but L is real in space -> rfft(L) not equal). Simpler:
        # We'll compute in full spatial domain using real transforms: compute E_full * fields and then transform.
        # This is a small overhead but keeps code clear.
        with torch.no_grad():
            term1 = self._irfft2(phi_hat * 1.0, self.H, self.W)  # phi in spatial (should equal fields)
        # Exact integrating-factor update (spatial operations) using spectral multipliers:
        # compute E * phi in spectral: E_full * phi_hat_full -> inverse FFT to spatial
        # we'll compute by converting E_full (real grid) to spectral by rfft2(E_full) and convolving: that's messy.
        # Simpler and stable approach: apply integrating factor in spectral domain per-frequency using full FFT (complex) from scratch:
        # compute complex full-spectrum phi_hat_full via fft2 (complex), but we used rfft2 earlier. Instead use torch.fft.fft2 (complex).
        phi_hat_c = torch.fft.fft2(fields, norm="ortho")   # complex (nf,H,W)
        N_hat_c   = torch.fft.fft2(N, norm="ortho")
        # L is real (nf,H,W) and matches full FFT grid
        E = torch.exp(L * dt)   # (nf,H,W) real positive
        # avoid division by zero for L near zero
        denom = L
        small = denom.abs() < 1e-12
        # compute phi_hat_new (complex) per-grid
        # formula: phi_hat_new = E * phi_hat_c + ((E - 1.0) / L) * dt * N_hat_c  (elementwise)
        coef = (E - 1.0) / (denom + small.float() * 1.0)  # safe denom
        coef = coef * (~small).to(coef.dtype) + dt * (~small).to(coef.dtype) * small.to(coef.dtype)  # handle small L later
        # proper handling: where small, (E-1)/L -> dt (limit)
        coef = (E - 1.0) / (denom + 1e-20)  # stable numeric; fine for practical use
        phi_hat_new = E * phi_hat_c + coef * dt * N_hat_c
        # invert to spatial
        phi_new = torch.fft.ifft2(phi_hat_new, norm="ortho").real
        return phi_new

    def pde_step(self, dt=0.02, use_spectral_integrator=True):
        if use_spectral_integrator and self.use_spectral:
            with torch.no_grad():
                new_phi = self.spectral_etd_step(dt, self.phi)
                self.phi.copy_(new_phi)
        else:
            # fallback explicit euler local integration
            fields = self.phi
            lap = self.laplacian(fields)
            bilap = self.biharmonic(fields)
            diffusion = - self.gamma.view(-1,1,1) * lap + self.beta.view(-1,1,1) * bilap
            reaction = self.field_reaction(fields)
            dphi = diffusion + reaction + self.resonant_coupling(fields)
            with torch.no_grad():
                self.phi.add_(dt * dphi)

    # ---- SNN step ----
    def snn_step(self, input_spikes):
        excitability = self.phi[0]
        plasticity = self.phi[4]
        threshold = self.threshold_base - 0.4 * excitability.clamp(-0.9,0.9)
        leak = self.leak_base / (1.0 + 0.2 * plasticity.abs())
        self.mem = leak * self.mem + input_spikes.to(self.device, self.dtype)
        spikes = (self.mem > threshold).to(self.dtype)
        self.mem = self.mem * (1.0 - spikes)
        return spikes

    # ---- forward wrapper ----
    def forward(self, input_spikes=None, pde_steps=1, dt=0.02, use_spectral_integrator=True):
        for _ in range(max(1, pde_steps)):
            self.pde_step(dt=dt, use_spectral_integrator=use_spectral_integrator)
            # small resonant coupling applied directly (weak)
            with torch.no_grad():
                coupling = self.resonant_coupling(self.phi)
                self.phi.add_(coupling * 0.5)

        if input_spikes is None:
            input_spikes = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)
        spikes = self.snn_step(input_spikes)
        with torch.no_grad():
            self.phi[2].add_(0.005 * spikes)
        return {"fields": self.phi, "spikes": spikes}

# ---- spectral target mask for dodecagonal (12-fold) pattern ----
def dodecagonal_mask(H, W, k0=8.0, bandwidth=1.5, angle_tolerance=6.0, device=None, dtype=torch.float32):
    """
    Create a mask in (H,W) Fourier space selecting energy near radius k0 and angles at multiples of 30 degrees (12-fold).
    Returns mask (H,W) real in spatial-frequency grid (full FFT frequency grid, not rfft truncated).
    k0 is in units coherent with fftfreq*(2pi) used in net; if you use default grid above, tune k0 accordingly.
    angle_tolerance in degrees around each preferred direction.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype
    kx = torch.fft.fftfreq(W, 1.0).to(device, dtype=dtype) * (2*math.pi)
    ky = torch.fft.fftfreq(H, 1.0).to(device, dtype=dtype) * (2*math.pi)
    kxg, kyg = torch.meshgrid(kx, ky, indexing="xy")
    kxg = kxg.T; kyg = kyg.T
    k2 = kxg**2 + kyg**2
    kr = torch.sqrt(k2)
    theta = torch.atan2(kyg, kxg)  # radians
    angles_deg = (theta * 180.0 / math.pi) % 360.0
    mask = torch.zeros_like(kr)
    # target angles multiples of 30 degrees (0,30,..330) -> 12 directions
    target_angles = torch.arange(0,360,30, device=device, dtype=dtype)
    tol = angle_tolerance
    for a in target_angles:
        # compute angular difference
        diff = (angles_deg - a + 180.0) % 360.0 - 180.0
        ang_mask = (diff.abs() <= tol).float()
        rad_mask = torch.exp(-((kr - k0)**2) / (2 * (bandwidth**2)))
        mask += ang_mask * rad_mask
    # normalize mask
    mask = mask / (mask.max().clamp(min=1.0))
    return mask

# ---- spectral loss: maximize energy at mask locations (we minimize negative of energy fraction) ----
def spectral_target_loss(fields, mask):
    """
    fields: (nf,H,W) real; combine fields (e.g., sum) for spectral analysis
    mask: (H,W) full fft grid (not truncated)
    """
    # combine fields into single real pattern (e.g., sum over fields)
    pattern = fields.sum(dim=0)  # (H,W)
    # compute full complex FFT2
    P_hat = torch.fft.fft2(pattern, norm="ortho")
    power = (P_hat.abs()**2)
    # mask is real on full grid; ensure shapes match
    # energy_in_mask: sum_{k} power[k] * mask[k]
    E_mask = (power * mask).sum()
    E_tot = power.sum() + 1e-12
    # we want high fraction in mask: loss = -E_mask/E_tot
    frac = E_mask / E_tot
    loss = -frac
    # return loss and diagnostic fraction
    return loss, frac.item()

# ---- training loop to encourage dodecagonal patterns ----
def train_for_patterns(net: AtlanteanFieldNetExtended,
                       steps=200,
                       lr=1e-3,
                       dt=0.05,
                       pde_steps=2,
                       target_k0=8.0,
                       device=None,
                       optimize_params=("U", "V", "S", "extra", "gamma", "beta", "spectral_filter"),
                       verbose=True):
    device = device or net.device
    net.to(device)
    # build mask once (full FFT grid)
    mask = dodecagonal_mask(net.H, net.W, k0=target_k0, device=device, dtype=net.dtype)
    # collect parameters to train
    params = []
    name_map = {
        "U": net.U, "V": net.V, "S": net.S, "extra": list(net.extra.parameters()),
        "gamma": net.gamma, "beta": net.beta, "spectral_filter": net.spectral_filter
    }
    for key in optimize_params:
        val = name_map.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            params += val
        else:
            params.append(val)
    # flatten param list
    params = [p for p in params if p is not None]
    if len(params) == 0:
        raise ValueError("No parameters selected for optimization.")
    opt = torch.optim.Adam(params, lr=lr)
    # regularization weights
    param_reg_w = 1e-4
    stability_w = 1e-2
    for s in range(steps):
        opt.zero_grad()
        # random sparse inputs to excite dynamics (small Poisson)
        inp = (torch.rand(net.H, net.W, device=device) < 0.005).float()
        out = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, use_spectral_integrator=True)
        fields = out["fields"]  # (nf,H,W)
        # spectral loss
        loss_spec, frac = spectral_target_loss(fields, mask)
        # stability loss: penalize large amplitude & temporal churn (mean abs)
        stability = fields.abs().mean()
        # parameter regularization
        reg = 0.0
        for p in params:
            reg = reg + (p**2).sum()
        loss = loss_spec + stability_w * stability + param_reg_w * reg
        loss.backward()
        opt.step()
        if verbose and (s % max(1, steps//10) == 0 or s < 5):
            print(f"step {s:4d} loss {loss.item():.6f} spec_frac {frac:.4f} stab {stability.item():.6f}")
    return net

# ---- profiler: measure time and (optionally) GPU memory ----
def profile_forward(net: AtlanteanFieldNetExtended, runs=20, pde_steps=2, dt=0.02, device=None):
    device = device or net.device
    net.to(device)
    # warm-up
    inp = (torch.rand(net.H, net.W, device=device) < 0.01).float()
    for _ in range(5):
        _ = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, use_spectral_integrator=True)
    # timed runs
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, use_spectral_integrator=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / runs
    # memory: GPU peak if available
    mem = None
    if device.type == "cuda":
        mem_bytes = torch.cuda.max_memory_allocated(device)
        mem = mem_bytes / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    return {"time_per_forward_s": elapsed, "gpu_mem_mb": mem}

# ---- Usage example ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AtlanteanFieldNetExtended(H=64, W=64, nf=10, rank=3, use_spectral=True, device=device)
    print("Profiling initial forward...")
    p = profile_forward(net, runs=10, pde_steps=2, dt=0.03, device=device)
    print("profile:", p)
    # quick training to nudge toward dodecagonal energy
    print("Training for spectral patterns (quick demo)...")
    net = train_for_patterns(net, steps=80, lr=5e-4, dt=0.04, pde_steps=2, target_k0=8.0, device=device,
                             optimize_params=("S", "U", "V", "spectral_filter", "extra"))
    # final profile
    p2 = profile_forward(net, runs=10, pde_steps=2, dt=0.04, device=device)
    print("final profile:", p2)
    # inspect spectral fraction after last forward
    out = net(input_spikes=None, pde_steps=4, dt=0.04, use_spectral_integrator=True)
    mask = dodecagonal_mask(net.H, net.W, k0=8.0, device=device, dtype=net.dtype)
    loss, frac = spectral_target_loss(out["fields"], mask)
    print("final spectral fraction:", frac)