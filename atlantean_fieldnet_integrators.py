# AtlanteanFieldNetAdvanced: ETD2 and ETDRK4 integrators + low-rank separable spectral filters
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtlanteanFieldNetAdvanced(nn.Module):
    """
    Advanced Atlantean Field Net with:
      - ETD2 and ETDRK4 spectral integrators (ETDRK4 uses Kassam & Trefethen contour sampling)
      - Low-rank separable spectral_filter parameterization for memory/perf efficiency
      - Low-rank resonant coupling (U,V,S)
      - Lightweight SNN interaction
    """

    def __init__(self, H=64, W=64, nf=10, rank=3, spec_rank=4,
                 use_spectral=True, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.H, self.W = H, W
        self.nf = nf
        self.rank = rank            # spatial resonant low-rank
        self.spec_rank = spec_rank  # spectral filter separable rank
        self.use_spectral = use_spectral

        # Fields (learnable / state)
        self.phi = nn.Parameter(torch.randn(self.nf, H, W, device=self.device, dtype=self.dtype) * 0.02)

        # diffusion-like coefficients
        self.gamma = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.3)
        self.beta  = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.05)
        self.alpha = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.2)

        # reaction-specific small constants
        self.extra = nn.ParameterDict({
            f"c_{i}": nn.Parameter(torch.tensor(0.1 * (1.0 + 0.1*i), device=self.device, dtype=self.dtype))
            for i in range(self.nf)
        })

        # low-rank resonant coupling (U,V,S) same as before
        self.U = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.V = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.S = nn.Parameter(torch.ones(self.rank, device=self.device, dtype=self.dtype) * 0.02)

        # low-rank separable spectral filter parameterization (per-field)
        # For each field i and spectral-rank r, we parameterize:
        #   Fy[i,r,:] (H) and Fx[i,r,:] (W), and a scalar weight a[i,r].
        # The full spatial filter for field i is sum_r a[i,r] * outer(Fy[i,r,:], Fx[i,r,:])
        if self.use_spectral:
            self.Fy = nn.Parameter(torch.randn(self.nf, self.spec_rank, self.H, device=self.device, dtype=self.dtype) * 0.01)
            self.Fx = nn.Parameter(torch.randn(self.nf, self.spec_rank, self.W, device=self.device, dtype=self.dtype) * 0.01)
            self.Fa = nn.Parameter(torch.ones(self.nf, self.spec_rank, device=self.device, dtype=self.dtype) * 0.05)
        else:
            self.register_parameter("Fy", None)
            self.register_parameter("Fx", None)
            self.register_parameter("Fa", None)

        # Precompute spectral grids (full complex FFT grid) and local laplacian kernel
        kx = torch.fft.fftfreq(self.W, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        ky = torch.fft.fftfreq(self.H, 1.0).to(self.device, dtype=self.dtype) * (2*math.pi)
        kxg, kyg = torch.meshgrid(kx, ky, indexing="xy")
        kxg = kxg.T; kyg = kyg.T
        self.register_buffer("kx", kxg)
        self.register_buffer("ky", kyg)
        self.register_buffer("k2", (kxg**2 + kyg**2))
        self.register_buffer("k4", (kxg**2 + kyg**2)**2)

        # laplacian conv kernel for local operations
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=self.device, dtype=self.dtype)
        lap_k = lap.view(1,1,3,3).repeat(self.nf,1,1,1)
        self.register_buffer("lap_kernel", lap_k)

        # SNN membrane state
        self.register_buffer("mem", torch.zeros(H, W, device=self.device, dtype=self.dtype))
        self.threshold_base = 1.0
        self.leak_base = 0.9

    # ----- utility transforms -----
    def fft2c(self, x):
        # full complex FFT2 (nf,H,W) -> complex (nf,H,W)
        return torch.fft.fft2(x, norm="ortho")

    def ifft2c(self, X):
        return torch.fft.ifft2(X, norm="ortho").real

    # ----- low-rank spectral filter builder -----
    def build_spectral_filter(self):
        """
        Build full spatial spectral_filter (nf, H, W) from separable low-rank params:
          filter_i(x,y) = sum_r Fa[i,r] * Fy[i,r,y] * Fx[i,r,x]
        Note: Fy is (nf, R, H) -> corresponds to ky axis; Fx is (nf,R,W) -> kx axis.
        This is spatial-domain filter; we'll convolve by multiplication in spectral domain via FFT.
        """
        if not self.use_spectral:
            return None
        # compute outer products per-field and rank and sum
        # result shape (nf,H,W)
        nf, R = self.Fy.shape[0], self.spec_rank
        # expand Fy: (nf, R, H, 1) and Fx: (nf, R, 1, W)
        Fy_e = self.Fy.unsqueeze(-1)   # (nf,R,H,1)
        Fx_e = self.Fx.unsqueeze(-2)   # (nf,R,1,W)
        # outer: (nf,R,H,W), multiply by Fa weights and sum over R
        prod = (Fy_e * Fx_e) * self.Fa.unsqueeze(-1).unsqueeze(-1)
        filt = prod.sum(dim=1)   # (nf,H,W)
        return filt

    # ----- local laplacian/biharmonic -----
    def laplacian(self, fields):
        x = fields.unsqueeze(1)       # (nf,1,H,W)
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=self.nf)
        return lap.squeeze(1)

    def biharmonic(self, fields):
        return self.laplacian(self.laplacian(fields))

    # ----- low-rank resonant coupling -----
    def resonant_coupling(self, fields):
        US = self.U * self.S.unsqueeze(0)   # (nf,rank)
        w = US @ self.V.t()                 # (nf,nf)
        factor = torch.einsum("ij,jxy->ixy", w, fields)
        return 1e-3 * fields * factor

    # ----- reaction terms as before -----
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

    # ----- spectral nonlocal coupling via filter in spectral domain -----
    def spectral_coupling(self, fields):
        """
        Multiply field FFT by FFT(filter) per-field and inverse transform.
        We build filter via low-rank separable builder (spatial domain) and rfft it.
        """
        filt_spatial = self.build_spectral_filter()   # (nf,H,W)
        if filt_spatial is None:
            return torch.zeros_like(fields)
        F_hat = self.fft2c(filt_spatial)   # complex (nf,H,W)
        phi_hat = self.fft2c(fields)
        coupled_hat = phi_hat * F_hat
        coupled = self.ifft2c(coupled_hat)
        return 1e-2 * coupled  # weak scaling

    # ----- ETD helper phi function (phi1) -----
    @staticmethod
    def phi1(z):
        # phi1(z) = (exp(z) - 1) / z ; handle z~0
        small = z.abs() < 1e-12
        out = (torch.exp(z) - 1.0) / (z + 1e-20)
        out = out * (~small).to(out.dtype) + 1.0 * small.to(out.dtype)
        return out

    # ----- ETD2 integrator (second order) -----
    def spectral_etd2_step(self, dt, fields):
        """
        ETD2 integrator using integrating-factor midpoint:
          E = exp(L dt), E2 = exp(L dt/2)
          u_half = E2 * u + dt/2 * phi1(L dt/2) * N(u)
          u_{n+1} = E * u + dt * phi1(L dt) * N(u_half)
        All multiplications performed in spectral domain (full complex fft).
        """
        # linear symbol L (nf,H,W)
        gamma = self.gamma.view(-1,1,1)
        beta  = self.beta.view(-1,1,1)
        L = gamma * self.k2 + beta * self.k4

        u = fields
        uhat = self.fft2c(u)
        N_u = self.field_reaction(u) + self.resonant_coupling(u)
        N_u += self.spectral_coupling(u) if self.use_spectral else 0.0
        Nhat_u = self.fft2c(N_u)

        E2 = torch.exp(0.5 * dt * L)
        E = torch.exp(dt * L)

        phi1_E2 = self.phi1(0.5 * dt * L)
        phi1_E  = self.phi1(dt * L)

        # compute u_half in spectral: uhat_half = E2 * uhat + (dt/2) * phi1_E2 * Nhat_u
        uhat_half = E2 * uhat + (dt*0.5) * phi1_E2 * Nhat_u
        u_half = self.ifft2c(uhat_half)

        # compute N(u_half)
        N_half = self.field_reaction(u_half) + self.resonant_coupling(u_half)
        N_half += self.spectral_coupling(u_half) if self.use_spectral else 0.0
        Nhat_half = self.fft2c(N_half)

        # final update
        uhat_new = E * uhat + dt * phi1_E * Nhat_half
        u_new = self.ifft2c(uhat_new)
        return u_new

    # ----- ETDRK4 integrator (Kassam & Trefethen coefficients) -----
    def spectral_etdrk4_step(self, dt, fields, M=32):
        """
        ETDRK4 with coefficients computed by contour integral sampling (Kassam & Trefethen).
        This returns phi_{n+1} spatial array.
        """
        # linear symbol L (nf,H,W)
        gamma = self.gamma.view(-1,1,1)
        beta  = self.beta.view(-1,1,1)
        L = gamma * self.k2 + beta * self.k4    # real array (nf,H,W)

        nf, H, W = L.shape
        Npoints = nf * H * W
        L_flat = L.reshape(-1)  # shape (Npoints,)

        # contour integral sample points on complex unit circle (shifted)
        j = torch.arange(1, M+1, device=self.device)
        r = torch.exp(1j * math.pi * (j - 0.5) / M).to(self.device)  # complex tensor (M,)

        # build LR = dt*L_flat[:,None] + r[None,:]
        LR = dt * L_flat.unsqueeze(1).to(torch.complex64) + r.unsqueeze(0)  # (Npoints, M) complex

        # compute exponentials
        expLR2 = torch.exp(LR / 2.0)
        expLR  = torch.exp(LR)

        # compute coefficients (complex) via trapezoidal mean approximations
        # Q = dt * mean( (exp(LR/2) - 1) / LR )
        Q_flat = dt * torch.mean((expLR2 - 1.0) / LR, dim=1)   # (Npoints,) complex
        f1_flat = dt * torch.mean((-4.0 - LR + expLR * (4.0 - 3.0 * LR + LR**2)) / (LR**3), dim=1)
        f2_flat = dt * torch.mean((2.0 + LR + expLR * (-2.0 + LR)) / (LR**3), dim=1)
        f3_flat = dt * torch.mean((-4.0 - 3.0 * LR - LR**2 + expLR * (4.0 - LR)) / (LR**3), dim=1)

        # reshape to (nf,H,W)
        Q = Q_flat.reshape(nf, H, W)
        f1 = f1_flat.reshape(nf, H, W)
        f2 = f2_flat.reshape(nf, H, W)
        f3 = f3_flat.reshape(nf, H, W)

        # Precompute exponentials on full grid
        E = torch.exp(dt * L).to(torch.complex64)
        E2 = torch.exp(dt * L / 2.0).to(torch.complex64)

        # Work in spectral domain with complex FFTs
        u = fields
        uhat = torch.fft.fft2(u, norm="ortho")   # complex (nf,H,W)
        # stage 0
        N0 = self.field_reaction(u) + self.resonant_coupling(u)
        if self.use_spectral:
            N0 = N0 + self.spectral_coupling(u)
        N0_hat = torch.fft.fft2(N0, norm="ortho")

        # stage a
        uhat_a = E2 * uhat + Q * N0_hat
        ua = torch.fft.ifft2(uhat_a, norm="ortho").real
        Na = self.field_reaction(ua) + self.resonant_coupling(ua)
        if self.use_spectral:
            Na = Na + self.spectral_coupling(ua)
        Na_hat = torch.fft.fft2(Na, norm="ortho")

        # stage b
        uhat_b = E2 * uhat + Q * Na_hat
        ub = torch.fft.ifft2(uhat_b, norm="ortho").real
        Nb = self.field_reaction(ub) + self.resonant_coupling(ub)
        if self.use_spectral:
            Nb = Nb + self.spectral_coupling(ub)
        Nb_hat = torch.fft.fft2(Nb, norm="ortho")

        # stage c (combine)
        uhat_c = E * uhat + Q * (2.0 * Nb_hat - N0_hat)
        uc = torch.fft.ifft2(uhat_c, norm="ortho").real
        Nc = self.field_reaction(uc) + self.resonant_coupling(uc)
        if self.use_spectral:
            Nc = Nc + self.spectral_coupling(uc)
        Nc_hat = torch.fft.fft2(Nc, norm="ortho")

        # final update in spectral domain (complex)
        uhat_new = E * uhat + f1 * N0_hat + 2.0 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        u_new = torch.fft.ifft2(uhat_new, norm="ortho").real
        return u_new

    # ----- PDE step selecting integrator -----
    def pde_step(self, dt=0.02, integrator="etdrk4"):
        if integrator == "etdrk4":
            new_phi = self.spectral_etdrk4_step(dt, self.phi)
        elif integrator == "etd2":
            new_phi = self.spectral_etd2_step(dt, self.phi)
        else:
            # fallback explicit Euler
            fields = self.phi
            lap = self.laplacian(fields)
            bilap = self.biharmonic(fields)
            diffusion = - self.gamma.view(-1,1,1) * lap + self.beta.view(-1,1,1) * bilap
            reaction = self.field_reaction(fields)
            dphi = diffusion + reaction + self.resonant_coupling(fields)
            new_phi = fields + dt * dphi
        with torch.no_grad():
            self.phi.copy_(new_phi)

    # ----- SNN step -----
    def snn_step(self, input_spikes):
        excitability = self.phi[0]
        plasticity = self.phi[4]
        threshold = self.threshold_base - 0.4 * excitability.clamp(-0.9, 0.9)
        leak = self.leak_base / (1.0 + 0.2 * plasticity.abs())
        self.mem = leak * self.mem + input_spikes.to(self.device, self.dtype)
        spikes = (self.mem > threshold).to(self.dtype)
        self.mem = self.mem * (1.0 - spikes)
        return spikes

    # ----- forward -----
    def forward(self, input_spikes=None, pde_steps=1, dt=0.02, integrator="etdrk4"):
        for _ in range(max(1, pde_steps)):
            self.pde_step(dt=dt, integrator=integrator)
            with torch.no_grad():
                coupling = self.resonant_coupling(self.phi)
                self.phi.add_(coupling * 0.5)
        if input_spikes is None:
            input_spikes = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)
        spikes = self.snn_step(input_spikes)
        with torch.no_grad():
            self.phi[2].add_(0.005 * spikes)
        return {"fields": self.phi, "spikes": spikes}

# ---- small spectral target mask (dodecagonal) reused ----
def dodecagonal_mask(H, W, k0=8.0, bandwidth=1.5, angle_tol=6.0, device=None, dtype=torch.float32):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        ang_mask = (diff.abs() <= angle_tol).float()
        rad_mask = torch.exp(-((kr - k0)**2) / (2.0 * (bandwidth**2)))
        mask += ang_mask * rad_mask
    mask = mask / (mask.max().clamp(min=1.0))
    return mask

# ---- spectral loss function (same idea) ----
def spectral_target_loss(fields, mask):
    pattern = fields.sum(dim=0)
    P_hat = torch.fft.fft2(pattern, norm="ortho")
    power = (P_hat.abs()**2)
    E_mask = (power * mask).sum()
    E_tot = power.sum() + 1e-12
    frac = E_mask / E_tot
    loss = -frac
    return loss, frac.item()

# ---- simple profiler ----
def profile_forward(net: AtlanteanFieldNetAdvanced, runs=10, pde_steps=2, dt=0.03, integrator="etdrk4", device=None):
    device = device or net.device
    net.to(device)
    inp = (torch.rand(net.H, net.W, device=device) < 0.01).float()
    # warmup
    for _ in range(4):
        _ = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, integrator=integrator)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, integrator=integrator)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / runs
    mem_mb = None
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    return {"time_per_forward_s": elapsed, "gpu_mem_mb": mem_mb}

# ---- quick training routine (uses spectral_target_loss) ----
def train_for_spectral(net: AtlanteanFieldNetAdvanced, steps=200, lr=1e-3, dt=0.04, pde_steps=2,
                       target_k0=8.0, device=None, optimize_params=("Fa","Fy","Fx","U","V","S","extra"),
                       integrator="etdrk4", verbose=True):
    device = device or net.device
    net.to(device)
    mask = dodecagonal_mask(net.H, net.W, k0=target_k0, device=device, dtype=net.dtype)
    # gather trainable params
    param_list = []
    nm = {
        "Fa": net.Fa, "Fy": net.Fy, "Fx": net.Fx,
        "U": net.U, "V": net.V, "S": net.S, "extra": list(net.extra.parameters()),
        "gamma": net.gamma, "beta": net.beta
    }
    for key in optimize_params:
        val = nm.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            param_list += val
        else:
            param_list.append(val)
    param_list = [p for p in param_list if p is not None]
    if not param_list:
        raise ValueError("No parameters selected for optimization.")
    opt = torch.optim.Adam(param_list, lr=lr)
    for s in range(steps):
        opt.zero_grad()
        inp = (torch.rand(net.H, net.W, device=device) < 0.005).float()
        out = net(input_spikes=inp, pde_steps=pde_steps, dt=dt, integrator=integrator)
        fields = out["fields"]
        loss_spec, frac = spectral_target_loss(fields, mask)
        loss = loss_spec + 1e-2 * fields.abs().mean() + 1e-4 * sum((p**2).sum() for p in param_list)
        loss.backward()
        opt.step()
        if verbose and (s % max(1, steps//8) == 0 or s < 4):
            print(f"step {s:4d} loss {loss.item():.6f} spec_frac {frac:.4f}")
    return net

# ---- usage example ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AtlanteanFieldNetAdvanced(H=48, W=48, nf=10, rank=2, spec_rank=3, use_spectral=True, device=device)
    print("Profiling ETD2 integrator:")
    print(profile_forward(net, runs=6, pde_steps=2, dt=0.05, integrator="etd2", device=device))
    print("Profiling ETDRK4 integrator:")
    print(profile_forward(net, runs=6, pde_steps=2, dt=0.05, integrator="etdrk4", device=device))
    print("Quick training to nudge spectral energy...")
    net = train_for_spectral(net, steps=80, lr=5e-4, dt=0.04, pde_steps=2, target_k0=7.5,
                             optimize_params=("Fa","Fy","Fx","S","U","V","extra"), integrator="etdrk4")
    print("Final profile (ETDRK4):", profile_forward(net, runs=6, pde_steps=2, dt=0.04, integrator="etdrk4", device=device))