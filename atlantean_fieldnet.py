# lean implementation of "AtlanteanFieldNet" -- 10 coupled PDE-like fields + low-rank resonant coupling + SNN step
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtlanteanFieldNet(nn.Module):
    """
    - 10 spatial fields phi_i of shape (H, W) each, stored as Parameters.
    - Vectorized Laplacian / Biharmonic using grouped conv2d for efficiency.
    - Field-specific nonlinear reaction terms inspired by the user's specification.
    - A weak low-rank ("rank-3") resonant coupling that transfers information across fields.
    - A tiny spiking neuron interaction that reads/writes from fields.
    - Designed to be small, efficient and readable.
    """

    def __init__(self, H=64, W=64, rank=3, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.H, self.W = H, W
        self.nf = 10   # number of fields
        self.rank = rank

        # Fields: initialize small, learnable fields
        init = (torch.randn(self.nf, H, W, device=self.device, dtype=self.dtype) * 0.05)
        self.phi = nn.Parameter(init)

        # diffusion-like coefficients (learnable, per-field)
        self.gamma = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.3)
        self.beta  = nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * 0.05)

        # small set of reaction coefficients per field (matching the 10 different nonlinearties)
        # We store them in a vector dict for readability
        init_consts = lambda v: nn.Parameter(torch.ones(self.nf, device=self.device, dtype=self.dtype) * v)
        self.alpha = init_consts(0.2)   # generic cubic-like scale
        self.extra = nn.ParameterDict({
            # mapping fields 0..9 to specialized coefficients (start small)
            f"c_{i}": nn.Parameter(torch.tensor(0.1 * (1.0 + 0.1*i), device=self.device, dtype=self.dtype))
            for i in range(self.nf)
        })

        # low-rank resonant coupling factors (rank-3 decomposition for "rank-3 tensor")
        # U, V are small matrices (nf x rank) that produce bilinear weights between fields
        self.U = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        self.V = nn.Parameter(torch.randn(self.nf, self.rank, device=self.device, dtype=self.dtype) * 0.05)
        # a small learned scaling per rank component
        self.S = nn.Parameter(torch.ones(self.rank, device=self.device, dtype=self.dtype) * 0.02)

        # SNN state (kept small)
        self.register_buffer("mem", torch.zeros(H, W, device=self.device, dtype=self.dtype))
        self.threshold_base = 1.0
        self.leak_base = 0.9

        # prebuild laplacian kernel for grouped conv2d: shape (nf,1,3,3) with same kernel repeated
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], device=self.device, dtype=self.dtype)
        lap_k = lap.view(1,1,3,3).repeat(self.nf,1,1,1)
        self.register_buffer("lap_kernel", lap_k)

    def laplacian(self, fields):
        # fields: tensor (nf, H, W)
        x = fields.unsqueeze(1)       # (nf,1,H,W) - treat nf as batch with groups
        lap = F.conv2d(x, self.lap_kernel, padding=1, groups=self.nf)
        return lap.squeeze(1)         # (nf, H, W)

    def biharmonic(self, fields):
        return self.laplacian(self.laplacian(fields))

    def resonant_coupling(self, fields):
        """
        Weak low-rank rank-3 resonant coupling:
        produce per-field coupling tensor of same spatial shape:
          coupling_i(x,y) = phi_i(x,y) * sum_j w_ij * phi_j(x,y)
        where w_ij = sum_r S[r] * (U[i,r] * V[j,r]).
        This is low-rank (R small) and cheap.
        """
        # compute w_ij via low-rank factors: (nf,rank) x (rank,nf) -> (nf,nf)
        # w_ij = sum_r S[r] * U[i,r] * V[j,r]
        # compute outer product contributions efficiently:
        # U @ diag(S) @ V^T  -> (nf, nf)
        US = self.U * self.S.unsqueeze(0)    # (nf,rank) broadcast multiply
        w = US @ self.V.t()                  # (nf, nf)

        # fields: (nf, H, W). compute sum_j w_ij * phi_j
        # use einsum: out_i(x,y) = sum_j w_ij * phi_j(x,y)
        coupling_factor = torch.einsum("ij, jxy -> ixy", w, fields)
        coupling = fields * coupling_factor
        # make coupling weak (scale) to avoid overwhelming self-dynamics
        return 1e-3 * coupling

    def field_reaction(self, fields):
        """
        Apply per-field nonlinear reaction functions inspired by the user's 10 equations.
        Returns reaction tensor shape (nf,H,W).
        """
        r = torch.zeros_like(fields)
        abs_phi = fields.abs()
        # convenience access to constants
        c = lambda i: self.extra[f"c_{i}"]

        # Field 0: cubic + additional cubic-weight (∂φ1/∂t = -γ1 φ1^3 - α1 |φ1|^2 φ1 + β1 ☐^2 φ1)
        r[0] = -self.gamma[0] * fields[0]**3 - self.alpha[0] * (abs_phi[0]**2 * fields[0])

        # Field 1: -γ φ^3 + λ sin(phi) + η phi^2 * phi^3  (interpreting the intended nonlinearities)
        r[1] = -self.gamma[1] * fields[1]**3 + c(1) * torch.sin(fields[1]) + 0.01 * (fields[1]**5)

        # Field 2: cubic damping + Stuart-Landau-like term: κ φ (1 - |φ|^2)
        r[2] = -self.gamma[2] * fields[2]**3 + c(2) * fields[2] * (1.0 - abs_phi[2]**2)

        # Field 3: quintic nonlinearity: -μ φ^5 + ν Δ^2 φ (we keep reaction as quintic)
        r[3] = -c(3) * fields[3]**5 + 0.0

        # Field 4: ω φ ln|φ| (safe log)
        safe_log = torch.log(abs_phi[4] + 1e-6)
        r[4] = -self.gamma[4] * fields[4]**3 + c(4) * fields[4] * safe_log

        # Field 5: ξ φ^3  * φ^2  -> interpret as higher-order cubic-coupling: xi * phi^5 (small)
        r[5] = -self.gamma[5] * fields[5]**3 + c(5) * fields[5]**5

        # Field 6: σ φ^4 φ  -> sigma * phi^5 (quartic-to-quintic)
        r[6] = -self.gamma[6] * fields[6]**3 + c(6) * fields[6]**5

        # Field 7: τ sinh(phi)
        r[7] = -self.gamma[7] * fields[7]**3 + c(7) * torch.sinh(fields[7])

        # Field 8: ρ phi^2 phi -> rho * phi^3 (simple quadratic-derived cubic)
        r[8] = -self.gamma[8] * fields[8]**3 + c(8) * fields[8]**3

        # Field 9: ε phi^6 phi -> epsilon * phi^7 (very strong saturating nonlinearity, keep small)
        r[9] = -self.gamma[9] * fields[9]**3 + c(9) * fields[9]**7 * 1e-3

        return r

    def pde_step(self, dt=0.02):
        """
        Single PDE time integration step (explicit Euler).
        Vectorized across all fields.
        """
        fields = self.phi       # (nf, H, W)
        lap = self.laplacian(fields)
        bilap = self.biharmonic(fields)

        # diffusion-like piece: -gamma * lap + beta * bilap
        diffusion = - self.gamma.view(-1,1,1) * lap + self.beta.view(-1,1,1) * bilap

        reaction = self.field_reaction(fields)

        dphi = diffusion + reaction

        # Euler update (in-place on Parameter data for fields)
        with torch.no_grad():
            self.phi.add_(dt * dphi)

    def snn_step(self, input_spikes):
        """
        Very small spiking model interacting with fields.
        input_spikes: tensor (H,W) float
        returns: spikes (H,W) float
        """
        excitability = self.phi[0]      # use field 0 as excitability map
        plasticity = self.phi[4]        # field 4 influences leak

        threshold = self.threshold_base - 0.4 * excitability.clamp(-0.9,0.9)
        leak = self.leak_base / (1.0 + 0.2 * plasticity.abs())

        # simple membrane update
        self.mem = leak * self.mem + input_spikes.to(self.device, self.dtype)
        spikes = (self.mem > threshold).to(self.dtype)
        # reset membrane where spikes occurred
        self.mem = self.mem * (1.0 - spikes)
        return spikes

    def forward(self, input_spikes=None, pde_steps=1, dt=0.02):
        """
        - input_spikes: None or tensor (H,W)
        - pde_steps: number of PDE substeps before reading SNN
        Returns dict with fields and spikes for inspection.
        """
        # small PDE integration loop
        for _ in range(max(1, pde_steps)):
            self.pde_step(dt=dt)
            # weak resonant coupling added to fields (vectorized)
            with torch.no_grad():
                coupling = self.resonant_coupling(self.phi)
                self.phi.add_(coupling)

        # SNN read / write
        if input_spikes is None:
            input_spikes = torch.zeros(self.H, self.W, device=self.device, dtype=self.dtype)

        spikes = self.snn_step(input_spikes)

        # feedback: write spikes back into a chosen field (field 2)
        with torch.no_grad():
            self.phi[2].add_(0.005 * spikes)

        return {"fields": self.phi, "spikes": spikes}

if __name__ == "__main__":
    # example usage
    device = torch.device("cpu")
    net = AtlanteanFieldNet(H=48, W=48, rank=3, device=device)
    # random low-rate Poisson-like input for 10 time steps
    for t in range(10):
        inp = (torch.rand(net.H, net.W) < 0.01).float()  # sparse spikes
        out = net(inp, pde_steps=2, dt=0.015)
    # inspect field 0 summary
    print("field0 mean, std:", out["fields"][0].mean().item(), out["fields"][0].std().item())