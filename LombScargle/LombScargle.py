import torch
import math
import numpy as np

def compute_freqs(t, samples_per_peak = 1):
    """Compute a default set of frequencies (cycles per unit of `t`).

    Parameters
    ----------
    t : 1‑D tensor or array‑like
        Time samples; only the relative spacing matters.
    samples_per_peak : int, optional
        Oversampling factor; larger gives a denser grid.

    Returns
    -------
    torch.Tensor
        1‑D tensor of monotonically increasing frequencies.
    """
    # Convert to a flat torch tensor
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float32)
    t = t.reshape(-1).to(dtype=torch.float32)

    # Sort, then smallest gap → Nyquist‑like upper limit
    t, _ = torch.sort(t)
    dt_min = torch.min(t[1:] - t[:-1])
    f_max = 0.5 / dt_min  # = (π/dt_min)/(2π)

    n_freqs = t.numel()
    freqs = torch.linspace(1e-5, f_max, samples_per_peak * n_freqs, dtype=torch.float32, device=t.device)
    return freqs


class LombScargle(torch.nn.Module):
    """Batched Lomb–Scargle spectrum estimator."""
    def __init__(self, freqs = None):
        super().__init__()
        if freqs is not None:
            self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))
        else:
            self.freqs = None

    @staticmethod
    def _compute_fap_weights(ps, eps = 1e-5):
        m = ps.shape[-1]
        fap = 1.0 - (1.0 - torch.exp(-ps)) ** m
        return 1.0 / (fap + eps)

    def forward(self, t, y, mask = None, freqs = None, fap = True, norm = True):
        # Ensure 2‑D (B, N)
        if t.ndim == 1:
            t = t.unsqueeze(0)
            y = y.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
        if t.ndim != 2 or y.ndim != 2:
            raise ValueError("`t` and `y` must be 1‑D or 2‑D with matching shapes.")

        if mask is None:
            mask = torch.ones_like(t, dtype=t.dtype)

        # Choose frequency grid
        if freqs is not None:
            freqs_t = torch.as_tensor(freqs, dtype=t.dtype, device=t.device)
        elif self.freqs is not None:
            freqs_t = self.freqs.to(dtype=t.dtype, device=t.device)
        else:
            freqs_t = compute_freqs(t[0], samples_per_peak = 1).to(dtype=t.dtype, device=t.device)

        omega = (2.0 * math.pi * freqs_t).unsqueeze(0).unsqueeze(2)  # (1, M, 1)

        # Broadcast to (B, M, N)
        t = t.unsqueeze(1)
        y = y.unsqueeze(1)
        mask = mask.unsqueeze(1)

        two_omega_t = 2.0 * omega * t
        sin_2wt = torch.sin(two_omega_t) * mask
        cos_2wt = torch.cos(two_omega_t) * mask

        tau = torch.atan2(sin_2wt.sum(-1), cos_2wt.sum(-1)) / (2.0 * omega.squeeze())

        omega_t_tau = omega * (t - tau.unsqueeze(2))
        cos_term = torch.cos(omega_t_tau) * mask
        sin_term = torch.sin(omega_t_tau) * mask

        p_cos = ((y * cos_term).sum(-1) ** 2) / (cos_term.pow(2).sum(-1) + 1e-10)
        p_sin = ((y * sin_term).sum(-1) ** 2) / (sin_term.pow(2).sum(-1) + 1e-10)
        p = 0.5 * (p_cos + p_sin)

        if fap:
            p = p * self._compute_fap_weights(p)

        if norm:
            df = torch.diff(freqs_t)
            integral = torch.sum(0.5 * (p[:, :-1] + p[:, 1:]) * df, dim=-1) + 1e-10
            p = p / integral.unsqueeze(1)

        return p
