# LombScargle

A PyTorch implementation of the Lomb-Scargle periodogram that supports batching, masking, false-alarm probability (FAP) weighting, and optional normalization.
Lomb-Scargle can be used for spectral analysis of unevenly sampled time series.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/asztr/LombScargle.git
```

## Usage

Import modules, define time series timestamps, values and mask (optional):
```python
import torch
import numpy as np
from LombScargle import LS_omegas, LombScargleBatchMask

# Example: batch of two time series
t = torch.tensor([[0.1, 0.4, 0.7, 1.0], [0.0, 0.5, 1.0, 1.5]])
y = torch.tensor([[0.0, 1.0, 0.0, -1.0], [1.0, 0.5, -0.5, -1.0]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])  # All points valid
```

Define (angular) frequencies to evaluate. Frequencies can be arbitrary. Use `LS_omegas` to use the same frequencies defined by FFT in a regular grid:
```python
# Frequencies to evaluate (in angular frequency)
t_np = t[0].numpy()
omegas = torch.tensor(LS_omegas(t_np, samples_per_peak=5), dtype=torch.float32)
```

Compute the normalized spectrum:
```python
ls = LombScargleBatchMask(omegas)
P = ls(t, y, mask=mask, fap=True, norm=True)  # [B, M] array of power values
```

## Features

- Batching: process multiple time series in parallel.
- Masking: ignore missing or invalid observations.
- False Alarm Probability (FAP) weighting.
- Frequency normalization (ensures ∫P(f) df = 1).
- Implemented in PyTorch (seamless integration into deep learning pipelines).

## License

MIT License
