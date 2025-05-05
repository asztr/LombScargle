# LombScargle

[**Project Page**](https://asztr.github.io/lscd) | [**Paper (PDF)**](https://asztr.github.io/publications/2025_lscd/2025_lscd.pdf)

A **PyTorch implementation** of the Lomb-Scargle periodogram that supports batching, masking, false-alarm probability (FAP) weighting, and optional normalization. Lomb-Scargle can be used for spectral analysis of unevenly sampled time series. This code implements part of the paper:

> **LSCD: Lomb–Scargle Conditioned Diffusion for Time Series Imputation**  
> _International Conference on Machine Learning_ (ICML 2025).  
> E. Fons*, A. Sztrajman*, Y. El-Laham, L. Ferrer, S. Vyetrenko, M. Veloso.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/asztr/LombScargle.git
```

## Basic Usage Example
```python
import torch
import math
import LombScargle

# Define example time series with single frequency = 5
t = torch.linspace(0, 10.0, 200) #timestamps
y = torch.sin(2*math.pi*5.0*t) #values

# Select frequencies to evaluate
freqs = torch.linspace(1e-5, 10.0, 100)

# Compute the normalized spectrum
ls = LombScargle.LombScargle(freqs)
P = ls(t, y, fap=True, norm=True)  # [1, 100] array of power values
```

## Features

- Batching: process multiple time series in parallel.
- Masking: ignore missing or invalid observations.
- False Alarm Probability (FAP) weighting.
- Frequency normalization (ensures ∫P(f) df = 1).
- Implemented in PyTorch (seamless integration into deep learning pipelines).

## Citation
If you found this code useful, please cite our work:
```latex
@inproceedings{lscd2025,
  title     = {LSCD: Lomb–Scargle Conditioned Diffusion for Time-Series Imputation},
  author    = {Elizabeth Fons and Alejandro Sztrajman and Yousef El-Laham and Luciana Ferrer and
               Svitlana Vyetrenko and Manuela Veloso},
  booktitle = {Proc. 42nd International Conference on Machine Learning},
  year      = {2025}
}
```

## License

MIT License
