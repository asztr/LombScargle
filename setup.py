from setuptools import setup, find_packages

setup(
    name="LombScargle",
    version="0.1.0",
    description="Batch Lomb-Scargle periodogram in PyTorch",
    author="A. Sztrajman, E. Fons",
    packages=find_packages(),
    install_requires=["torch", "numpy"],
    python_requires=">=3.6",
)
