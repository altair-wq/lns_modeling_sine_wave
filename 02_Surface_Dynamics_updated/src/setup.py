from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fluid_dynamics",
    version="1.0.0",
    author="Wongyung",
    author_email="your.email@stanford.edu",
    description="Fluid surface dynamics simulation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fluid_dynamics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "mpi": ["mpi4py>=3.1.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0"],
    },
)
