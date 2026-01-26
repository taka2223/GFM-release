#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="gfm",
    version="0.1.0",
    description="3D Shape to Vector Set conversion package",
    author="Anonymous",
    author_email="anonymous@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision",
        "numpy",
        "einops",
        "timm",
        "torch-cluster",
        "PyMCubes",
        "trimesh",
        "matplotlib",
        "tqdm",
        "PyYAML",
        "scipy",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "gfm-train-ae=gfm.main_ae:main",
            "gfm-train-dm=gfm.main_class_cond:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
