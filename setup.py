#!/usr/bin/env python3
"""
Setup script for hierarchical time series forecasting package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="hierarchical-time-series-forecasting",
    version="1.0.0",
    author="Time Series Analysis Team",
    author_email="team@timeseries.com",
    description="A comprehensive package for hierarchical time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/hierarchical-time-series-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "streamlit": [
            "streamlit>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hts-forecast=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.yaml"],
    },
    zip_safe=False,
)
